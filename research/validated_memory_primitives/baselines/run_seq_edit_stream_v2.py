"""Controlled HLM5 E5-v2 sequential-memory scale experiment.

This is the implementation companion to:
  D:/HLM2/docs/hlm5-e5v2-sequential-memory-scale-prereg-2026-08-10.md

It uses a frozen HLM5 G3 trunk, a fixed key calibration, head-reachable target
rows, per-fact certified value doses, matched read controls, explicit lifecycle
operations, and staged K stop gates. It never trains model weights or launches
remote compute. A ``--smoke`` run is diagnostic and cannot support a model
verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hlm5_lm import HLM5LM  # noqa: E402
from hlm5_memory import EditableHLM5Memory, unit  # noqa: E402
from run_fineweb_kb_inject import novel_country_names  # noqa: E402


BOS = 2
EOS = 3
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "hlm5_lm_baseline_fineweb_g3_final.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizers" / "fineweb-65536-compat.json"
DEFAULT_OUTPUT = ROOT / "baselines" / "results" / "seq_edit_stream_v2.json"
SMOKE_OUTPUT = ROOT / "baselines" / "results" / "seq_edit_stream_v2_smoke.json"

NEUTRAL_PROMPTS = [
    "The sky is",
    "Two plus two is",
    "Once upon a time",
    "The weather today is",
    "She walked toward the",
    "A careful measurement shows",
    "This document describes",
    "The committee decided that",
    "A small machine can",
    "The river passes through",
    "During the experiment",
    "The final result was",
    "Researchers observed that",
    "In the morning we",
    "The next chapter begins",
    "Nothing in this sentence",
]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    degree: int = 5
    read_mode: str = "dense"
    retrieval_k: int = 8
    shuffled_keys: bool = False


ARM_SPECS = {
    "dense_d5": ArmSpec("dense_d5", "hlm5", read_mode="dense"),
    "masked_d5": ArmSpec("masked_d5", "hlm5", read_mode="support_masked"),
    "topk_d5": ArmSpec("topk_d5", "hlm5", read_mode="top_k", retrieval_k=8),
    "hard_d5": ArmSpec("hard_d5", "hlm5", read_mode="hard_top1"),
    "cosine_kv": ArmSpec("cosine_kv", "cosine_cache", degree=1),
    "shuffled_null": ArmSpec(
        "shuffled_null",
        "hlm5",
        read_mode="support_masked",
        shuffled_keys=True,
    ),
}


def parse_int_csv(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_arm_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in values if item not in ARM_SPECS]
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown arms {unknown}; choose from {sorted(ARM_SPECS)}"
        )
    return values


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start: float, count: int, device: torch.device) -> float:
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / max(1, count)


def make_entity_names(count: int, namespace: str) -> list[str]:
    names: list[str] = []
    if namespace == "fact":
        names.extend(novel_country_names())
    index = 0
    seen = set(names)
    while len(names) < count:
        candidate = f"{namespace.title()}Q{index:06d}ria"
        index += 1
        if candidate not in seen:
            names.append(candidate)
            seen.add(candidate)
    return names[:count]


def encode_prompt(tokenizer: HFTokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text).ids
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if ids and ids[-1] == EOS:
        ids = ids[:-1]
    if not ids:
        raise ValueError(f"tokenizer produced an empty prompt for {text!r}")
    return ids


@torch.inference_mode()
def extract_last_hiddens(
    model: HLM5LM,
    tokenizer: HFTokenizer,
    prompts: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded = [encode_prompt(tokenizer, prompt) for prompt in prompts]
    by_length: dict[int, list[int]] = {}
    for index, ids in enumerate(encoded):
        by_length.setdefault(len(ids), []).append(index)

    output = torch.empty(len(prompts), model.dim, device=device, dtype=torch.float32)
    for length in sorted(by_length):
        indices = by_length[length]
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tokens = torch.tensor(
                [encoded[index] for index in selected],
                device=device,
                dtype=torch.long,
            )
            hidden = model.hidden(tokens)[:, -1].float()
            output[torch.tensor(selected, device=device)] = hidden
    return output


@torch.inference_mode()
def fit_fixed_key_transform(
    calibration_hiddens: torch.Tensor,
    neutral_hiddens: torch.Tensor,
    floor_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    key_mean = calibration_hiddens.mean(dim=0)
    residuals = torch.cat((calibration_hiddens, neutral_hiddens), dim=0) - key_mean
    covariance = residuals.T @ residuals / float(residuals.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    mean_eigenvalue = float(eigenvalues.mean().item())
    floor = max(torch.finfo(torch.float32).eps, floor_fraction * mean_eigenvalue)
    inverse_root = (eigenvalues + floor).clamp_min(torch.finfo(torch.float32).eps).rsqrt()
    transform = eigenvectors @ torch.diag(inverse_root) @ eigenvectors.T
    metadata = {
        "population": int(residuals.shape[0]),
        "floor_fraction": floor_fraction,
        "floor": floor,
        "eigenvalue_min": float(eigenvalues.min().item()),
        "eigenvalue_median": float(eigenvalues.median().item()),
        "eigenvalue_max": float(eigenvalues.max().item()),
    }
    return key_mean, transform, metadata


@torch.inference_mode()
def transformed_keys(
    hiddens: torch.Tensor,
    key_mean: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    return unit((hiddens - key_mean) @ transform, dim=-1)


@torch.inference_mode()
def base_predictions(
    hiddens: torch.Tensor,
    head_weight: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    predictions = []
    for start in range(0, hiddens.shape[0], batch_size):
        hidden = hiddens[start : start + batch_size]
        predictions.append((hidden @ head_weight.T).argmax(dim=-1))
    return torch.cat(predictions)


@torch.inference_mode()
def find_reachable_target_pool(
    head_weight: torch.Tensor,
    needed: int,
    seed: int,
    scan_batch_size: int,
) -> tuple[torch.Tensor, dict]:
    """Find deterministic head rows that are their own raw-dot-product argmax."""
    vocab = head_weight.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(vocab - 4, generator=generator) + 4
    reachable: list[int] = []
    scanned = 0
    for start in range(0, order.numel(), scan_batch_size):
        candidate_cpu = order[start : start + scan_batch_size]
        candidates = candidate_cpu.to(head_weight.device)
        winners = (head_weight[candidates] @ head_weight.T).argmax(dim=-1)
        keep = winners.eq(candidates)
        reachable.extend(candidates[keep].detach().cpu().tolist())
        scanned += int(candidates.numel())
        if len(reachable) >= needed:
            break
    if len(reachable) < needed:
        raise RuntimeError(
            f"found only {len(reachable)} reachable targets after scanning {scanned}; "
            f"need {needed}"
        )
    selected = torch.tensor(reachable[:needed], device=head_weight.device, dtype=torch.long)
    return selected, {
        "needed": needed,
        "found_before_truncation": len(reachable),
        "scanned": scanned,
        "scan_seed": seed,
        "scan_batch_size": scan_batch_size,
    }


def assign_distinct_targets(
    ordered_pool: torch.Tensor,
    base_pred: torch.Tensor,
    count: int,
    offset: int,
) -> torch.Tensor:
    chosen: list[int] = []
    used: set[int] = set()
    cursor = offset
    pool = ordered_pool.detach().cpu().tolist()
    base = base_pred[:count].detach().cpu().tolist()
    while len(chosen) < count:
        if cursor >= len(pool):
            raise RuntimeError("reachable target pool exhausted during assignment")
        candidate = int(pool[cursor])
        current_index = len(chosen)
        cursor += 1
        if candidate in used or candidate == int(base[current_index]):
            continue
        chosen.append(candidate)
        used.add(candidate)
    return torch.tensor(chosen, device=base_pred.device, dtype=torch.long)


def make_replacement_targets(
    initial_targets: torch.Tensor,
    base_pred: torch.Tensor,
) -> torch.Tensor:
    """Rotate a unique pool until every fact gets a new, non-base target."""
    count = initial_targets.numel()
    if count < 2:
        raise ValueError("replacement-target construction needs at least two facts")
    replacements = torch.roll(initial_targets, shifts=1)
    for shift in range(2, count + 1):
        bad = replacements.eq(initial_targets) | replacements.eq(base_pred[:count])
        if not bool(bad.any()):
            return replacements
        candidate = torch.roll(initial_targets, shifts=shift)
        replacements[bad] = candidate[bad]
    bad = replacements.eq(initial_targets) | replacements.eq(base_pred[:count])
    if bool(bad.any()):
        raise RuntimeError("could not construct non-base replacement targets")
    return replacements


@torch.inference_mode()
def certify_value_doses(
    hiddens: torch.Tensor,
    target_ids: torch.Tensor,
    head_weight: torch.Tensor,
    batch_size: int,
    dose_margin: float,
    dose_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return unit target directions and finite per-fact top-1 doses."""
    directions = unit(head_weight[target_ids].float(), dim=-1)
    doses = torch.empty(target_ids.shape[0], device=head_weight.device)
    margins_after = torch.empty_like(doses)
    blockers: list[int] = []

    for start in range(0, target_ids.shape[0], batch_size):
        stop = min(start + batch_size, target_ids.shape[0])
        hidden = hiddens[start:stop]
        target = target_ids[start:stop]
        direction = directions[start:stop]
        base = hidden @ head_weight.T
        delta = direction @ head_weight.T
        row = torch.arange(stop - start, device=head_weight.device)
        base_target = base[row, target]
        delta_target = delta[row, target]
        gap = base - base_target[:, None]
        slope = delta_target[:, None] - delta
        required = gap > 0
        blocked = required & (slope <= 1e-8)
        if bool(blocked.any()):
            blocked_rows = torch.nonzero(blocked.any(dim=-1)).flatten() + start
            blockers.extend(blocked_rows.detach().cpu().tolist())

        ratios = torch.where(
            required & (slope > 1e-8),
            gap / slope.clamp_min(1e-8),
            torch.zeros_like(gap),
        )
        lower = ratios.max(dim=-1).values
        beta = dose_margin * lower + dose_epsilon
        doses[start:stop] = beta

        edited_logits = base + beta[:, None] * delta
        target_logits = edited_logits[row, target]
        edited_logits[row, target] = float("-inf")
        margins_after[start:stop] = target_logits - edited_logits.max(dim=-1).values

    if blockers:
        raise RuntimeError(
            f"reachable-target contract failed for fact indices {blockers[:16]}"
        )
    return directions, doses, {
        "dose_margin": dose_margin,
        "dose_epsilon": dose_epsilon,
        "dose_min": float(doses.min().item()),
        "dose_median": float(doses.median().item()),
        "dose_p90": float(torch.quantile(doses, 0.9).item()),
        "dose_max": float(doses.max().item()),
        "post_margin_min": float(margins_after.min().item()),
        "post_margin_median": float(margins_after.median().item()),
    }


def arm_threshold(spec: ArmSpec, tau_cos: float) -> float:
    return float(tau_cos**spec.degree)


@torch.inference_mode()
def build_memory(
    spec: ArmSpec,
    keys: torch.Tensor,
    values: torch.Tensor,
    doses: torch.Tensor,
    tau_cos: float,
    device: torch.device,
) -> tuple[EditableHLM5Memory, float]:
    threshold = arm_threshold(spec, tau_cos)
    memory = EditableHLM5Memory(
        dim=keys.shape[1],
        memory_size=keys.shape[0] + 8,
        degree=spec.degree,
        temperature=0.10,
        read_mode=spec.read_mode,
        retrieval_k=spec.retrieval_k,
        support_threshold=threshold if spec.read_mode == "support_masked" else None,
    ).to(device)
    synchronize(device)
    start = time.perf_counter()
    count = keys.shape[0]
    for index in range(count):
        key_index = (index + 1) % count if spec.shuffled_keys else index
        memory.inject(
            keys[key_index],
            value=values[index],
            value_scale=float(doses[index].item()),
            label=f"fact:{index}",
        )
    return memory, elapsed_ms(start, count, device)


@torch.inference_mode()
def retrieve_delta(
    memory: EditableHLM5Memory,
    query: torch.Tensor,
    spec: ArmSpec,
    tau_cos: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active = memory.active
    if spec.kind == "cosine_cache":
        scores = unit(query, dim=-1) @ unit(memory.keys, dim=-1).T
        scores = scores.masked_fill(~active, float("-inf"))
        selected_score, selected_slot = scores.max(dim=-1)
        routed = selected_score >= tau_cos
        retrieved = memory.values[selected_slot]
        delta = torch.where(routed[:, None], retrieved, torch.zeros_like(retrieved))
        support_size = (scores >= tau_cos).sum(dim=-1)
        selected_weight = routed.to(query.dtype)
        return delta, routed, selected_slot, selected_weight, support_size

    raw_scores = memory.score(query)
    selected_score, selected_slot = raw_scores.max(dim=-1)
    threshold = arm_threshold(spec, tau_cos)
    routed = selected_score >= threshold
    retrieved, _ = memory(
        query[:, None, :],
        support_threshold=threshold,
    )
    retrieved = retrieved[:, 0]
    delta = selected_score[:, None] * retrieved
    delta = torch.where(routed[:, None], delta, torch.zeros_like(delta))

    selected_scores = memory._selected_scores(raw_scores, threshold)
    has_support = torch.isfinite(selected_scores).any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_support, selected_scores, torch.zeros_like(selected_scores))
    weights = torch.softmax(safe_scores / memory.temperature, dim=-1)
    weights = torch.where(has_support, weights, torch.zeros_like(weights))
    selected_weight = weights.gather(-1, selected_slot[:, None]).squeeze(-1)
    support_size = (raw_scores >= threshold).sum(dim=-1)
    return delta, routed, selected_slot, selected_weight, support_size


@torch.inference_mode()
def evaluate_queries(
    memory: EditableHLM5Memory,
    query: torch.Tensor,
    hidden: torch.Tensor,
    head_weight: torch.Tensor,
    spec: ArmSpec,
    tau_cos: float,
    batch_size: int,
) -> dict[str, torch.Tensor | float | bool]:
    predictions = []
    routes = []
    slots = []
    weights = []
    supports = []
    exact_gate_off = True
    synchronize(hidden.device)
    start = time.perf_counter()
    for offset in range(0, hidden.shape[0], batch_size):
        q = query[offset : offset + batch_size]
        h = hidden[offset : offset + batch_size]
        delta, routed, selected_slot, selected_weight, support_size = retrieve_delta(
            memory,
            q,
            spec,
            tau_cos,
        )
        edited = torch.where(routed[:, None], h + delta, h)
        if bool((~routed).any()):
            exact_gate_off = exact_gate_off and torch.equal(edited[~routed], h[~routed])
        predictions.append((edited @ head_weight.T).argmax(dim=-1))
        routes.append(routed)
        slots.append(selected_slot)
        weights.append(selected_weight)
        supports.append(support_size)
    query_ms = elapsed_ms(start, hidden.shape[0], hidden.device)
    return {
        "prediction": torch.cat(predictions),
        "routed": torch.cat(routes),
        "selected_slot": torch.cat(slots),
        "selected_weight": torch.cat(weights),
        "support_size": torch.cat(supports),
        "gate_off_bit_identical": exact_gate_off,
        "query_ms_per_item": query_ms,
    }


def tensor_rate(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() else 1.0


def metric_summary(result: dict[str, torch.Tensor | float | bool]) -> dict:
    weights = result["selected_weight"]
    supports = result["support_size"].float()
    return {
        "route_rate": tensor_rate(result["routed"]),
        "selected_weight_median": float(weights.median().item()),
        "selected_weight_min": float(weights.min().item()),
        "support_size_median": float(supports.median().item()),
        "support_size_max": int(supports.max().item()),
        "gate_off_bit_identical": bool(result["gate_off_bit_identical"]),
        "query_ms_per_item": float(result["query_ms_per_item"]),
    }


@torch.inference_mode()
def run_arm(
    spec: ArmSpec,
    seed: int,
    fact_hidden: torch.Tensor,
    fact_keys: torch.Tensor,
    base_pred: torch.Tensor,
    target_ids: torch.Tensor,
    target_values: torch.Tensor,
    target_doses: torch.Tensor,
    edit_target_ids: torch.Tensor,
    edit_values: torch.Tensor,
    edit_doses: torch.Tensor,
    off_hidden: torch.Tensor,
    off_keys: torch.Tensor,
    head_weight: torch.Tensor,
    tau_cos: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    count = fact_hidden.shape[0]
    memory, inject_ms = build_memory(
        spec,
        fact_keys,
        target_values,
        target_doses,
        tau_cos,
        device,
    )
    config = memory.retrieval_config()
    config["kind"] = spec.kind
    config["tau_cos"] = tau_cos
    config["tau_score"] = arm_threshold(spec, tau_cos)
    config["shuffled_keys"] = spec.shuffled_keys

    create = evaluate_queries(
        memory,
        fact_keys,
        fact_hidden,
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    create_success = tensor_rate(create["prediction"].eq(target_ids))
    audit_correct = tensor_rate(
        create["selected_slot"].eq(torch.arange(count, device=device))
    )

    if spec.shuffled_keys:
        result = {
            "arm": spec.name,
            "retrieval": config,
            "K": count,
            "seed": seed,
            "create_target_top1": create_success,
            "audit_correct": audit_correct,
            "inject_ms_per_op": inject_ms,
            **metric_summary(create),
        }
        result["gates"] = {
            "shuffled_target_success<=0.01": create_success <= 0.01,
        }
        del memory
        return result

    generator = torch.Generator(device="cpu").manual_seed(10_000 + seed + count)
    operation_order = torch.randperm(count, generator=generator).tolist()
    n_edit = max(1, count // 10)
    n_deactivate = max(1, count // 10)
    n_hard = max(1, count // 20)
    edit_indices = operation_order[:n_edit]
    deactivate_indices = operation_order[n_edit : n_edit + n_deactivate]
    hard_indices = operation_order[
        n_edit + n_deactivate : n_edit + n_deactivate + n_hard
    ]
    reactivate_indices = deactivate_indices[: max(1, len(deactivate_indices) // 2)]
    remain_deactivated = [
        index for index in deactivate_indices if index not in set(reactivate_indices)
    ]

    current_targets = target_ids.clone()
    synchronize(device)
    start = time.perf_counter()
    for index in edit_indices:
        memory.edit_value(
            index,
            edit_values[index],
            value_scale=float(edit_doses[index].item()),
            label=f"fact:{index}:edited",
        )
        current_targets[index] = edit_target_ids[index]
    edit_ms = elapsed_ms(start, len(edit_indices), device)
    edit_eval = evaluate_queries(
        memory,
        fact_keys[edit_indices],
        fact_hidden[edit_indices],
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    edit_success = tensor_rate(
        edit_eval["prediction"].eq(current_targets[edit_indices])
    )

    synchronize(device)
    start = time.perf_counter()
    snapshots = {index: memory.deactivate(index) for index in deactivate_indices}
    deactivate_ms = elapsed_ms(start, len(deactivate_indices), device)
    deactivated_eval = evaluate_queries(
        memory,
        fact_keys[deactivate_indices],
        fact_hidden[deactivate_indices],
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    deactivate_base = tensor_rate(
        deactivated_eval["prediction"].eq(base_pred[deactivate_indices])
    )

    synchronize(device)
    start = time.perf_counter()
    for index in reactivate_indices:
        memory.reactivate(snapshots[index])
    reactivate_ms = elapsed_ms(start, len(reactivate_indices), device)
    reactivated_eval = evaluate_queries(
        memory,
        fact_keys[reactivate_indices],
        fact_hidden[reactivate_indices],
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    reactivate_success = tensor_rate(
        reactivated_eval["prediction"].eq(current_targets[reactivate_indices])
    )

    synchronize(device)
    start = time.perf_counter()
    for index in hard_indices:
        memory.hard_delete(index, rerandomize=False)
    hard_delete_ms = elapsed_ms(start, len(hard_indices), device)
    hard_eval = evaluate_queries(
        memory,
        fact_keys[hard_indices],
        fact_hidden[hard_indices],
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    hard_delete_base = tensor_rate(hard_eval["prediction"].eq(base_pred[hard_indices]))
    hard_overwritten = all(
        not memory.is_occupied(index)
        and int(torch.count_nonzero(memory.keys[index])) == 0
        and int(torch.count_nonzero(memory.values[index])) == 0
        for index in hard_indices
    )

    touched = set(edit_indices) | set(deactivate_indices) | set(hard_indices)
    untouched = [index for index in range(count) if index not in touched]
    untouched_eval = evaluate_queries(
        memory,
        fact_keys[untouched],
        fact_hidden[untouched],
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    untouched_correct = tensor_rate(
        untouched_eval["prediction"].eq(current_targets[untouched])
    )

    if remain_deactivated:
        remain_eval = evaluate_queries(
            memory,
            fact_keys[remain_deactivated],
            fact_hidden[remain_deactivated],
            head_weight,
            spec,
            tau_cos,
            batch_size,
        )
        remain_deactivated_base = tensor_rate(
            remain_eval["prediction"].eq(base_pred[remain_deactivated])
        )
    else:
        remain_deactivated_base = 1.0

    off_support = evaluate_queries(
        memory,
        off_keys,
        off_hidden,
        head_weight,
        spec,
        tau_cos,
        batch_size,
    )
    off_activation = tensor_rate(off_support["routed"])

    active_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (memory.keys, memory.values, memory.alphas, memory.active)
    )
    metrics = {
        "create_target_top1": create_success,
        "edit_target_top1": edit_success,
        "untouched_target_top1": untouched_correct,
        "deactivate_returns_base_top1": deactivate_base,
        "remaining_deactivated_return_base_top1": remain_deactivated_base,
        "reactivate_target_top1": reactivate_success,
        "hard_delete_returns_base_top1": hard_delete_base,
        "hard_delete_overwritten": bool(hard_overwritten),
        "audit_correct": audit_correct,
        "off_support_activation": off_activation,
        "gate_off_bit_identical": bool(off_support["gate_off_bit_identical"]),
    }
    gates = {
        "create>=0.99": create_success >= 0.99,
        "edit>=0.99": edit_success >= 0.99,
        "untouched>=0.99": untouched_correct >= 0.99,
        "deactivate_base==1": deactivate_base == 1.0,
        "remaining_deactivated_base==1": remain_deactivated_base == 1.0,
        "reactivate>=0.99": reactivate_success >= 0.99,
        "hard_delete_base==1": hard_delete_base == 1.0,
        "hard_delete_overwritten==1": bool(hard_overwritten),
        "audit>=0.99": audit_correct >= 0.99,
        "off_support<=0.01": off_activation <= 0.01,
        "gate_off_bit_identical": bool(off_support["gate_off_bit_identical"]),
    }
    gates["all_primary"] = all(gates.values())
    result = {
        "arm": spec.name,
        "retrieval": config,
        "K": count,
        "seed": seed,
        "operations": {
            "edit": len(edit_indices),
            "deactivate": len(deactivate_indices),
            "reactivate": len(reactivate_indices),
            "hard_delete": len(hard_indices),
        },
        "metrics": metrics,
        "create_read": metric_summary(create),
        "latency_ms_per_op": {
            "inject": inject_ms,
            "edit": edit_ms,
            "deactivate": deactivate_ms,
            "reactivate": reactivate_ms,
            "hard_delete": hard_delete_ms,
        },
        "memory_tensor_bytes": active_bytes,
        "gates": gates,
    }
    del memory
    return result


def load_model(checkpoint: Path, device: torch.device) -> tuple[HLM5LM, dict]:
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = HLM5LM(vocab=checkpoint_data["vocab"], **checkpoint_data["cfg"])
    model.load_state_dict(checkpoint_data["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k-values", type=parse_int_csv, default=parse_int_csv("64,256,1024,4096"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv("1,2,3,4,5"))
    parser.add_argument(
        "--arms",
        type=parse_arm_csv,
        default=parse_arm_csv(
            "dense_d5,masked_d5,topk_d5,hard_d5,cosine_kv,shuffled_null"
        ),
    )
    parser.add_argument("--tau-cos", type=float, default=0.95)
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--calibration-count", type=int, default=256)
    parser.add_argument("--floor-fraction", type=float, default=0.01)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--target-scan-batch-size", type=int, default=256)
    parser.add_argument("--dose-batch-size", type=int, default=16)
    parser.add_argument("--dose-margin", type=float, default=1.05)
    parser.add_argument("--dose-epsilon", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not 0 < args.tau_cos <= 1:
        parser.error("--tau-cos must be in (0, 1]")
    if args.retrieval_k < 1:
        parser.error("--retrieval-k must be >= 1")
    if args.smoke:
        args.k_values = [16]
        args.seeds = [1]
    args.k_values = sorted(dict.fromkeys(args.k_values))
    args.seeds = list(dict.fromkeys(args.seeds))
    output_path = args.out or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)

    for name in args.arms:
        if name == "topk_d5":
            ARM_SPECS[name] = ArmSpec(
                "topk_d5",
                "hlm5",
                read_mode="top_k",
                retrieval_k=args.retrieval_k,
            )

    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    print(f"loading frozen checkpoint on {device}: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    tokenizer_hash = sha256_file(tokenizer_path)
    model, checkpoint_data = load_model(checkpoint, device)
    tokenizer = HFTokenizer.from_file(str(tokenizer_path))
    head_weight = model.head.weight.detach().float()

    max_k = max(args.k_values)
    calibration_count = min(args.calibration_count, max(16, args.calibration_count))
    fact_names = make_entity_names(max_k, "fact")
    calibration_names = make_entity_names(calibration_count, "calibration")
    fact_prompts = [f"The capital of {name} is" for name in fact_names]
    calibration_prompts = [f"The capital of {name} is" for name in calibration_names]
    near_miss_prompts = [
        f"The capital city of {name} is" for name in fact_names[: min(64, max_k)]
    ]
    off_prompts = NEUTRAL_PROMPTS + near_miss_prompts

    print(
        f"extracting hiddens: facts={len(fact_prompts)} "
        f"calibration={len(calibration_prompts)} off_support={len(off_prompts)}"
    )
    fact_hidden = extract_last_hiddens(
        model,
        tokenizer,
        fact_prompts,
        device,
        args.hidden_batch_size,
    )
    calibration_hidden = extract_last_hiddens(
        model,
        tokenizer,
        calibration_prompts,
        device,
        args.hidden_batch_size,
    )
    off_hidden = extract_last_hiddens(
        model,
        tokenizer,
        off_prompts,
        device,
        args.hidden_batch_size,
    )
    neutral_count = min(len(NEUTRAL_PROMPTS), off_hidden.shape[0])
    key_mean, key_transform, calibration_meta = fit_fixed_key_transform(
        calibration_hidden,
        off_hidden[:neutral_count],
        args.floor_fraction,
    )
    fact_keys = transformed_keys(fact_hidden, key_mean, key_transform)
    off_keys = transformed_keys(off_hidden, key_mean, key_transform)
    base_pred = base_predictions(fact_hidden, head_weight, args.eval_batch_size)

    target_pool_needed = max_k + 64
    print(f"finding {target_pool_needed} deterministic head-reachable target rows")
    reachable_pool, target_pool_meta = find_reachable_target_pool(
        head_weight,
        target_pool_needed,
        seed=1729,
        scan_batch_size=args.target_scan_batch_size,
    )

    prepared_by_seed: dict[int, dict] = {}
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(reachable_pool.numel(), generator=generator)
        ordered_pool = reachable_pool[permutation.to(device)]
        targets = assign_distinct_targets(ordered_pool, base_pred, max_k, offset=0)
        edit_targets = make_replacement_targets(targets, base_pred)
        target_values, target_doses, target_dose_meta = certify_value_doses(
            fact_hidden,
            targets,
            head_weight,
            args.dose_batch_size,
            args.dose_margin,
            args.dose_epsilon,
        )
        edit_values, edit_doses, edit_dose_meta = certify_value_doses(
            fact_hidden,
            edit_targets,
            head_weight,
            args.dose_batch_size,
            args.dose_margin,
            args.dose_epsilon,
        )
        prepared_by_seed[seed] = {
            "target_ids": targets,
            "target_values": target_values,
            "target_doses": target_doses,
            "edit_target_ids": edit_targets,
            "edit_values": edit_values,
            "edit_doses": edit_doses,
            "target_dose_meta": target_dose_meta,
            "edit_dose_meta": edit_dose_meta,
        }

    report = {
        "schema": "hlm5-e5v2-sequential-memory-scale-v1",
        "registered_prereg": "D:/HLM2/docs/hlm5-e5v2-sequential-memory-scale-prereg-2026-08-10.md",
        "measurement_class": "diagnostic_smoke" if args.smoke else "registered_measurement",
        "command": [sys.executable, *sys.argv],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_hash,
            "checkpoint_cfg": checkpoint_data["cfg"],
            "vocab": int(checkpoint_data["vocab"]),
        },
        "protocol": {
            "k_values_requested": args.k_values,
            "seeds": args.seeds,
            "arms": args.arms,
            "tau_cos": args.tau_cos,
            "retrieval_k": args.retrieval_k,
            "calibration_count": calibration_count,
            "calibration": calibration_meta,
            "target_pool": target_pool_meta,
            "target_pool_ids": reachable_pool.detach().cpu().tolist(),
        },
        "seed_preparation": {
            str(seed): {
                "target_dose": prepared_by_seed[seed]["target_dose_meta"],
                "edit_dose": prepared_by_seed[seed]["edit_dose_meta"],
                "target_ids_sha256": hashlib.sha256(
                    prepared_by_seed[seed]["target_ids"].detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "edit_target_ids_sha256": hashlib.sha256(
                    prepared_by_seed[seed]["edit_target_ids"].detach().cpu().numpy().tobytes()
                ).hexdigest(),
            }
            for seed in args.seeds
        },
        "runs": [],
        "stopped_early": False,
        "stop_reason": None,
    }

    for k_value in args.k_values:
        print(f"\n=== E5-v2 K={k_value} ===")
        k_results = []
        for seed in args.seeds:
            prepared = prepared_by_seed[seed]
            for arm_name in args.arms:
                spec = ARM_SPECS[arm_name]
                print(f"K={k_value} seed={seed} arm={arm_name}")
                result = run_arm(
                    spec,
                    seed,
                    fact_hidden[:k_value],
                    fact_keys[:k_value],
                    base_pred[:k_value],
                    prepared["target_ids"][:k_value],
                    prepared["target_values"][:k_value],
                    prepared["target_doses"][:k_value],
                    prepared["edit_target_ids"][:k_value],
                    prepared["edit_values"][:k_value],
                    prepared["edit_doses"][:k_value],
                    off_hidden,
                    off_keys,
                    head_weight,
                    args.tau_cos,
                    args.eval_batch_size,
                    device,
                )
                report["runs"].append(result)
                k_results.append(result)
                if arm_name == "shuffled_null":
                    summary = result["create_target_top1"]
                else:
                    summary = result["metrics"]["create_target_top1"]
                print(f"  create_target_top1={summary:.3f} gates={result['gates']}")

        masked = [
            result
            for result in k_results
            if result["arm"] == "masked_d5"
        ]
        nulls = [
            result
            for result in k_results
            if result["arm"] == "shuffled_null"
        ]
        masked_pass = bool(masked) and all(
            result["gates"].get("all_primary", False) for result in masked
        )
        null_pass = not nulls or all(
            result["gates"]["shuffled_target_success<=0.01"] for result in nulls
        )
        if not args.smoke and (not masked_pass or not null_pass):
            report["stopped_early"] = True
            report["stop_reason"] = (
                f"registered stop gate failed at K={k_value}: "
                f"masked_pass={masked_pass}, shuffled_null_pass={null_pass}"
            )
            break

    report["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    if args.smoke:
        report["verdict_candidate"] = "SMOKE_ONLY"
    elif report["stopped_early"]:
        first_requested = min(args.k_values)
        report["verdict_candidate"] = (
            "HARNESS_INVALID" if first_requested == 64 and "K=64" in report["stop_reason"]
            else "STOP_GATE_FAILED"
        )
    else:
        report["verdict_candidate"] = "PASS_BARS_PENDING_INTERPRETATION"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {report['measurement_class']} -> {output_path}")
    print(f"verdict candidate: {report['verdict_candidate']}")


if __name__ == "__main__":
    main()
