"""Frozen-3B residual-router training and evaluation runtime for HLM5 E14."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .certify import calibrate_whitening
from .e7_contract import stable_json_sha256, tensor_sha256
from .e7_runtime import native_head_logits
from .e8_runtime import (
    baseline_case_rows,
    direct_admission_rows,
    paired_geometry,
    strongest_competitor,
)
from .e13_runtime import CANDIDATE as FULL_MAHALANOBIS, paired_directions


TRAIN_COUNT = 256
EVAL_CANDIDATE_COUNT = 1_200
ACTIVE_COUNT = 1_024
LOCALITY_CASE_COUNT = 998
LOCALITY_COUNT = 2_994
MEMORY_SIZE = 1_032
HIDDEN_SIZE = 2_048
RANK = 64
ROUTER_SEED = 1_401
STEPS = 400
LEARNING_RATE = 3e-3
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
DRIFT_WEIGHT = 0.10
DEGREE = 5
HLM_TEMPERATURE = 0.10
COSINE_TEMPERATURE = 0.02
GATE_THRESHOLD = 0.95
KEY_FLOOR_FRACTION = 0.01
MODEL_FORWARD_BATCH = 16
HEAD_BATCH = 16
GATE_BATCH = 256
TELEMETRY_STEPS = (0, 1, 50, 100, 200, 400)

IDENTITY = "identity"
HLM = "hlm_degree5"
COSINE = "cosine_control"
ARMS = (IDENTITY, HLM, COSINE)
KINDS = ("exact", "paraphrase", "neighborhood")


def load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("CounterFact data must be a JSON list")
    return value


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def select_train_and_eval(
    records: list[dict[str, Any]],
    tokenizer: Any,
    excluded_prompts: Iterable[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the frozen disjoint train and evaluation fact rows."""
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    required = TRAIN_COUNT + EVAL_CANDIDATE_COUNT
    for record in records:
        rewrite = record["requested_rewrite"]
        exact = str(rewrite["prompt"]).format(rewrite["subject"])
        paraphrases = _unique_strings(record.get("paraphrase_prompts", []))
        target = " " + str(rewrite["target_new"]["str"]).strip()
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not exact or not paraphrases or len(target_ids) != 1:
            continue
        paraphrase = paraphrases[0]
        prompts = (exact, paraphrase)
        if exact == paraphrase:
            continue
        if any(prompt in excluded or prompt in used for prompt in prompts):
            continue
        rows.append(
            {
                "case_id": int(record["case_id"]),
                "exact": exact,
                "paraphrase": paraphrase,
                "target": target,
                "target_id": int(target_ids[0]),
                "relation_id": str(rewrite["relation_id"]),
                "subject": str(rewrite["subject"]),
            }
        )
        used.update(prompts)
        if len(rows) == required:
            break
    if len(rows) != required:
        raise RuntimeError(f"E14 selected {len(rows)} fact rows, expected {required}")
    return rows[:TRAIN_COUNT], rows[TRAIN_COUNT:]


def select_locality(
    records: list[dict[str, Any]],
    excluded_prompts: Iterable[str],
) -> list[dict[str, Any]]:
    """Exhaustively select remaining disjoint three-prompt locality cases."""
    excluded = set(excluded_prompts)
    used: set[str] = set()
    rows: list[dict[str, Any]] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        exact = str(rewrite["prompt"]).format(rewrite["subject"])
        paraphrases = _unique_strings(record.get("paraphrase_prompts", []))
        neighborhoods = _unique_strings(record.get("neighborhood_prompts", []))
        if not exact or not paraphrases or not neighborhoods:
            continue
        prompts = (exact, paraphrases[0], neighborhoods[0])
        if len(set(prompts)) != len(KINDS):
            continue
        if any(prompt in excluded or prompt in used for prompt in prompts):
            continue
        rows.append(
            {
                "case_id": int(record["case_id"]),
                "exact": prompts[0],
                "paraphrase": prompts[1],
                "neighborhood": prompts[2],
            }
        )
        used.update(prompts)
    if len(rows) != LOCALITY_CASE_COUNT:
        raise RuntimeError(
            f"E14 selected {len(rows)} locality cases, expected {LOCALITY_CASE_COUNT}"
        )
    return rows


def fact_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bound_rows = [
        {
            name: row[name]
            for name in ("case_id", "exact", "paraphrase", "target", "target_id")
        }
        for row in rows
    ]
    return {
        "count": len(rows),
        "first_case_id": int(rows[0]["case_id"]),
        "last_case_id": int(rows[-1]["case_id"]),
        "row_sha256": stable_json_sha256(bound_rows),
        "case_id_sha256": stable_json_sha256([row["case_id"] for row in rows]),
        "exact_sha256": stable_json_sha256([row["exact"] for row in rows]),
        "paraphrase_sha256": stable_json_sha256(
            [row["paraphrase"] for row in rows]
        ),
        "target_id_sha256": stable_json_sha256([row["target_id"] for row in rows]),
        "target_sha256": stable_json_sha256([row["target"] for row in rows]),
    }


def locality_prompt_order(rows: list[dict[str, Any]]) -> list[str]:
    return [row[kind] for row in rows for kind in KINDS]


def locality_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "prompt_count": len(rows) * len(KINDS),
        "first_case_id": int(rows[0]["case_id"]),
        "last_case_id": int(rows[-1]["case_id"]),
        "row_sha256": stable_json_sha256(rows),
        "case_id_sha256": stable_json_sha256([row["case_id"] for row in rows]),
        "prompt_order_sha256": stable_json_sha256(locality_prompt_order(rows)),
        "kind_sha256": {
            kind: stable_json_sha256([row[kind] for row in rows]) for kind in KINDS
        },
    }


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class ResidualRouter(nn.Module):
    """Rank-64 residual query/key router with frozen deterministic init."""

    def __init__(self, dim: int = HIDDEN_SIZE, rank: int = RANK) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(ROUTER_SEED)
        a = torch.randn(dim, rank, generator=generator, dtype=torch.float32) * 0.02
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(torch.zeros(rank, dim, dtype=torch.float32))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        query = normalize_rows(query.to(torch.float32))
        return normalize_rows(query + (query @ self.a) @ self.b)


def router_record(router: ResidualRouter) -> dict[str, Any]:
    return {
        "a_sha256": tensor_sha256(router.a.detach()),
        "b_sha256": tensor_sha256(router.b.detach()),
        "a_shape": list(router.a.shape),
        "b_shape": list(router.b.shape),
        "dtype": "float32",
        "parameter_count": sum(parameter.numel() for parameter in router.parameters()),
    }


def whiten(
    hidden: torch.Tensor, mean: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    return normalize_rows((hidden.to(torch.float32) - mean) @ transform)


@torch.no_grad()
def key_operator(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    # The E14 protocol supplies the complete transductive fit set as one matrix.
    empty = hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
    mean, transform = calibrate_whitening(
        hidden.to(torch.float32), empty, floor_frac=KEY_FLOOR_FRACTION
    )
    return mean, transform, {
        "fit_count": int(hidden.shape[0]),
        "floor_fraction": KEY_FLOOR_FRACTION,
        "mean_sha256": tensor_sha256(mean),
        "transform_sha256": tensor_sha256(transform),
    }


def _kernel_logits(query: torch.Tensor, keys: torch.Tensor, arm: str) -> torch.Tensor:
    cosine = torch.relu(query @ keys.T)
    if arm == HLM:
        return cosine.pow(DEGREE) / HLM_TEMPERATURE
    if arm == COSINE:
        return cosine / COSINE_TEMPERATURE
    raise ValueError(f"untrainable E14 arm: {arm}")


def _update_norm(router: ResidualRouter, initial: dict[str, torch.Tensor]) -> float:
    squared = torch.zeros((), device=router.a.device, dtype=torch.float64)
    for name, parameter in router.named_parameters():
        delta = parameter.detach().to(torch.float64) - initial[name].to(
            device=parameter.device, dtype=torch.float64
        )
        squared += delta.square().sum()
    return float(squared.sqrt())


def train_router(
    arm: str,
    train_exact: torch.Tensor,
    train_paraphrase: torch.Tensor,
) -> tuple[ResidualRouter, dict[str, Any]]:
    """Train one matched router without accepting any evaluation tensor."""
    if arm not in (HLM, COSINE):
        raise ValueError("E14 trains only the HLM and cosine-control arms")
    if train_exact.shape != train_paraphrase.shape or train_exact.shape[0] != TRAIN_COUNT:
        raise ValueError("E14 training tensor shape mismatch")
    router = ResidualRouter(train_exact.shape[1]).to(train_exact.device)
    initial = {name: value.detach().cpu().clone() for name, value in router.named_parameters()}
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    labels = torch.arange(TRAIN_COUNT, device=train_exact.device)
    telemetry: list[dict[str, Any]] = []

    def observe(step: int, gradient_norm: float | None) -> None:
        with torch.no_grad():
            keys = router(train_exact)
            queries = router(train_paraphrase)
            logits = _kernel_logits(queries, keys, arm)
            ce = F.cross_entropy(logits, labels)
            drift = (keys - train_exact).square().mean()
            loss = ce + DRIFT_WEIGHT * drift
            telemetry.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "cross_entropy": float(ce),
                    "identity_displacement_mse": float(drift),
                    "gradient_norm": gradient_norm,
                    "parameter_update_norm": _update_norm(router, initial),
                    "own_slot_accuracy": int((logits.argmax(dim=1) == labels).sum()),
                }
            )

    optimizer.zero_grad(set_to_none=True)
    initial_keys = router(train_exact)
    initial_queries = router(train_paraphrase)
    initial_logits = _kernel_logits(initial_queries, initial_keys, arm)
    initial_loss = F.cross_entropy(initial_logits, labels) + DRIFT_WEIGHT * (
        initial_keys - train_exact
    ).square().mean()
    initial_loss.backward()
    initial_grad_norm = float(
        nn.utils.clip_grad_norm_(router.parameters(), GRAD_CLIP)
    )
    if not math.isfinite(initial_grad_norm):
        raise FloatingPointError(f"E14 {arm} produced nonfinite initial gradient")
    optimizer.zero_grad(set_to_none=True)
    observe(0, initial_grad_norm)
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        keys = router(train_exact)
        queries = router(train_paraphrase)
        logits = _kernel_logits(queries, keys, arm)
        ce = F.cross_entropy(logits, labels)
        drift = (keys - train_exact).square().mean()
        loss = ce + DRIFT_WEIGHT * drift
        if not torch.isfinite(loss):
            raise FloatingPointError(f"E14 {arm} produced nonfinite loss at step {step}")
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(router.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"E14 {arm} produced nonfinite gradient")
        optimizer.step()
        if step in TELEMETRY_STEPS:
            observe(step, grad_norm)
    expected_steps = list(TELEMETRY_STEPS)
    if [row["step"] for row in telemetry] != expected_steps:
        raise RuntimeError("E14 training telemetry schedule drift")
    return router.eval(), {"arm": arm, "telemetry": telemetry, "router": router_record(router)}


@torch.no_grad()
def full_mahalanobis_admission(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, list[int]]:
    target_ids = torch.tensor(
        [row["target_id"] for row in cases], device=exact_hidden.device, dtype=torch.long
    )
    head = model.get_output_embeddings().weight.detach()
    directions_by_arm, basis = paired_directions(head, target_ids)
    directions = directions_by_arm[FULL_MAHALANOBIS]
    geometry = paired_geometry(head.to(torch.float64), exact_hidden, target_ids, directions)
    baseline = baseline_case_rows(
        model,
        exact_hidden,
        [
            {"case_id": row["case_id"], "target_id": row["target_id"]}
            for row in cases
        ],
    )
    rows = direct_admission_rows(model, exact_hidden, directions, geometry, baseline)
    selected = [
        index for index, row in enumerate(rows) if row["deployment_admitted"]
    ][:ACTIVE_COUNT]
    return {
        "basis_and_directions": basis,
        "rows": rows,
        "admitted_count": sum(row["deployment_admitted"] for row in rows),
        "selected_count": len(selected),
        "selected_indices": selected,
        "selected_indices_sha256": stable_json_sha256(selected),
        "selected_case_ids": [int(cases[index]["case_id"]) for index in selected],
        "selected_case_ids_sha256": stable_json_sha256(
            [int(cases[index]["case_id"]) for index in selected]
        ),
    }, directions, selected


class RoutedMemory:
    """Frozen tensor implementation of the E14 pre-head memory."""

    def __init__(
        self,
        arm: str,
        router: ResidualRouter | None,
        key_mean: torch.Tensor,
        key_transform: torch.Tensor,
        base_keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        self.arm = arm
        self.router = router
        self.key_mean = key_mean
        self.key_transform = key_transform
        self.base_keys = normalize_rows(base_keys)
        if self.base_keys.shape[0] > MEMORY_SIZE:
            raise ValueError("E14 active keys exceed physical memory")
        routed = self.route(self.base_keys)
        self.keys = torch.zeros(
            MEMORY_SIZE,
            routed.shape[1],
            device=routed.device,
            dtype=routed.dtype,
        )
        self.values = torch.zeros_like(self.keys)
        self.keys[: routed.shape[0]].copy_(routed)
        self.values[: values.shape[0]].copy_(values.to(torch.float32))
        self.active = torch.zeros(MEMORY_SIZE, device=routed.device, dtype=torch.bool)
        self.active[: routed.shape[0]] = True

    def route(self, base: torch.Tensor) -> torch.Tensor:
        if self.router is None:
            return normalize_rows(base)
        return self.router(base)

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.route(whiten(hidden, self.key_mean, self.key_transform))

    def apply(self, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = self.transform(hidden)
        if not bool(self.active.any()):
            count = hidden.shape[0]
            zero = torch.zeros_like(hidden)
            return hidden + zero, {
                "selected_score": torch.zeros(count, device=hidden.device),
                "selected_slot": torch.full(
                    (count,), -1, device=hidden.device, dtype=torch.long
                ),
                "gate_open": torch.zeros(count, device=hidden.device, dtype=torch.bool),
                "delta": zero,
            }
        cosine = torch.relu(query @ self.keys.T)
        selection = cosine if self.arm == COSINE else cosine.pow(DEGREE)
        selection = selection.masked_fill(~self.active[None, :], float("-inf"))
        selected_score, selected_slot = selection.max(dim=1)
        gate_open = torch.isfinite(selected_score) & (selected_score >= GATE_THRESHOLD)
        temperature = COSINE_TEMPERATURE if self.arm == COSINE else HLM_TEMPERATURE
        attention = F.softmax(selection / temperature, dim=1)
        retrieved = attention @ self.values
        # Preserve E13 injection dose for every arm.
        gain = cosine.gather(1, selected_slot[:, None]).squeeze(1).pow(DEGREE)
        gain = torch.where(gate_open, gain, torch.zeros_like(gain))
        delta = (gain[:, None] * retrieved).to(hidden.dtype)
        return hidden + delta, {
            "selected_score": selected_score,
            "selected_slot": selected_slot,
            "gate_open": gate_open,
            "delta": delta,
        }

    def coherence(self) -> float:
        keys = self.keys[self.active]
        overlap = torch.abs(keys @ keys.T)
        overlap.fill_diagonal_(0)
        return float(overlap.max())


@torch.no_grad()
def scan_facts(
    model: Any,
    memory: RoutedMemory,
    cases: list[dict[str, Any]],
    selected: list[int],
    hidden: torch.Tensor,
    kind: str,
) -> dict[str, Any]:
    source = hidden[selected]
    changed, audit = memory.apply(source)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(selected), HEAD_BATCH):
        end = min(start + HEAD_BATCH, len(selected))
        logits = native_head_logits(model, changed[start:end])
        for offset, candidate_index in enumerate(selected[start:end]):
            local = start + offset
            target_id = int(cases[candidate_index]["target_id"])
            row_logits = logits[offset].contiguous()
            competitor_id, competitor_logit = strongest_competitor(row_logits, target_id)
            target_logit = float(row_logits[target_id])
            margin = float(
                torch.tensor(target_logit, dtype=torch.float32)
                - torch.tensor(competitor_logit, dtype=torch.float32)
            )
            delta = audit["delta"][local].detach().cpu().contiguous()
            rows.append(
                {
                    "kind": kind,
                    "case_id": int(cases[candidate_index]["case_id"]),
                    "candidate_index": candidate_index,
                    "own_slot": local,
                    "selected_slot": int(audit["selected_slot"][local]),
                    "selected_score": float(audit["selected_score"][local]),
                    "gate_open": bool(audit["gate_open"][local]),
                    "delta_zero": int(torch.count_nonzero(delta)) == 0,
                    "delta_sha256": tensor_sha256(delta),
                    "target_id": target_id,
                    "target_logit": target_logit,
                    "competitor_id": competitor_id,
                    "competitor_logit": competitor_logit,
                    "target_margin": margin,
                    "prediction": int(row_logits.argmax()),
                    "strict_target_success": int(row_logits.argmax()) == target_id
                    and margin > 0.0,
                    "logits_sha256": tensor_sha256(row_logits.detach().cpu()),
                }
            )
    return {
        "rows": rows,
        "summary": {
            "count": len(rows),
            "gate_open_count": sum(row["gate_open"] for row in rows),
            "own_slot_count": sum(
                row["selected_slot"] == row["own_slot"] for row in rows
            ),
            "strict_target_success_count": sum(
                row["strict_target_success"] for row in rows
            ),
            "nonzero_delta_count": sum(not row["delta_zero"] for row in rows),
            "minimum_margin": min(row["target_margin"] for row in rows),
        },
    }


@torch.no_grad()
def scan_locality(
    memory: RoutedMemory,
    rows: list[dict[str, Any]],
    hidden: torch.Tensor,
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    prompts = locality_prompt_order(rows)
    for start in range(0, len(prompts), GATE_BATCH):
        end = min(start + GATE_BATCH, len(prompts))
        native = hidden[start:end]
        changed, audit = memory.apply(native)
        for offset in range(end - start):
            index = start + offset
            case_index, kind_index = divmod(index, len(KINDS))
            delta = audit["delta"][offset].detach().cpu().contiguous()
            native_row = native[offset].detach().cpu().contiguous()
            changed_row = changed[offset].detach().cpu().contiguous()
            output.append(
                {
                    "pool_index": index,
                    "case_id": int(rows[case_index]["case_id"]),
                    "kind": KINDS[kind_index],
                    "prompt_sha256": stable_json_sha256(prompts[index]),
                    "selected_score": float(audit["selected_score"][offset]),
                    "selected_slot": int(audit["selected_slot"][offset]),
                    "gate_open": bool(audit["gate_open"][offset]),
                    "delta_zero": int(torch.count_nonzero(delta)) == 0,
                    "hidden_bit_identical": torch.equal(native_row, changed_row),
                    "native_hidden_sha256": tensor_sha256(native_row),
                    "adapted_hidden_sha256": tensor_sha256(changed_row),
                }
            )
    kind_summary = {
        kind: {
            "count": sum(row["kind"] == kind for row in output),
            "gate_open_count": sum(
                row["kind"] == kind and row["gate_open"] for row in output
            ),
            "nonzero_delta_count": sum(
                row["kind"] == kind and not row["delta_zero"] for row in output
            ),
            "changed_hidden_count": sum(
                row["kind"] == kind and not row["hidden_bit_identical"]
                for row in output
            ),
        }
        for kind in KINDS
    }
    return {
        "rows": output,
        "summary": {
            "count": len(output),
            "gate_open_count": sum(row["gate_open"] for row in output),
            "nonzero_delta_count": sum(not row["delta_zero"] for row in output),
            "changed_hidden_count": sum(
                not row["hidden_bit_identical"] for row in output
            ),
            "by_kind": kind_summary,
        },
    }


@torch.no_grad()
def rollback_scan(
    memory: RoutedMemory,
    exact_hidden: torch.Tensor,
    locality_hidden: torch.Tensor,
) -> dict[str, Any]:
    memory.active.fill_(False)
    exact_changed, exact_audit = memory.apply(exact_hidden)
    local_changed, local_audit = memory.apply(locality_hidden)
    return {
        "exact_count": int(exact_hidden.shape[0]),
        "locality_count": int(locality_hidden.shape[0]),
        "exact_gate_open_count": int(exact_audit["gate_open"].sum()),
        "locality_gate_open_count": int(local_audit["gate_open"].sum()),
        "exact_bit_identical": torch.equal(exact_hidden, exact_changed),
        "locality_bit_identical": torch.equal(locality_hidden, local_changed),
        "exact_nonzero_delta": int(torch.count_nonzero(exact_audit["delta"])),
        "locality_nonzero_delta": int(torch.count_nonzero(local_audit["delta"])),
    }


def model_parameter_sha256(model: Any) -> str:
    """Stream a cryptographic fingerprint without retaining a second 3B state."""
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def environment_record(device: torch.device) -> dict[str, Any]:
    from .e7b_runtime import environment_record as base_environment_record

    value = base_environment_record(device)
    value["node"] = platform.node()
    return value


def bundle_tensors(
    routers: dict[str, ResidualRouter],
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "hlm_a": routers[HLM].a.detach().cpu().contiguous(),
        "hlm_b": routers[HLM].b.detach().cpu().contiguous(),
        "cosine_a": routers[COSINE].a.detach().cpu().contiguous(),
        "cosine_b": routers[COSINE].b.detach().cpu().contiguous(),
        "key_mean": key_mean.detach().cpu().contiguous(),
        "key_transform": key_transform.detach().cpu().contiguous(),
    }


def bundle_manifest(payload: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        }
        for name, value in payload.items()
    }


def routers_from_bundle(
    payload: dict[str, torch.Tensor], device: torch.device
) -> dict[str, ResidualRouter]:
    routers: dict[str, ResidualRouter] = {}
    for arm, prefix in ((HLM, "hlm"), (COSINE, "cosine")):
        router = ResidualRouter().to(device)
        router.a.data.copy_(payload[f"{prefix}_a"].to(device))
        router.b.data.copy_(payload[f"{prefix}_b"].to(device))
        routers[arm] = router.eval()
    return routers


def training_config() -> dict[str, Any]:
    return {
        "rank": RANK,
        "seed": ROUTER_SEED,
        "steps": STEPS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "drift_weight": DRIFT_WEIGHT,
        "train_count": TRAIN_COUNT,
        "telemetry_steps": list(TELEMETRY_STEPS),
    }


@torch.no_grad()
def evaluate_arms(
    model: Any,
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    paraphrase_hidden: torch.Tensor,
    locality_rows: list[dict[str, Any]],
    locality_hidden: torch.Tensor,
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
    routers: dict[str, ResidualRouter],
) -> dict[str, Any]:
    """Run the frozen E14 direct, route, locality, and rollback measurements."""
    direct, directions, selected = full_mahalanobis_admission(
        model, cases, exact_hidden
    )
    arms: dict[str, Any] = {}
    if len(selected) < ACTIVE_COUNT:
        return {"direct": direct, "arms": arms}
    selected = selected[:ACTIVE_COUNT]
    base_keys = whiten(exact_hidden[selected], key_mean, key_transform)
    values = torch.stack(
        [
            directions[index].to(torch.float32)
            * float(direct["rows"][index]["beta"])
            for index in selected
        ]
    )
    for arm in ARMS:
        router = None if arm == IDENTITY else routers[arm]
        memory = RoutedMemory(
            arm=arm,
            router=router,
            key_mean=key_mean,
            key_transform=key_transform,
            base_keys=base_keys,
            values=values,
        )
        exact = scan_facts(
            model, memory, cases, selected, exact_hidden, "exact"
        )
        paraphrase = scan_facts(
            model, memory, cases, selected, paraphrase_hidden, "paraphrase"
        )
        locality = scan_locality(memory, locality_rows, locality_hidden)
        coherence = memory.coherence()
        memory_record = {
            "physical_slots": MEMORY_SIZE,
            "active_slots": ACTIVE_COUNT,
            "degree": DEGREE,
            "hlm_temperature": HLM_TEMPERATURE,
            "cosine_temperature": (
                COSINE_TEMPERATURE if arm == COSINE else None
            ),
            "gate_threshold": GATE_THRESHOLD,
            "active_mask_sha256": tensor_sha256(memory.active),
            "active_keys_sha256": tensor_sha256(memory.keys[memory.active]),
            "active_values_sha256": tensor_sha256(memory.values[memory.active]),
            "key_coherence": coherence,
            "degree_five_max_crosstalk": coherence**DEGREE,
        }
        rollback = rollback_scan(
            memory, exact_hidden[selected], locality_hidden
        )
        arms[arm] = {
            "router": (
                {"kind": "identity", "parameter_count": 0}
                if router is None
                else router_record(router)
            ),
            "memory": memory_record,
            "exact": exact,
            "paraphrase": paraphrase,
            "locality": locality,
            "rollback": rollback,
        }
    return {"direct": direct, "arms": arms}


def scientific_summary(measurement: dict[str, Any]) -> dict[str, Any]:
    """Compact binding summary used by the formal verdict."""
    direct = measurement["evaluation"]["direct"]
    arms = measurement["evaluation"]["arms"]
    summary: dict[str, Any] = {
        "direct_admitted": int(direct["admitted_count"]),
        "selected_count": int(direct["selected_count"]),
        "trunk_unchanged": measurement["trunk_parameter_sha256_before"]
        == measurement["trunk_parameter_sha256_after"],
        "arms": {},
    }
    for arm, value in arms.items():
        summary["arms"][arm] = {
            "train_correct": (
                None
                if arm == IDENTITY
                else measurement["training"][arm]["telemetry"][-1][
                    "own_slot_accuracy"
                ]
            ),
            "exact_gate": value["exact"]["summary"]["gate_open_count"],
            "exact_own": value["exact"]["summary"]["own_slot_count"],
            "exact_strict": value["exact"]["summary"][
                "strict_target_success_count"
            ],
            "exact_nonzero": value["exact"]["summary"]["nonzero_delta_count"],
            "paraphrase_gate": value["paraphrase"]["summary"]["gate_open_count"],
            "paraphrase_own": value["paraphrase"]["summary"]["own_slot_count"],
            "paraphrase_strict": value["paraphrase"]["summary"][
                "strict_target_success_count"
            ],
            "locality_gate": value["locality"]["summary"]["gate_open_count"],
            "locality_nonzero": value["locality"]["summary"][
                "nonzero_delta_count"
            ],
            "locality_changed": value["locality"]["summary"][
                "changed_hidden_count"
            ],
            "rollback": value["rollback"],
        }
    return summary
