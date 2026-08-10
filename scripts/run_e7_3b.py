"""Run the registered frozen-3B HLM5 E7 portability experiment."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.certify import calibrate_whitening, certify, dose
from hlm5.e7_contract import (
    DEGREE,
    GATE_THRESHOLD,
    MODEL_HIDDEN_SIZE,
    RESULT_PATH,
    RESULT_SCHEMA,
    SEED,
    TARGET_POOL_COUNT,
    TEMPERATURE,
    atomic_write_json,
    float64_boundary_record,
    load_execution_receipt,
    registered_samples,
    result_contract,
    snapshot_path,
    stable_json_sha256,
    target_pool,
    tensor_sha256,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
    native_head_logits,
)
from hlm5.memory import EditableHLM5Memory
from hlm5.public_adapter import HLM5PreHeadAdapter


TARGET_CHUNK = 40
DOSE_GRID = 120
DOSE_GRID_CHUNK = 8
HEAD_BATCH = 16


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def token_label(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(token).replace("Ġ", " ")


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


@torch.inference_mode()
def build_slope_matrix(
    head64: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = head64.norm(dim=1)
    directions = head64[target_ids] / (norms[target_ids, None] + 1e-12)
    vocab = head64.shape[0]
    target_count = target_ids.numel()
    slopes_cpu = torch.empty((vocab, target_count), dtype=torch.float64)
    for start in range(0, target_count, TARGET_CHUNK):
        end = min(start + TARGET_CHUNK, target_count)
        projected = head64 @ directions[start:end].T
        diagonal = projected[
            target_ids[start:end],
            torch.arange(end - start, device=head64.device),
        ]
        slopes_cpu[:, start:end].copy_((diagonal.unsqueeze(0) - projected).cpu())
        del projected, diagonal
    return directions, slopes_cpu


@torch.inference_mode()
def envelope_for_key(
    base: torch.Tensor,
    slopes_cpu: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    keep_arrays: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    device = base.device
    reach_parts: list[torch.Tensor] = []
    hard_parts: list[torch.Tensor] = []
    lower_parts: list[torch.Tensor] = []
    upper_parts: list[torch.Tensor] = []
    slack_parts: list[torch.Tensor] = []
    risk_counts = {"safe": 0, "narrow": 0, "brittle": 0, "unreachable": 0}
    all_inside = True
    all_positive = True
    minimum_margin: float | None = None
    total_reachable = 0
    hard_count = 0
    interval_only = 0
    finite_slacks: list[float] = []

    for start in range(0, target_ids.numel(), TARGET_CHUNK):
        end = min(start + TARGET_CHUNK, target_ids.numel())
        tids = target_ids[start:end]
        slopes = slopes_cpu[:, start:end].to(device)
        intercepts = base[tids].unsqueeze(0) - base.unsqueeze(1)
        intercepts[tids, torch.arange(end - start, device=device)] = float("inf")
        positive = slopes > 0
        negative = slopes < 0
        hard = ((slopes <= 0) & (intercepts <= 0)).any(dim=0)
        ratios = -intercepts / slopes
        lower = torch.clamp(
            torch.where(positive, ratios, float("-inf")).amax(dim=0),
            min=0.0,
        )
        upper = torch.where(negative, ratios, float("inf")).amin(dim=0)
        reach = (~hard) & (lower < upper)
        slack = upper - lower
        candidate = 1.05 * lower + 1.0
        candidate = torch.where(
            torch.isfinite(upper) & (candidate >= upper),
            0.5 * (lower + upper),
            candidate,
        )
        candidate_margin = (
            intercepts + slopes * candidate.unsqueeze(0)
        ).amin(dim=0)
        inside = reach & (lower < candidate) & (
            ~torch.isfinite(upper) | (candidate < upper)
        )
        positive_margin = reach & (candidate_margin > 0)
        safe = reach & (slack > 5 * candidate)
        narrow = reach & ~safe & (slack > candidate)
        brittle = reach & ~safe & ~narrow

        count = int(reach.sum())
        total_reachable += count
        hard_count += int(hard.sum())
        interval_only += int(((~reach) & (~hard)).sum())
        risk_counts["safe"] += int(safe.sum())
        risk_counts["narrow"] += int(narrow.sum())
        risk_counts["brittle"] += int(brittle.sum())
        risk_counts["unreachable"] += int((~reach).sum())
        if count:
            all_inside = all_inside and bool(inside[reach].all())
            all_positive = all_positive and bool(positive_margin[reach].all())
            chunk_minimum = float(candidate_margin[reach].min())
            minimum_margin = (
                chunk_minimum
                if minimum_margin is None
                else min(minimum_margin, chunk_minimum)
            )
            finite = slack[reach & torch.isfinite(slack)].detach().cpu().tolist()
            finite_slacks.extend(float(value) for value in finite)

        if keep_arrays:
            reach_parts.append(reach.cpu())
            hard_parts.append(hard.cpu())
            lower_parts.append(lower.cpu())
            upper_parts.append(upper.cpu())
            slack_parts.append(slack.cpu())

    finite_slacks.sort()
    upper_median = (
        finite_slacks[len(finite_slacks) // 2] if finite_slacks else None
    )
    summary = {
        "n_reachable": total_reachable,
        "reachable_fraction": total_reachable / target_ids.numel(),
        "risk_counts": risk_counts,
        "median_slack_reachable": upper_median,
        "unreachable_hard_blocker": hard_count,
        "unreachable_interval_only": interval_only,
        "candidate_doses_checked": total_reachable,
        "all_candidate_doses_inside_open_interval": all_inside,
        "all_candidate_margins_positive": all_positive,
        "minimum_candidate_margin": minimum_margin,
    }
    arrays = None
    if keep_arrays:
        arrays = {
            "reach": torch.cat(reach_parts),
            "hard": torch.cat(hard_parts),
            "lower": torch.cat(lower_parts),
            "upper": torch.cat(upper_parts),
            "slack": torch.cat(slack_parts),
        }
    return summary, arrays


@torch.inference_mode()
def scalar_original_check(
    base_cpu: torch.Tensor,
    slopes_cpu: torch.Tensor,
    target_ids_cpu: torch.Tensor,
    vector: dict[str, torch.Tensor],
) -> dict[str, Any]:
    reaches: list[bool] = []
    hards: list[bool] = []
    lowers: list[float] = []
    uppers: list[float] = []
    vocab = base_cpu.numel()
    mask = torch.ones(vocab, dtype=torch.bool)
    for column, target_id in enumerate(target_ids_cpu.tolist()):
        intercept = base_cpu[target_id] - base_cpu
        slope = slopes_cpu[:, column]
        mask.fill_(True)
        mask[target_id] = False
        a = intercept[mask]
        b = slope[mask]
        positive = b > 0
        negative = b < 0
        hard = bool(((b <= 0) & (a <= 0)).any())
        ratio = -a / b
        lower = (
            float(torch.clamp(ratio[positive].max(), min=0.0))
            if bool(positive.any())
            else 0.0
        )
        upper = float(ratio[negative].min()) if bool(negative.any()) else math.inf
        reaches.append((not hard) and lower < upper)
        hards.append(hard)
        lowers.append(lower)
        uppers.append(upper)

    scalar_reach = torch.tensor(reaches, dtype=torch.bool)
    scalar_hard = torch.tensor(hards, dtype=torch.bool)
    scalar_lower = torch.tensor(lowers, dtype=torch.float64)
    scalar_upper = torch.tensor(uppers, dtype=torch.float64)
    finite_upper = torch.isfinite(scalar_upper) & torch.isfinite(vector["upper"])
    lower_difference = (scalar_lower - vector["lower"]).abs()
    upper_difference = (
        (scalar_upper[finite_upper] - vector["upper"][finite_upper]).abs()
        if bool(finite_upper.any())
        else torch.zeros(1, dtype=torch.float64)
    )
    all_match = bool(
        torch.equal(scalar_reach, vector["reach"])
        and torch.equal(scalar_hard, vector["hard"])
        and torch.allclose(
            scalar_lower,
            vector["lower"],
            rtol=1e-12,
            atol=1e-10,
        )
        and torch.equal(torch.isinf(scalar_upper), torch.isinf(vector["upper"]))
        and torch.allclose(
            scalar_upper[finite_upper],
            vector["upper"][finite_upper],
            rtol=1e-12,
            atol=1e-10,
        )
    )
    return {
        "targets_checked": len(reaches),
        "all_match": all_match,
        "maximum_lower_abs_difference": float(lower_difference.max()),
        "maximum_finite_upper_abs_difference": float(upper_difference.max()),
    }


@torch.inference_mode()
def grid_dose(
    intercept: torch.Tensor,
    slope: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    cap = lower + max(50.0, 5.0 * lower)
    high = min(upper, cap) if math.isfinite(upper) else cap
    if not lower < high:
        raise RuntimeError(f"empty registered dose interval: ({lower}, {high})")
    betas = lower + torch.linspace(
        0.0,
        1.0,
        DOSE_GRID + 2,
        dtype=torch.float64,
        device=intercept.device,
    )[1:-1] * (high - lower)
    worst_parts: list[torch.Tensor] = []
    for start in range(0, DOSE_GRID, DOSE_GRID_CHUNK):
        chunk = betas[start : start + DOSE_GRID_CHUNK]
        margins = intercept[:, None] + slope[:, None] * chunk[None, :]
        worst_parts.append(margins.amin(dim=0))
    worst = torch.cat(worst_parts)
    index = int(worst.argmax())
    beta = float(betas[index])
    margin = float(worst[index])
    if not lower < beta or (math.isfinite(upper) and not beta < upper):
        raise RuntimeError("registered grid dose is outside the open interval")
    if margin <= 0:
        raise RuntimeError("registered grid dose has non-positive float64 margin")
    return beta, margin


@torch.inference_mode()
def direct_certified_rows(
    model: Any,
    head64: torch.Tensor,
    original_hidden: torch.Tensor,
    directions: torch.Tensor,
    slopes_cpu: torch.Tensor,
    target_ids: torch.Tensor,
    original_arrays: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    base = head64 @ original_hidden.to(torch.float64)
    accepted: list[dict[str, Any]] = []
    modified: list[torch.Tensor] = []
    for column in torch.nonzero(original_arrays["reach"], as_tuple=False).flatten().tolist():
        target_id = int(target_ids[column])
        intercept = base[target_id] - base
        intercept[target_id] = float("inf")
        slope = slopes_cpu[:, column].to(head64.device)
        lower = float(original_arrays["lower"][column])
        upper = float(original_arrays["upper"][column])
        beta, margin = grid_dose(intercept, slope, lower, upper)
        direction = directions[column]
        edited = (
            original_hidden.to(torch.float64) + beta * direction
        ).to(torch.bfloat16)
        modified.append(edited)
        accepted.append(
            {
                "target_id": target_id,
                "L": lower,
                "U": finite_or_none(upper),
                "beta": beta,
                "float64_margin": margin,
                "inside_open_interval": lower < beta
                and (not math.isfinite(upper) or beta < upper),
            }
        )

    for start in range(0, len(modified), HEAD_BATCH):
        hidden_batch = torch.stack(modified[start : start + HEAD_BATCH])
        logits = native_head_logits(model, hidden_batch)
        for offset, row in enumerate(accepted[start : start + HEAD_BATCH]):
            native = logits[offset]
            target_id = row["target_id"]
            competitor = torch.cat([native[:target_id], native[target_id + 1 :]]).max()
            row["ordinary_head_checked"] = True
            row["ordinary_head_prediction"] = int(native.argmax())
            row["ordinary_head_success"] = int(native.argmax()) == target_id
            row["ordinary_bfloat16_margin"] = float(
                native[target_id].float() - competitor.float()
            )
            row["ordinary_logits_sha256"] = tensor_sha256(native)
    return accepted


def unique_prompts(samples: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    for key, _target, paraphrases in samples["FACTS"]:
        prompts.append(key)
        prompts.extend(paraphrases)
    prompts.extend(samples["NEUTRAL"])
    prompts.extend(samples["WHITEN_TEXT"])
    prompts.extend(samples["NOVEL"])
    prompts.extend(samples["REAL"])
    prompts.extend(samples["GENERIC"])
    return list(dict.fromkeys(prompts))


@torch.inference_mode()
def faithful_evaluation(
    model: Any,
    tokenizer: Any,
    head64: torch.Tensor,
    hidden_by_prompt: dict[str, torch.Tensor],
    samples: dict[str, Any],
) -> dict[str, Any]:
    head_native = model.get_output_embeddings().weight
    facts: list[dict[str, Any]] = []
    for index, (key, target, paraphrases) in enumerate(samples["FACTS"]):
        ids = tokenizer.encode(target, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"registered target is not one token: {target!r} -> {ids}")
        target_id = int(ids[0])
        hidden = hidden_by_prompt[key]
        baseline_logits = native_head_logits(model, hidden.unsqueeze(0))[0]
        prediction = int(baseline_logits.argmax())
        facts.append(
            {
                "fact_index": index,
                "key": key,
                "target": target,
                "target_id": target_id,
                "paraphrases": list(paraphrases),
                "baseline_prediction": prediction,
                "baseline_correct": prediction == target_id,
                "baseline_logits_sha256": tensor_sha256(baseline_logits),
            }
        )

    eligible = [row for row in facts if not row["baseline_correct"]]
    if not eligible:
        key_mean = torch.zeros(MODEL_HIDDEN_SIZE, device=head64.device)
        key_transform = torch.eye(MODEL_HIDDEN_SIZE, device=head64.device)
    else:
        fact_hidden = torch.stack([hidden_by_prompt[row["key"]] for row in eligible])
        whitening_hidden = torch.stack(
            [hidden_by_prompt[text] for text in samples["WHITEN_TEXT"]]
        )
        key_mean, key_transform = calibrate_whitening(
            fact_hidden.float(),
            whitening_hidden.float(),
            floor_frac=0.01,
        )

    accepted: list[dict[str, Any]] = []
    for row in eligible:
        hidden64 = hidden_by_prompt[row["key"]].to(torch.float64)
        certificate = certify(head64, hidden64, row["target_id"])
        row["certificate"] = {
            "reachable": certificate.reachable,
            "hard_blocker": certificate.hard_blocker,
            "blocker_id": certificate.blocker_id,
            "risk": certificate.risk,
            "L": certificate.L,
            "U": finite_or_none(certificate.U),
            "slack": finite_or_none(certificate.slack),
        }
        if not certificate.reachable:
            row["admission"] = "REFUSE"
            continue
        try:
            beta = dose(certificate, grid=DOSE_GRID)
        except ValueError as error:
            row["admission"] = "REFUSE"
            row["refusal_reason"] = str(error)
            continue
        margin = certificate.margin_at(beta)
        if not certificate.L < beta or (
            math.isfinite(certificate.U) and not beta < certificate.U
        ):
            raise RuntimeError("faithful dose escaped its open interval")
        if margin <= 0:
            raise RuntimeError("faithful dose has non-positive float64 margin")
        row["admission"] = "ADMIT"
        row["beta"] = beta
        row["float64_margin"] = margin
        row["direction"] = certificate.v
        accepted.append(row)

    memory = EditableHLM5Memory(
        dim=MODEL_HIDDEN_SIZE,
        memory_size=max(4, len(accepted) + 4),
        degree=DEGREE,
        temperature=TEMPERATURE,
        learnable_memory=False,
    ).to(head64.device)
    slot_by_fact: dict[int, int] = {}
    for row in accepted:
        hidden = hidden_by_prompt[row["key"]].float()
        whitened = (hidden - key_mean) @ key_transform
        direction = row["direction"] / (row["direction"].norm() + 1e-12)
        slot = memory.inject(
            whitened,
            value=direction.float(),
            label=row["target"],
        )
        memory.values[slot].copy_((direction * row["beta"]).float())
        row["slot"] = slot
        slot_by_fact[row["fact_index"]] = slot

    adapter = HLM5PreHeadAdapter(
        memory,
        key_mean,
        key_transform,
        boost=1.0,
        gate_thresh=GATE_THRESHOLD,
    )
    gate_rows: list[dict[str, Any]] = []

    def evaluate_gate(
        prompt: str,
        kind: str,
        fact: dict[str, Any] | None,
    ) -> dict[str, Any]:
        hidden = hidden_by_prompt[prompt]
        adapted_hidden, audit = adapter(hidden.unsqueeze(0), return_audit=True)
        if audit is None:
            raise RuntimeError("E7 adapter did not return required audit evidence")
        baseline_logits = native_head_logits(model, hidden.unsqueeze(0))[0]
        adapted_logits = native_head_logits(model, adapted_hidden)[0]
        selected_score = float(audit.selected_scores[0])
        gate_open = bool(audit.gate_open[0])
        selected_slot = int(audit.selected_slots[0])
        row = {
            "kind": kind,
            "prompt": prompt,
            "fact_index": None if fact is None else fact["fact_index"],
            "actual_score": selected_score,
            "selected_slot": selected_slot,
            "gate_open": gate_open,
            "baseline_logits_sha256": tensor_sha256(baseline_logits),
            "adapted_logits_sha256": tensor_sha256(adapted_logits),
            "logits_bit_identical": torch.equal(baseline_logits, adapted_logits),
            "baseline_prediction": int(baseline_logits.argmax()),
            "adapted_prediction": int(adapted_logits.argmax()),
        }
        if fact is not None:
            target_id = fact["target_id"]
            row["target_id"] = target_id
            row["target_success"] = int(adapted_logits.argmax()) == target_id
            own_slot = slot_by_fact.get(fact["fact_index"])
            row["own_slot"] = own_slot
            row["own_slot_weight"] = (
                float(audit.attention[0, own_slot]) if own_slot is not None else None
            )
            if fact.get("admission") == "ADMIT":
                static = baseline_logits.clone()
                static[target_id] += (
                    fact["beta"]
                    * selected_score
                    * float(head_native[target_id].float().norm())
                )
                row["static_logit_prediction"] = int(static.argmax())
                row["static_logit_success"] = int(static.argmax()) == target_id
        return row

    for fact in facts:
        gate_rows.append(evaluate_gate(fact["key"], "exact", fact))
        for paraphrase in fact["paraphrases"]:
            gate_rows.append(evaluate_gate(paraphrase, "paraphrase", fact))
    for neutral in samples["NEUTRAL"]:
        gate_rows.append(evaluate_gate(neutral, "neutral", None))

    accepted_indices = {row["fact_index"] for row in accepted}
    refused_indices = {
        row["fact_index"]
        for row in eligible
        if row.get("admission") == "REFUSE"
    }
    exact_accepted = [
        row
        for row in gate_rows
        if row["kind"] == "exact" and row["fact_index"] in accepted_indices
    ]
    exact_refused = [
        row
        for row in gate_rows
        if row["kind"] == "exact" and row["fact_index"] in refused_indices
    ]
    paraphrases = [row for row in gate_rows if row["kind"] == "paraphrase"]
    neutrals = [row for row in gate_rows if row["kind"] == "neutral"]
    false_gate_rows = exact_refused + paraphrases + neutrals

    public_facts = []
    for row in facts:
        public_facts.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"direction", "paraphrases"}
            }
        )
    return {
        "registered_fact_count": len(facts),
        "eligible_fact_count": len(eligible),
        "baseline_correct_count": sum(row["baseline_correct"] for row in facts),
        "accepted_count": len(accepted),
        "refused_count": len(refused_indices),
        "facts": public_facts,
        "gate_rows": gate_rows,
        "summary": {
            "accepted_exact_checked": len(exact_accepted),
            "accepted_exact_gate_successes": sum(row["gate_open"] for row in exact_accepted),
            "accepted_exact_full_successes": sum(row["target_success"] for row in exact_accepted),
            "refused_exact_checked": len(exact_refused),
            "paraphrases_checked": len(paraphrases),
            "neutral_checked": len(neutrals),
            "false_gate_applications": sum(row["gate_open"] for row in false_gate_rows),
            "neutral_bit_identical": sum(row["logits_bit_identical"] for row in neutrals),
            "static_logit_successes": sum(
                bool(row.get("static_logit_success")) for row in exact_accepted
            ),
            "own_slot_selected": sum(
                row["selected_slot"] == row.get("own_slot") for row in exact_accepted
            ),
        },
        "whitening": {
            "key_mean_sha256": tensor_sha256(key_mean),
            "key_transform_sha256": tensor_sha256(key_transform),
            "floor_fraction": 0.01,
        },
    }


def main() -> None:
    started = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    load_execution_receipt()
    if not torch.cuda.is_available():
        raise RuntimeError("registered E7 run requires CUDA")
    device = torch.device("cuda", 0)
    snapshot = snapshot_path()
    tokenizer = load_pinned_tokenizer(snapshot)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    samples = registered_samples()
    pool = target_pool(tokenizer)
    if len(pool) != TARGET_POOL_COUNT:
        raise RuntimeError("registered E7 target pool is incomplete")

    prompts = unique_prompts(samples)
    hidden = final_hidden_batch(model, tokenizer, prompts, device, batch_size=16)
    hidden_by_prompt = {prompt: hidden[index] for index, prompt in enumerate(prompts)}
    head64 = model.get_output_embeddings().weight.detach().to(torch.float64)
    target_ids = torch.tensor(pool, device=device, dtype=torch.long)
    directions, slopes_cpu = build_slope_matrix(head64, target_ids)

    keys = (
        [(prompt, "novel") for prompt in samples["NOVEL"]]
        + [(prompt, "real") for prompt in samples["REAL"]]
        + [(prompt, "generic") for prompt in samples["GENERIC"]]
    )
    envelope_rows: list[dict[str, Any]] = []
    original_arrays: dict[str, torch.Tensor] | None = None
    original_base: torch.Tensor | None = None
    for prompt, category in keys:
        base = head64 @ hidden_by_prompt[prompt].to(torch.float64)
        keep = prompt == samples["ORIG_FACT"]
        summary, arrays = envelope_for_key(
            base,
            slopes_cpu,
            target_ids,
            keep_arrays=keep,
        )
        envelope_rows.append(
            {
                "prompt": prompt,
                "category": category,
                "is_original_key": keep,
                "base_argmax": int(base.argmax()),
                **summary,
            }
        )
        if keep:
            original_arrays = arrays
            original_base = base
    if original_arrays is None or original_base is None:
        raise RuntimeError("registered original envelope key was not evaluated")

    scalar_check = scalar_original_check(
        original_base.cpu(),
        slopes_cpu,
        target_ids.cpu(),
        original_arrays,
    )
    original_hidden = hidden_by_prompt[samples["ORIG_FACT"]]
    direct_rows = direct_certified_rows(
        model,
        head64,
        original_hidden,
        directions,
        slopes_cpu,
        target_ids,
        original_arrays,
    )
    faithful = faithful_evaluation(
        model,
        tokenizer,
        head64,
        hidden_by_prompt,
        samples,
    )

    reachable_fractions = [row["reachable_fraction"] for row in envelope_rows]
    median_slacks = [
        row["median_slack_reachable"]
        for row in envelope_rows
        if row["median_slack_reachable"] is not None
    ]
    candidate_checked = sum(row["candidate_doses_checked"] for row in envelope_rows)
    minimum_margins = [
        row["minimum_candidate_margin"]
        for row in envelope_rows
        if row["minimum_candidate_margin"] is not None
    ]
    gate_required_fields = {
        "actual_score",
        "selected_slot",
        "gate_open",
        "baseline_logits_sha256",
        "adapted_logits_sha256",
    }
    validity = {
        "model_invariants_match": invariants["parameter_count"] > 0,
        "keys_checked": len(envelope_rows),
        "target_pool_count": len(pool),
        "envelope_candidates_checked": candidate_checked,
        "all_envelope_candidate_doses_inside": all(
            row["all_candidate_doses_inside_open_interval"] for row in envelope_rows
        ),
        "all_envelope_candidate_margins_positive": all(
            row["all_candidate_margins_positive"] for row in envelope_rows
        ),
        "original_scalar_vector_match": scalar_check["all_match"],
        "all_risk_counts_sum_to_pool": all(
            sum(row["risk_counts"].values()) == TARGET_POOL_COUNT
            for row in envelope_rows
        ),
        "direct_accepted_checked": len(direct_rows),
        "all_direct_doses_inside_with_positive_float64_margin": all(
            row["inside_open_interval"] and row["float64_margin"] > 0
            for row in direct_rows
        ),
        "all_direct_rows_checked_by_ordinary_bfloat16_head": all(
            row.get("ordinary_head_checked") for row in direct_rows
        ),
        "all_faithful_doses_inside_with_positive_float64_margin": all(
            row.get("L", 0) < fact["beta"]
            and (row.get("U") is None or fact["beta"] < row["U"])
            and fact["float64_margin"] > 0
            for fact in faithful["facts"]
            if fact.get("admission") == "ADMIT"
            for row in [fact["certificate"]]
        ),
        "all_gate_rows_have_required_evidence": all(
            gate_required_fields.issubset(row) for row in faithful["gate_rows"]
        ),
        "gate_rows_checked": len(faithful["gate_rows"]),
    }
    result = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE",
        "certificate_contract": result_contract(__file__),
        "model_invariants": invariants,
        "sample_sha256": stable_json_sha256(samples),
        "certificate_boundary_dtypes": float64_boundary_record(
            head=head64,
            hidden=original_hidden.to(torch.float64),
            directions=directions,
            slopes=slopes_cpu,
        ),
        "envelope": {
            "n_keys": len(envelope_rows),
            "pool": len(pool),
            "target_pool_sha256": stable_json_sha256(pool),
            "keys": envelope_rows,
            "original_scalar_vector_check": scalar_check,
            "aggregates": {
                "reachable_fraction": stats(reachable_fractions),
                "median_slack_reachable": stats(median_slacks),
                "minimum_candidate_margin": min(minimum_margins),
            },
        },
        "direct_certified_injection": {
            "original_key": samples["ORIG_FACT"],
            "accepted_count": len(direct_rows),
            "ordinary_head_success_count": sum(
                row["ordinary_head_success"] for row in direct_rows
            ),
            "rows": direct_rows,
        },
        "faithful": faithful,
        "validity": validity,
        "runtime_seconds": time.time() - started,
    }
    atomic_write_json(RESULT_PATH, result)
    print(f"saved registered E7 result -> {RESULT_PATH}")


if __name__ == "__main__":
    main()
