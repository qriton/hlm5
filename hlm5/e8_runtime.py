"""Fresh-case selection and composed HLM5 full-path runtime for E8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .certify import calibrate_whitening
from .e7_contract import stable_json_sha256, tensor_sha256
from .e7_runtime import native_head_logits
from .e7b_runtime import (
    GEOMETRIC_REACHABLE,
    NATIVE_ADMIT,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    finite_or_none,
    grid_dose,
    native_decision,
)
from .memory import EditableHLM5Memory
from .public_adapter import HLM5PreHeadAdapter


CASE_START = 20_000
CASE_COUNT = 64
PARAPHRASES_PER_CASE = 2
NEIGHBORHOODS_PER_CASE = 2
MEMORY_SIZE = 72
DEGREE = 5
TEMPERATURE = 0.10
GATE_THRESHOLD = 0.95
HEAD_BATCH = 16
PAIR_CHUNK = 16
BASELINE_CORRECT = "BASELINE_CORRECT"


def select_cases(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Select the exact registered E8 cases using data/tokenizer structure only."""
    selected: list[dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        if case_id < CASE_START:
            continue
        rewrite = record["requested_rewrite"]
        target = " " + str(rewrite["target_new"]["str"]).strip()
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        prompt = str(rewrite["prompt"]).format(rewrite["subject"])
        paraphrases = list(dict.fromkeys(record.get("paraphrase_prompts", [])))
        neighborhoods = list(dict.fromkeys(record.get("neighborhood_prompts", [])))
        if (
            len(target_ids) != 1
            or not prompt
            or len(paraphrases) < PARAPHRASES_PER_CASE
            or len(neighborhoods) < NEIGHBORHOODS_PER_CASE
        ):
            continue
        selected.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "target": target,
                "target_id": int(target_ids[0]),
                "paraphrases": paraphrases[:PARAPHRASES_PER_CASE],
                "neighborhood_prompts": neighborhoods[:NEIGHBORHOODS_PER_CASE],
                "relation_id": str(rewrite["relation_id"]),
                "subject": str(rewrite["subject"]),
            }
        )
        if len(selected) == CASE_COUNT:
            break
    if len(selected) != CASE_COUNT:
        raise RuntimeError(f"E8 selected {len(selected)} cases, expected {CASE_COUNT}")
    return selected


def load_selected_cases(data_path: Path, tokenizer: Any) -> list[dict[str, Any]]:
    import json

    value = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("CounterFact data must be a JSON list")
    return select_cases(value, tokenizer)


def prompt_order(
    cases: list[dict[str, Any]],
    neutral_prompts: list[str],
    whitening_prompts: list[str],
) -> list[str]:
    prompts: list[str] = []
    for case in cases:
        prompts.append(case["prompt"])
        prompts.extend(case["paraphrases"])
        prompts.extend(case["neighborhood_prompts"])
    prompts.extend(neutral_prompts)
    prompts.extend(whitening_prompts)
    return list(dict.fromkeys(prompts))


def strongest_competitor(logits: torch.Tensor, target_id: int) -> tuple[int, float]:
    competitors = logits.clone()
    competitors[target_id] = float("-inf")
    value = competitors.max()
    ids = torch.nonzero(competitors == value, as_tuple=False).flatten()
    return int(ids[0]), float(value)


@torch.inference_mode()
def paired_geometry(
    head64: torch.Tensor,
    hidden_native: torch.Tensor,
    target_ids: torch.Tensor,
    directions: torch.Tensor,
) -> list[dict[str, Any]]:
    """Exact geometry for paired (hidden, target, direction) rows."""
    if head64.dtype != torch.float64 or directions.dtype != torch.float64:
        raise TypeError("E8 exact geometry requires float64 head and directions")
    if hidden_native.dtype != torch.bfloat16 or target_ids.dtype != torch.long:
        raise TypeError("E8 requires BF16 hiddens and long target ids")
    count = target_ids.numel()
    if hidden_native.shape != (count, head64.shape[1]) or directions.shape != (
        count,
        head64.shape[1],
    ):
        raise ValueError("E8 paired geometry shape mismatch")

    rows: list[dict[str, Any]] = []
    for start in range(0, count, PAIR_CHUNK):
        end = min(start + PAIR_CHUNK, count)
        tids = target_ids[start:end]
        local = torch.arange(end - start, device=head64.device)
        base = head64 @ hidden_native[start:end].to(torch.float64).T
        projected = head64 @ directions[start:end].T
        intercepts = base[tids, local].unsqueeze(0) - base
        slopes = projected[tids, local].unsqueeze(0) - projected
        intercepts[tids, local] = float("inf")
        positive = slopes > 0
        negative = slopes < 0
        hard_matrix = (slopes <= 0) & (intercepts <= 0)
        hard = hard_matrix.any(dim=0)
        ratios = -intercepts / slopes
        lower = torch.clamp(
            torch.where(positive, ratios, float("-inf")).amax(dim=0), min=0.0
        )
        upper = torch.where(negative, ratios, float("inf")).amin(dim=0)
        if not torch.isfinite(lower).all():
            raise RuntimeError("E8 lower bound is nonfinite")
        if torch.isnan(upper).any() or torch.isneginf(upper).any():
            raise RuntimeError("E8 upper bound has an invalid nonfinite value")

        for offset, index in enumerate(range(start, end)):
            lower_value = float(lower[offset])
            upper_value = float(upper[offset])
            row: dict[str, Any] = {
                "case_index": index,
                "target_id": int(target_ids[index]),
                "L": lower_value,
                "U": finite_or_none(upper_value),
                "hard_blocker": bool(hard[offset]),
                "hard_blocker_id": None,
                "beta": None,
                "float64_margin": None,
            }
            if row["hard_blocker"]:
                blockers = torch.nonzero(
                    hard_matrix[:, offset], as_tuple=False
                ).flatten()
                row["hard_blocker_id"] = int(blockers[0])
                row["geometry_class"] = REFUSE_HARD
                row["decision"] = REFUSE_HARD
            elif not lower_value < upper_value:
                row["geometry_class"] = REFUSE_INTERVAL
                row["decision"] = REFUSE_INTERVAL
            else:
                beta, margin = grid_dose(
                    intercepts[:, offset],
                    slopes[:, offset],
                    lower_value,
                    upper_value,
                )
                row.update(
                    {
                        "geometry_class": GEOMETRIC_REACHABLE,
                        "decision": None,
                        "beta": beta,
                        "float64_margin": margin,
                    }
                )
            rows.append(row)
    if len(rows) != count:
        raise RuntimeError("E8 did not classify every paired row")
    return rows


@torch.inference_mode()
def baseline_case_rows(
    model: Any,
    hidden_native: torch.Tensor,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(cases), HEAD_BATCH):
        logits = native_head_logits(model, hidden_native[start : start + HEAD_BATCH])
        for offset, case in enumerate(cases[start : start + HEAD_BATCH]):
            native = logits[offset].contiguous()
            target_id = int(case["target_id"])
            competitor_id, competitor_logit = strongest_competitor(native, target_id)
            target_logit = float(native[target_id])
            prediction = int(native.argmax())
            rows.append(
                {
                    "case_index": start + offset,
                    "case_id": int(case["case_id"]),
                    "target_id": target_id,
                    "prediction": prediction,
                    "baseline_correct": prediction == target_id,
                    "target_logit": target_logit,
                    "competitor_id": competitor_id,
                    "competitor_logit": competitor_logit,
                    "target_margin": float(
                        torch.tensor(target_logit, dtype=torch.float32)
                        - torch.tensor(competitor_logit, dtype=torch.float32)
                    ),
                    "logits_sha256": tensor_sha256(native),
                    "logits_dtype": "bfloat16",
                    "logits_shape": list(native.shape),
                }
            )
    return rows


@torch.inference_mode()
def direct_admission_rows(
    model: Any,
    hidden_native: torch.Tensor,
    directions: torch.Tensor,
    geometry: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = [dict(row) for row in geometry]
    reachable_indices = [
        index
        for index, row in enumerate(output)
        if row["geometry_class"] == GEOMETRIC_REACHABLE
    ]
    for start in range(0, len(reachable_indices), HEAD_BATCH):
        indices = reachable_indices[start : start + HEAD_BATCH]
        edited = torch.stack(
            [
                (
                    hidden_native[index].to(torch.float64)
                    + float(output[index]["beta"]) * directions[index]
                ).to(torch.bfloat16)
                for index in indices
            ]
        )
        logits = native_head_logits(model, edited)
        for offset, index in enumerate(indices):
            native = logits[offset].contiguous()
            target_id = int(output[index]["target_id"])
            competitor_id, competitor_logit = strongest_competitor(native, target_id)
            target_logit = float(native[target_id])
            margin = float(
                torch.tensor(target_logit, dtype=torch.float32)
                - torch.tensor(competitor_logit, dtype=torch.float32)
            )
            prediction = int(native.argmax())
            output[index].update(
                {
                    "native_checked": True,
                    "native_target_logit": target_logit,
                    "native_competitor_id": competitor_id,
                    "native_competitor_logit": competitor_logit,
                    "native_margin": margin,
                    "native_prediction": prediction,
                    "native_logits_sha256": tensor_sha256(native),
                    "native_logits_dtype": "bfloat16",
                    "native_logits_shape": list(native.shape),
                    "native_batch_start": start,
                    "native_batch_size": len(indices),
                    "native_batch_offset": offset,
                    "decision": native_decision(
                        target_id=target_id,
                        prediction=prediction,
                        native_margin=margin,
                    ),
                }
            )
    for index, row in enumerate(output):
        baseline = baseline_rows[index]
        row.update(
            {
                "case_id": baseline["case_id"],
                "baseline_prediction": baseline["prediction"],
                "baseline_correct": baseline["baseline_correct"],
                "baseline_logits_sha256": baseline["logits_sha256"],
                "eligible": not baseline["baseline_correct"],
            }
        )
        row["deployment_admitted"] = bool(
            row["eligible"] and row.get("decision") == NATIVE_ADMIT
        )
        if baseline["baseline_correct"]:
            row["deployment_refusal_reason"] = BASELINE_CORRECT
        elif not row["deployment_admitted"]:
            row["deployment_refusal_reason"] = row.get("decision")
        else:
            row["deployment_refusal_reason"] = None
    return output


def direct_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    reachable = [row for row in rows if row["geometry_class"] == GEOMETRIC_REACHABLE]
    native_strict = [row for row in reachable if row.get("decision") == NATIVE_ADMIT]
    admitted = [row for row in rows if row["deployment_admitted"]]
    return {
        "case_count": len(rows),
        "eligible_count": len(eligible),
        "baseline_correct_count": len(rows) - len(eligible),
        "geometrically_reachable": len(reachable),
        "native_strict_wins": len(native_strict),
        "deployment_admitted": len(admitted),
        "admitted_case_ids": [int(row["case_id"]) for row in admitted],
        "decision_counts": {
            decision: sum(row.get("decision") == decision for row in rows)
            for decision in (
                NATIVE_ADMIT,
                REFUSE_HARD,
                REFUSE_INTERVAL,
                "REFUSE_NATIVE_TIE",
                "REFUSE_NATIVE_WRONG_ARGMAX",
            )
        },
    }


def memory_receipt(
    memory: EditableHLM5Memory,
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
    slot_by_case: dict[int, int],
) -> dict[str, Any]:
    active_slots = torch.nonzero(memory.active, as_tuple=False).flatten()
    return {
        "memory_size": memory.memory_size,
        "degree": memory.degree,
        "temperature": memory.temperature,
        "active_count": int(active_slots.numel()),
        "active_slots": [int(slot) for slot in active_slots],
        "active_alphas": [float(memory.alphas[slot]) for slot in active_slots],
        "active_mask_sha256": tensor_sha256(memory.active),
        "active_mask_dtype": str(memory.active.dtype).removeprefix("torch."),
        "active_mask_shape": list(memory.active.shape),
        "active_keys_sha256": tensor_sha256(memory.keys[active_slots].contiguous()),
        "active_keys_shape": list(memory.keys[active_slots].shape),
        "active_values_sha256": tensor_sha256(memory.values[active_slots].contiguous()),
        "active_values_shape": list(memory.values[active_slots].shape),
        "active_alphas_sha256": tensor_sha256(memory.alphas[active_slots].contiguous()),
        "active_alphas_shape": list(memory.alphas[active_slots].shape),
        "active_labels": [memory.labels[int(slot)] for slot in active_slots],
        "slot_by_case": {str(key): value for key, value in slot_by_case.items()},
        "key_mean_sha256": tensor_sha256(key_mean),
        "key_transform_sha256": tensor_sha256(key_transform),
        "key_dtype": str(memory.keys.dtype).removeprefix("torch."),
        "value_dtype": str(memory.values.dtype).removeprefix("torch."),
    }


@torch.inference_mode()
def build_memory(
    cases: list[dict[str, Any]],
    exact_hidden: torch.Tensor,
    direct_rows: list[dict[str, Any]],
    directions: torch.Tensor,
    key_mean: torch.Tensor,
    key_transform: torch.Tensor,
) -> tuple[EditableHLM5Memory, dict[int, int], dict[str, Any]]:
    memory = EditableHLM5Memory(
        dim=exact_hidden.shape[1],
        memory_size=MEMORY_SIZE,
        degree=DEGREE,
        temperature=TEMPERATURE,
        learnable_memory=False,
    ).to(exact_hidden.device)
    slot_by_case: dict[int, int] = {}
    for index, row in enumerate(direct_rows):
        if not row["deployment_admitted"]:
            continue
        whitened = (exact_hidden[index].float() - key_mean) @ key_transform
        slot = memory.inject(
            whitened,
            value=directions[index].float(),
            label=str(cases[index]["target"]),
        )
        memory.values[slot].copy_((directions[index] * float(row["beta"])).float())
        slot_by_case[index] = slot
    return (
        memory,
        slot_by_case,
        memory_receipt(memory, key_mean, key_transform, slot_by_case),
    )


def gate_specs(
    cases: list[dict[str, Any]], neutral_prompts: list[str]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        specs.append(
            {
                "kind": "exact",
                "prompt": case["prompt"],
                "case_index": index,
                "case_id": case["case_id"],
                "target_id": case["target_id"],
            }
        )
        for prompt in case["paraphrases"]:
            specs.append(
                {
                    "kind": "paraphrase",
                    "prompt": prompt,
                    "case_index": index,
                    "case_id": case["case_id"],
                    "target_id": case["target_id"],
                }
            )
        for prompt in case["neighborhood_prompts"]:
            specs.append(
                {
                    "kind": "neighborhood",
                    "prompt": prompt,
                    "case_index": index,
                    "case_id": case["case_id"],
                    "target_id": case["target_id"],
                }
            )
    for prompt in neutral_prompts:
        specs.append(
            {
                "kind": "neutral",
                "prompt": prompt,
                "case_index": None,
                "case_id": None,
                "target_id": None,
            }
        )
    return specs


@torch.inference_mode()
def evaluate_memory_arm(
    model: Any,
    adapter: HLM5PreHeadAdapter,
    hidden_by_prompt: dict[str, torch.Tensor],
    cases: list[dict[str, Any]],
    neutral_prompts: list[str],
    direct_rows: list[dict[str, Any]],
    slot_by_case: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = gate_specs(cases, neutral_prompts)
    hiddens = torch.stack([hidden_by_prompt[spec["prompt"]] for spec in specs])
    adapted, audit = adapter(hiddens, return_audit=True)
    if audit is None:
        raise RuntimeError("E8 adapter omitted audit evidence")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(specs), HEAD_BATCH):
        end = min(start + HEAD_BATCH, len(specs))
        baseline_logits = native_head_logits(model, hiddens[start:end])
        adapted_logits = native_head_logits(model, adapted[start:end])
        for offset, spec in enumerate(specs[start:end]):
            index = start + offset
            baseline = baseline_logits[offset].contiguous()
            changed = adapted_logits[offset].contiguous()
            gate_open = bool(audit.gate_open[index])
            selected_slot = int(audit.selected_slots[index])
            case_index = spec["case_index"]
            own_slot = None if case_index is None else slot_by_case.get(case_index)
            row = {
                **spec,
                "actual_score": float(audit.selected_scores[index]),
                "selected_slot": selected_slot,
                "own_slot": own_slot,
                "own_slot_weight": (
                    None
                    if own_slot is None
                    else float(audit.attention[index, own_slot])
                ),
                "gate_open": gate_open,
                "delta_sha256": tensor_sha256(audit.delta[index].contiguous()),
                "delta_zero": int(torch.count_nonzero(audit.delta[index])) == 0,
                "baseline_logits_sha256": tensor_sha256(baseline),
                "adapted_logits_sha256": tensor_sha256(changed),
                "logits_bit_identical": torch.equal(baseline, changed),
                "baseline_prediction": int(baseline.argmax()),
                "adapted_prediction": int(changed.argmax()),
                "logits_dtype": "bfloat16",
                "logits_shape": list(changed.shape),
                "head_batch_start": start,
                "head_batch_size": end - start,
                "head_batch_offset": offset,
            }
            if case_index is not None:
                target_id = int(spec["target_id"])
                competitor_id, competitor_logit = strongest_competitor(
                    changed, target_id
                )
                target_logit = float(changed[target_id])
                target_margin = float(
                    torch.tensor(target_logit, dtype=torch.float32)
                    - torch.tensor(competitor_logit, dtype=torch.float32)
                )
                row.update(
                    {
                        "target_logit": target_logit,
                        "competitor_id": competitor_id,
                        "competitor_logit": competitor_logit,
                        "target_margin": target_margin,
                        "target_success_strict": (
                            int(changed.argmax()) == target_id and target_margin > 0
                        ),
                        "deployment_admitted": direct_rows[case_index][
                            "deployment_admitted"
                        ],
                    }
                )
            rows.append(row)

    admitted_exact = [
        row for row in rows if row["kind"] == "exact" and row["deployment_admitted"]
    ]
    off_support = [
        row
        for row in rows
        if row["kind"] != "exact" or not row.get("deployment_admitted", False)
    ]
    closed = [row for row in rows if not row["gate_open"]]
    summary = {
        "gate_row_count": len(rows),
        "admitted_exact_checked": len(admitted_exact),
        "admitted_exact_gate_open": sum(row["gate_open"] for row in admitted_exact),
        "admitted_exact_own_slot": sum(
            row["selected_slot"] == row["own_slot"] for row in admitted_exact
        ),
        "admitted_exact_target_success": sum(
            row["target_success_strict"] for row in admitted_exact
        ),
        "off_support_checked": len(off_support),
        "false_gate_applications": sum(row["gate_open"] for row in off_support),
        "gate_closed_checked": len(closed),
        "gate_closed_bit_identical": sum(row["logits_bit_identical"] for row in closed),
        "paraphrases_checked": sum(row["kind"] == "paraphrase" for row in rows),
        "neighborhoods_checked": sum(row["kind"] == "neighborhood" for row in rows),
        "neutrals_checked": sum(row["kind"] == "neutral" for row in rows),
    }
    return rows, summary


@torch.inference_mode()
def rollback_memory(
    model: Any,
    memory: EditableHLM5Memory,
    adapter: HLM5PreHeadAdapter,
    prompt_list: list[str],
    hidden_by_prompt: dict[str, torch.Tensor],
    active_slots: list[int],
) -> dict[str, Any]:
    for slot in active_slots:
        memory.remove(slot)
    hiddens = torch.stack([hidden_by_prompt[prompt] for prompt in prompt_list])
    adapted, audit = adapter(hiddens, return_audit=True)
    if audit is None:
        raise RuntimeError("E8 rollback omitted audit evidence")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompt_list), HEAD_BATCH):
        end = min(start + HEAD_BATCH, len(prompt_list))
        baseline_logits = native_head_logits(model, hiddens[start:end])
        adapted_logits = native_head_logits(model, adapted[start:end])
        for offset, prompt in enumerate(prompt_list[start:end]):
            index = start + offset
            baseline = baseline_logits[offset].contiguous()
            changed = adapted_logits[offset].contiguous()
            rows.append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "gate_open": bool(audit.gate_open[index]),
                    "selected_slot": int(audit.selected_slots[index]),
                    "delta_sha256": tensor_sha256(audit.delta[index].contiguous()),
                    "delta_zero": int(torch.count_nonzero(audit.delta[index])) == 0,
                    "baseline_logits_sha256": tensor_sha256(baseline),
                    "adapted_logits_sha256": tensor_sha256(changed),
                    "logits_bit_identical": torch.equal(baseline, changed),
                    "head_batch_start": start,
                    "head_batch_size": end - start,
                    "head_batch_offset": offset,
                }
            )
    post = {
        "active_count": int(memory.active.sum()),
        "active_mask_sha256": tensor_sha256(memory.active),
        "removed_slots": active_slots,
        "removed_alphas_zero": all(
            float(memory.alphas[slot]) == 0 for slot in active_slots
        ),
        "removed_labels_none": all(
            memory.labels[slot] is None for slot in active_slots
        ),
    }
    summary = {
        "prompt_count": len(rows),
        "gate_open_count": sum(row["gate_open"] for row in rows),
        "delta_zero_count": sum(row["delta_zero"] for row in rows),
        "bit_identical_count": sum(row["logits_bit_identical"] for row in rows),
    }
    return {"post_remove_memory": post, "rows": rows, "summary": summary}


@torch.inference_mode()
def full_path_measurement(
    model: Any,
    cases: list[dict[str, Any]],
    prompt_list: list[str],
    hidden_by_prompt: dict[str, torch.Tensor],
    neutral_prompts: list[str],
    whitening_prompts: list[str],
    raw_directions: torch.Tensor,
    candidate_directions: torch.Tensor,
) -> dict[str, Any]:
    exact_hidden = torch.stack([hidden_by_prompt[case["prompt"]] for case in cases])
    whitening_hidden = torch.stack(
        [hidden_by_prompt[prompt] for prompt in whitening_prompts]
    )
    target_ids = torch.tensor(
        [case["target_id"] for case in cases],
        device=exact_hidden.device,
        dtype=torch.long,
    )
    head64 = model.get_output_embeddings().weight.detach().to(torch.float64)
    baseline = baseline_case_rows(model, exact_hidden, cases)
    key_mean, key_transform = calibrate_whitening(
        exact_hidden.float(), whitening_hidden.float(), floor_frac=0.01
    )
    key_record = {
        "mean_sha256": tensor_sha256(key_mean),
        "transform_sha256": tensor_sha256(key_transform),
        "mean_dtype": str(key_mean.dtype).removeprefix("torch."),
        "mean_shape": list(key_mean.shape),
        "transform_dtype": str(key_transform.dtype).removeprefix("torch."),
        "transform_shape": list(key_transform.shape),
        "floor_fraction": 0.01,
        "exact_key_count": len(cases),
        "whitening_prompt_count": len(whitening_prompts),
    }

    arms: dict[str, Any] = {}
    arm_objects: dict[str, tuple[EditableHLM5Memory, HLM5PreHeadAdapter]] = {}
    for name, directions in (
        ("raw", raw_directions),
        ("candidate", candidate_directions),
    ):
        geometry = paired_geometry(head64, exact_hidden, target_ids, directions)
        direct = direct_admission_rows(
            model, exact_hidden, directions, geometry, baseline
        )
        memory, slot_by_case, receipt = build_memory(
            cases,
            exact_hidden,
            direct,
            directions,
            key_mean,
            key_transform,
        )
        adapter = HLM5PreHeadAdapter(
            memory,
            key_mean,
            key_transform,
            boost=1.0,
            gate_thresh=GATE_THRESHOLD,
        )
        gate_rows, gate_summary = evaluate_memory_arm(
            model,
            adapter,
            hidden_by_prompt,
            cases,
            neutral_prompts,
            direct,
            slot_by_case,
        )
        arms[name] = {
            "direct_rows": direct,
            "direct_summary": direct_summary(direct),
            "memory_receipt": receipt,
            "gate_rows": gate_rows,
            "gate_summary": gate_summary,
        }
        arm_objects[name] = (memory, adapter)

    candidate_memory, candidate_adapter = arm_objects["candidate"]
    candidate_slots = arms["candidate"]["memory_receipt"]["active_slots"]
    rollback = rollback_memory(
        model,
        candidate_memory,
        candidate_adapter,
        prompt_list,
        hidden_by_prompt,
        candidate_slots,
    )
    raw_ids = set(arms["raw"]["direct_summary"]["admitted_case_ids"])
    candidate_ids = set(arms["candidate"]["direct_summary"]["admitted_case_ids"])
    return {
        "cases": cases,
        "case_sha256": stable_json_sha256(cases),
        "prompt_order": prompt_list,
        "prompt_order_sha256": stable_json_sha256(prompt_list),
        "baseline_rows": baseline,
        "key_whitening": key_record,
        "arms": arms,
        "rollback": rollback,
        "summary": {
            "case_count": len(cases),
            "eligible_count": arms["candidate"]["direct_summary"]["eligible_count"],
            "raw_admitted": len(raw_ids),
            "candidate_admitted": len(candidate_ids),
            "candidate_gain": len(candidate_ids) - len(raw_ids),
            "candidate_lost_raw_case_ids": sorted(raw_ids - candidate_ids),
            "candidate_lost_raw_count": len(raw_ids - candidate_ids),
            "candidate_gate": arms["candidate"]["gate_summary"],
            "rollback": rollback["summary"],
        },
    }


__all__ = [
    "BASELINE_CORRECT",
    "CASE_COUNT",
    "CASE_START",
    "DEGREE",
    "GATE_THRESHOLD",
    "HEAD_BATCH",
    "MEMORY_SIZE",
    "NEIGHBORHOODS_PER_CASE",
    "PARAPHRASES_PER_CASE",
    "TEMPERATURE",
    "baseline_case_rows",
    "build_memory",
    "direct_admission_rows",
    "direct_summary",
    "evaluate_memory_arm",
    "full_path_measurement",
    "gate_specs",
    "load_selected_cases",
    "paired_geometry",
    "prompt_order",
    "rollback_memory",
    "select_cases",
    "strongest_competitor",
]
