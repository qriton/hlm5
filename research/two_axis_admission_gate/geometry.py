"""Float64 class-margin calibration and metrics for the registered SCA-2 assay."""

from __future__ import annotations

import collections
from typing import Any, Iterable

import numpy as np
import torch

from research.natural_key_transfer_gate.geometry import (
    route_scores,
    support_geometry,
    tensor_sha256,
)
from research.source_calibrated_admission_gate.geometry import (
    KeyOperators,
    apply_arm,
    fit_key_operators,
    registered_random_basis,
)

from .contract import (
    ABSOLUTE_NEGATIVE_QUANTILE,
    ARM_NAMES,
    AUDIT_BARS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATE_ARM,
    MARGIN_ERROR_QUANTILE,
    MINIMUM_MARGIN_CALIBRATION_ERRORS,
    PASS_BARS,
    QUANTILE_METHOD,
)


def _vector(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def higher_quantile(values: torch.Tensor, quantile: float) -> float:
    values = _vector("quantile values", values).numpy()
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    return float(np.quantile(values, quantile, method=QUANTILE_METHOD))


def class_route_scores(
    query_keys: torch.Tensor,
    support_keys: torch.Tensor,
    support_labels: list[str],
) -> dict[str, Any]:
    """Route slots, then expose max-per-class and top-two class scores."""

    if len(support_labels) != support_keys.shape[0]:
        raise ValueError("support labels do not match support keys")
    class_labels = list(dict.fromkeys(support_labels))
    if len(class_labels) < 2:
        raise ValueError("class-margin routing needs at least two support classes")
    routed = route_scores(query_keys, support_keys)
    scores = routed["scores"].to(device="cpu", dtype=torch.float64)
    class_scores = torch.empty(
        (scores.shape[0], len(class_labels)), dtype=torch.float64
    )
    for class_index, label in enumerate(class_labels):
        mask = torch.tensor(
            [item == label for item in support_labels], dtype=torch.bool
        )
        class_scores[:, class_index] = torch.max(scores[:, mask], dim=1).values

    predicted = [support_labels[int(slot)] for slot in routed["slots"].tolist()]
    class_index = {label: index for index, label in enumerate(class_labels)}
    predicted_indices = torch.tensor(
        [class_index[label] for label in predicted], dtype=torch.long
    )
    row_indices = torch.arange(scores.shape[0], dtype=torch.long)
    winning = class_scores[row_indices, predicted_indices]
    masked = class_scores.clone()
    masked[row_indices, predicted_indices] = -torch.inf
    runner_up = torch.max(masked, dim=1).values
    margins = winning - runner_up
    if not torch.equal(winning, routed["maximum_scores"].to(dtype=torch.float64)):
        raise AssertionError("slot winner and class winner disagree")
    if not bool((margins >= 0.0).all()):
        raise AssertionError("class margin became negative")

    numpy_scores = scores.numpy()
    numpy_class_scores = np.empty(class_scores.shape, dtype=np.float64)
    for class_index_value, label in enumerate(class_labels):
        indices = [
            index for index, item in enumerate(support_labels) if item == label
        ]
        numpy_class_scores[:, class_index_value] = np.max(
            numpy_scores[:, indices], axis=1
        )
    numpy_margin = np.empty(len(predicted), dtype=np.float64)
    for row, predicted_index in enumerate(predicted_indices.tolist()):
        alternatives = np.delete(numpy_class_scores[row], predicted_index)
        numpy_margin[row] = (
            numpy_class_scores[row, predicted_index] - np.max(alternatives)
        )
    independent_error = float(
        np.max(np.abs(numpy_class_scores - class_scores.numpy()), initial=0.0)
    )
    margin_error = float(
        np.max(np.abs(numpy_margin - margins.numpy()), initial=0.0)
    )
    if independent_error > 1e-10 or margin_error > 1e-10:
        raise AssertionError("independent class scorer exceeds 1e-10")

    return {
        **routed,
        "class_labels": class_labels,
        "class_scores": class_scores,
        "class_scores_sha256": tensor_sha256(class_scores),
        "predicted": predicted,
        "predicted_class_indices": predicted_indices,
        "runner_up_scores": runner_up,
        "class_margins": margins,
        "class_margins_sha256": tensor_sha256(margins),
        "class_independent_max_abs_error": max(independent_error, margin_error),
    }


def calibrate_thresholds(
    negative_route: dict[str, Any],
    positive_route: dict[str, Any],
    positive_labels: list[str],
) -> dict[str, Any]:
    if len(positive_labels) != len(positive_route["predicted"]):
        raise ValueError("calibration-positive labels do not match routes")
    tau_absolute = higher_quantile(
        negative_route["maximum_scores"], ABSOLUTE_NEGATIVE_QUANTILE
    )
    correct = torch.tensor(
        [
            predicted == truth
            for predicted, truth in zip(
                positive_route["predicted"], positive_labels, strict=True
            )
        ],
        dtype=torch.bool,
    )
    wrong = ~correct
    wrong_count = int(wrong.sum().item())
    negative_absolute = (
        negative_route["maximum_scores"].to(dtype=torch.float64) > tau_absolute
    )
    record: dict[str, Any] = {
        "tau_absolute": tau_absolute,
        "tau_margin": None,
        "absolute_negative_quantile": ABSOLUTE_NEGATIVE_QUANTILE,
        "margin_error_quantile": MARGIN_ERROR_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "comparison": "strict_greater_than",
        "calibration_negative_count": int(len(negative_absolute)),
        "calibration_negative_absolute_admitted_count": int(
            negative_absolute.sum().item()
        ),
        "calibration_negative_absolute_rate": float(
            negative_absolute.to(dtype=torch.float64).mean().item()
        ),
        "calibration_positive_count": len(positive_labels),
        "calibration_error_count": wrong_count,
        "minimum_calibration_errors": MINIMUM_MARGIN_CALIBRATION_ERRORS,
        "sufficient_margin_errors": wrong_count
        >= MINIMUM_MARGIN_CALIBRATION_ERRORS,
        "calibration_error_margin_survivor_count": None,
        "calibration_error_margin_survival_rate": None,
    }
    if wrong_count < MINIMUM_MARGIN_CALIBRATION_ERRORS:
        return record
    tau_margin = higher_quantile(
        positive_route["class_margins"][wrong], MARGIN_ERROR_QUANTILE
    )
    wrong_survivors = positive_route["class_margins"][wrong] > tau_margin
    record.update(
        {
            "tau_margin": tau_margin,
            "calibration_error_margin_survivor_count": int(
                wrong_survivors.sum().item()
            ),
            "calibration_error_margin_survival_rate": float(
                wrong_survivors.to(dtype=torch.float64).mean().item()
            ),
        }
    )
    return record


def gate_masks(route: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, torch.Tensor]:
    tau_margin = thresholds.get("tau_margin")
    if tau_margin is None:
        raise ValueError("margin threshold is unavailable")
    absolute = route["maximum_scores"].to(dtype=torch.float64) > float(
        thresholds["tau_absolute"]
    )
    margin = route["class_margins"] > float(tau_margin)
    return {"absolute_only": absolute, "dual": absolute & margin}


def matched_proximity_mask(
    route: dict[str, Any],
    absolute_mask: torch.Tensor,
    admitted_count: int,
    row_ids: list[str],
) -> torch.Tensor:
    if len(row_ids) != len(absolute_mask):
        raise ValueError("row IDs do not match route rows")
    eligible = [index for index, value in enumerate(absolute_mask.tolist()) if value]
    if admitted_count < 0 or admitted_count > len(eligible):
        raise ValueError("matched admission count exceeds absolute pool")
    scores = route["maximum_scores"].to(dtype=torch.float64).tolist()
    ordered = sorted(eligible, key=lambda index: (-scores[index], row_ids[index]))
    selected = torch.zeros(len(row_ids), dtype=torch.bool)
    if admitted_count:
        selected[torch.tensor(ordered[:admitted_count], dtype=torch.long)] = True
    return selected


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = _vector("summary values", values)
    return {
        "0.05": float(torch.quantile(values, 0.05, interpolation="linear").item()),
        "0.5": float(torch.quantile(values, 0.5, interpolation="linear").item()),
        "0.95": float(torch.quantile(values, 0.95, interpolation="linear").item()),
    }


def _gate_summary(
    admitted: torch.Tensor,
    correct: torch.Tensor,
    query_labels: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    admitted = admitted.to(dtype=torch.bool, device="cpu")
    correct = correct.to(dtype=torch.bool, device="cpu")
    correct_admitted = correct & admitted
    per_intent: dict[str, dict[str, Any]] = {}
    for intent in sorted(set(query_labels)):
        mask = torch.tensor([label == intent for label in query_labels], dtype=torch.bool)
        count = int(mask.sum().item())
        admitted_count = int(admitted[mask].sum().item())
        correct_admitted_count = int(correct_admitted[mask].sum().item())
        per_intent[intent] = {
            "count": count,
            "admitted_count": admitted_count,
            "correct_admitted_count": correct_admitted_count,
            "coverage": admitted_count / float(count),
            "correct_admission_rate": correct_admitted_count / float(count),
            "selective_accuracy": (
                correct_admitted_count / float(admitted_count)
                if admitted_count
                else 0.0
            ),
            "selective_accuracy_defined": admitted_count > 0,
        }
    admitted_count = int(admitted.sum().item())
    correct_admitted_count = int(correct_admitted.sum().item())
    metrics = {
        "admitted_count": admitted_count,
        "correct_admitted_count": correct_admitted_count,
        "coverage": admitted_count / float(len(admitted)),
        "correct_admission_rate": correct_admitted_count / float(len(admitted)),
        "macro_coverage": float(
            np.mean([item["coverage"] for item in per_intent.values()], dtype=np.float64)
        ),
        "macro_correct_admission_rate": float(
            np.mean(
                [item["correct_admission_rate"] for item in per_intent.values()],
                dtype=np.float64,
            )
        ),
        "selective_accuracy": (
            correct_admitted_count / float(admitted_count) if admitted_count else 0.0
        ),
        "selective_accuracy_defined": admitted_count > 0,
    }
    return metrics, per_intent


def summarize_supported(
    route: dict[str, Any],
    query_labels: list[str],
    masks: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(query_labels) != len(route["predicted"]):
        raise ValueError("query labels do not match route rows")
    correct = torch.tensor(
        [
            predicted == truth
            for predicted, truth in zip(route["predicted"], query_labels, strict=True)
        ],
        dtype=torch.bool,
    )
    per_intent_accuracy: dict[str, float] = {}
    for intent in sorted(set(query_labels)):
        mask = torch.tensor([label == intent for label in query_labels], dtype=torch.bool)
        per_intent_accuracy[intent] = float(
            correct[mask].to(dtype=torch.float64).mean().item()
        )

    gate_metrics: dict[str, Any] = {}
    per_intent: dict[str, dict[str, Any]] = {
        intent: {"accuracy": accuracy}
        for intent, accuracy in per_intent_accuracy.items()
    }
    for name, admitted in masks.items():
        summary, gate_per_intent = _gate_summary(admitted, correct, query_labels)
        gate_metrics[name] = summary
        for intent, values in gate_per_intent.items():
            per_intent[intent][name] = values

    class_index = {label: index for index, label in enumerate(route["class_labels"])}
    true_scores = torch.empty(len(query_labels), dtype=torch.float64)
    false_scores = torch.empty(len(query_labels), dtype=torch.float64)
    for row, truth in enumerate(query_labels):
        if truth not in class_index:
            raise ValueError(f"truth {truth!r} is absent from support bank")
        truth_index = class_index[truth]
        true_scores[row] = route["class_scores"][row, truth_index]
        alternatives = torch.cat(
            (
                route["class_scores"][row, :truth_index],
                route["class_scores"][row, truth_index + 1 :],
            )
        )
        false_scores[row] = torch.max(alternatives)
    predicted_counts = collections.Counter(route["predicted"])
    metrics = {
        "row_count": len(query_labels),
        "top1_accuracy": float(correct.to(dtype=torch.float64).mean().item()),
        "macro_top1_accuracy": float(
            np.mean(list(per_intent_accuracy.values()), dtype=np.float64)
        ),
        "maximum_predicted_intent_share": max(predicted_counts.values())
        / float(len(query_labels)),
        "maximum_predicted_intent": sorted(
            predicted_counts,
            key=lambda label: (-predicted_counts[label], label),
        )[0],
        "maximum_score_quantiles": _quantiles(route["maximum_scores"]),
        "class_margin_quantiles": _quantiles(route["class_margins"]),
        "true_minus_best_false_score_quantiles": _quantiles(
            true_scores - false_scores
        ),
        "gates": gate_metrics,
        "per_intent": per_intent,
        "slot_scores_sha256": route["scores_sha256"],
        "class_scores_sha256": route["class_scores_sha256"],
        "class_margins_sha256": route["class_margins_sha256"],
        "independent_max_abs_error": max(
            route["independent_max_abs_error"],
            route["class_independent_max_abs_error"],
        ),
    }
    arrays = {
        "predicted": route["predicted"],
        "correct": correct,
        "maximum_scores": route["maximum_scores"].to(dtype=torch.float64),
        "class_margins": route["class_margins"].to(dtype=torch.float64),
        "masks": {name: mask.to(dtype=torch.bool) for name, mask in masks.items()},
    }
    return metrics, arrays


def summarize_negative(
    route: dict[str, Any], masks: dict[str, torch.Tensor]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gates: dict[str, Any] = {}
    for name, admitted in masks.items():
        admitted = admitted.to(dtype=torch.bool, device="cpu")
        gates[name] = {
            "admitted_count": int(admitted.sum().item()),
            "false_admission_rate": float(
                admitted.to(dtype=torch.float64).mean().item()
            ),
        }
    metrics = {
        "row_count": int(len(route["maximum_scores"])),
        "maximum_score_quantiles": _quantiles(route["maximum_scores"]),
        "class_margin_quantiles": _quantiles(route["class_margins"]),
        "gates": gates,
        "slot_scores_sha256": route["scores_sha256"],
        "class_scores_sha256": route["class_scores_sha256"],
        "class_margins_sha256": route["class_margins_sha256"],
        "independent_max_abs_error": max(
            route["independent_max_abs_error"],
            route["class_independent_max_abs_error"],
        ),
    }
    arrays = {
        "maximum_scores": route["maximum_scores"].to(dtype=torch.float64),
        "class_margins": route["class_margins"].to(dtype=torch.float64),
        "masks": {name: mask.to(dtype=torch.bool) for name, mask in masks.items()},
        "predicted": route["predicted"],
    }
    return metrics, arrays


def calibrate_arm(
    operators: KeyOperators,
    arm: str,
    support_values: torch.Tensor,
    support_labels: list[str],
    positive_values: torch.Tensor,
    positive_labels: list[str],
    negative_values: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = apply_arm(support_values, operators, arm)
    positive = apply_arm(positive_values, operators, arm)
    negative = apply_arm(negative_values, operators, arm)
    positive_route = class_route_scores(positive, support, support_labels)
    negative_route = class_route_scores(negative, support, support_labels)
    thresholds = calibrate_thresholds(
        negative_route, positive_route, positive_labels
    )
    record: dict[str, Any] = {
        "arm": arm,
        "thresholds": thresholds,
        "support_geometry": support_geometry(support, support_labels),
        "positive_independent_max_abs_error": max(
            positive_route["independent_max_abs_error"],
            positive_route["class_independent_max_abs_error"],
        ),
        "negative_independent_max_abs_error": max(
            negative_route["independent_max_abs_error"],
            negative_route["class_independent_max_abs_error"],
        ),
    }
    arrays: dict[str, Any] = {
        "support_keys": support,
        "positive_route": positive_route,
        "negative_route": negative_route,
    }
    if thresholds["sufficient_margin_errors"]:
        positive_masks = gate_masks(positive_route, thresholds)
        negative_masks = gate_masks(negative_route, thresholds)
        positive_metrics, positive_arrays = summarize_supported(
            positive_route, positive_labels, positive_masks
        )
        negative_metrics, negative_arrays = summarize_negative(
            negative_route, negative_masks
        )
        record["positive"] = positive_metrics
        record["negative"] = negative_metrics
        arrays["positive"] = positive_arrays
        arrays["negative"] = negative_arrays
    return record, arrays


def evaluate_stage(
    operators: KeyOperators,
    arm: str,
    thresholds: dict[str, Any],
    support_values: torch.Tensor,
    support_labels: list[str],
    positive_values: torch.Tensor,
    positive_labels: list[str],
    positive_row_ids: list[str],
    negative_values: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = apply_arm(support_values, operators, arm)
    positive = apply_arm(positive_values, operators, arm)
    negative = apply_arm(negative_values, operators, arm)
    positive_route = class_route_scores(positive, support, support_labels)
    negative_route = class_route_scores(negative, support, support_labels)
    positive_masks = gate_masks(positive_route, thresholds)
    negative_masks = gate_masks(negative_route, thresholds)
    dual_count = int(positive_masks["dual"].sum().item())
    positive_masks["matched_proximity_top_k"] = matched_proximity_mask(
        positive_route,
        positive_masks["absolute_only"],
        dual_count,
        positive_row_ids,
    )
    positive_metrics, positive_arrays = summarize_supported(
        positive_route, positive_labels, positive_masks
    )
    negative_metrics, negative_arrays = summarize_negative(
        negative_route, negative_masks
    )
    return (
        {
            "arm": arm,
            "thresholds": thresholds,
            "supported": positive_metrics,
            "off_support": negative_metrics,
            "support_geometry": support_geometry(support, support_labels),
        },
        {
            "support_keys": support,
            "supported": positive_arrays,
            "off_support": negative_arrays,
        },
    )


def _retention(candidate: dict[str, Any]) -> float:
    gates = candidate["supported"]["gates"]
    denominator = gates["absolute_only"]["correct_admitted_count"]
    if denominator == 0:
        return 0.0
    return gates["dual"]["correct_admitted_count"] / float(denominator)


def stage_comparisons(candidate: dict[str, Any]) -> dict[str, float]:
    gates = candidate["supported"]["gates"]
    return {
        "selective_gain_vs_absolute_only": gates["dual"]["selective_accuracy"]
        - gates["absolute_only"]["selective_accuracy"],
        "selective_gain_vs_matched_proximity": gates["dual"][
            "selective_accuracy"
        ]
        - gates["matched_proximity_top_k"]["selective_accuracy"],
        "correct_admission_retention_vs_absolute_only": _retention(candidate),
    }


def source_audit_verdict(
    arms: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool], dict[str, float]]:
    candidate = arms[CANDIDATE_ARM]
    gates = candidate["supported"]["gates"]
    comparisons = stage_comparisons(candidate)
    bars = {
        "candidate_macro_coverage": gates["dual"]["macro_coverage"]
        >= AUDIT_BARS["candidate_macro_coverage"],
        "candidate_macro_correct_admission": gates["dual"][
            "macro_correct_admission_rate"
        ]
        >= AUDIT_BARS["candidate_macro_correct_admission"],
        "candidate_selective_accuracy": gates["dual"]["selective_accuracy"]
        >= AUDIT_BARS["candidate_selective_accuracy"],
        "candidate_selective_gain_vs_absolute_only": comparisons[
            "selective_gain_vs_absolute_only"
        ]
        >= AUDIT_BARS["candidate_selective_gain_vs_absolute_only"],
        "candidate_correct_admission_retention_vs_absolute_only": comparisons[
            "correct_admission_retention_vs_absolute_only"
        ]
        >= AUDIT_BARS["candidate_correct_admission_retention_vs_absolute_only"],
        "candidate_off_support_false_admission_rate": candidate["off_support"][
            "gates"
        ]["dual"]["false_admission_rate"]
        <= AUDIT_BARS["candidate_off_support_false_admission_rate"],
    }
    return (
        "PASS_TO_TARGET" if all(bars.values()) else "STOP_BEFORE_TARGET",
        bars,
        comparisons,
    )


def development_verdict(
    arms: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool], dict[str, float]]:
    candidate = arms[CANDIDATE_ARM]
    raw = arms["raw"]
    gates = candidate["supported"]["gates"]
    comparisons = stage_comparisons(candidate)
    bars = {
        "candidate_macro_coverage": gates["dual"]["macro_coverage"]
        >= PASS_BARS["candidate_macro_coverage"],
        "candidate_macro_correct_admission": gates["dual"][
            "macro_correct_admission_rate"
        ]
        >= PASS_BARS["candidate_macro_correct_admission"],
        "candidate_selective_accuracy": gates["dual"]["selective_accuracy"]
        >= PASS_BARS["candidate_selective_accuracy"],
        "candidate_selective_gain_vs_absolute_only": comparisons[
            "selective_gain_vs_absolute_only"
        ]
        >= PASS_BARS["candidate_selective_gain_vs_absolute_only"],
        "candidate_selective_gain_vs_matched_proximity": comparisons[
            "selective_gain_vs_matched_proximity"
        ]
        >= PASS_BARS["candidate_selective_gain_vs_matched_proximity"],
        "candidate_correct_admission_retention_vs_absolute_only": comparisons[
            "correct_admission_retention_vs_absolute_only"
        ]
        >= PASS_BARS["candidate_correct_admission_retention_vs_absolute_only"],
        "candidate_off_support_false_admission_rate": candidate["off_support"][
            "gates"
        ]["dual"]["false_admission_rate"]
        <= PASS_BARS["candidate_off_support_false_admission_rate"],
        "candidate_max_predicted_intent_share": candidate["supported"][
            "maximum_predicted_intent_share"
        ]
        <= PASS_BARS["candidate_max_predicted_intent_share"],
        "candidate_top1_noninferiority": candidate["supported"][
            "macro_top1_accuracy"
        ]
        >= raw["supported"]["macro_top1_accuracy"]
        - PASS_BARS["candidate_top1_accuracy_drop_vs_raw"],
    }
    return (
        "PASS_TO_TEST" if all(bars.values()) else "STOP_BEFORE_TEST",
        bars,
        comparisons,
    )


def paired_intent_selective_bootstrap(
    stage: dict[str, Any],
    first_gate: str,
    second_gate: str,
) -> dict[str, Any]:
    per_intent = stage["supported"]["per_intent"]
    intents = sorted(per_intent)

    def statistic(indices: Iterable[int]) -> float:
        first_correct = 0
        first_admitted = 0
        second_correct = 0
        second_admitted = 0
        for index in indices:
            values = per_intent[intents[index]]
            first_correct += values[first_gate]["correct_admitted_count"]
            first_admitted += values[first_gate]["admitted_count"]
            second_correct += values[second_gate]["correct_admitted_count"]
            second_admitted += values[second_gate]["admitted_count"]
        first = first_correct / float(first_admitted) if first_admitted else 0.0
        second = second_correct / float(second_admitted) if second_admitted else 0.0
        return first - second

    point = statistic(range(len(intents)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(intents),
        size=(BOOTSTRAP_RESAMPLES, len(intents)),
        endpoint=False,
    )
    samples = np.asarray([statistic(sample) for sample in indices], dtype=np.float64)
    interval = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "metric": "aggregate_selective_accuracy_difference",
        "first_gate": first_gate,
        "second_gate": second_gate,
        "intents": intents,
        "point": point,
        "interval": [float(interval[0]), float(interval[1])],
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "interpretation": "descriptive_fixed_intent_stability_not_independent_coverage",
    }


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return True


__all__ = [
    "KeyOperators",
    "all_finite",
    "apply_arm",
    "calibrate_arm",
    "calibrate_thresholds",
    "class_route_scores",
    "development_verdict",
    "evaluate_stage",
    "fit_key_operators",
    "gate_masks",
    "higher_quantile",
    "matched_proximity_mask",
    "paired_intent_selective_bootstrap",
    "registered_random_basis",
    "source_audit_verdict",
    "stage_comparisons",
    "tensor_sha256",
]
