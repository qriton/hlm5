"""Geometry and metrics for the CA-1 canonical-address source screen."""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import torch

from research.natural_key_transfer_gate.geometry import tensor_sha256, unit_rows
from research.two_axis_admission_gate.geometry import all_finite, higher_quantile

from .contract import (
    CANONICAL_PREFIX,
    EXPECTED_PERMUTATION,
    EXPECTED_PERMUTATION_SHA256,
    MINILM_ARM,
    NEGATIVE_QUANTILE,
    PERMUTATION_SEED,
    PRIMARY_ARM,
    QUANTILE_METHOD,
    SHUFFLED_ARM,
    SOURCE_BARS,
)


def canonical_text(intent: str) -> str:
    if not intent or intent != intent.strip() or intent.lower() != intent:
        raise ValueError("intent must be a non-empty stripped lowercase label")
    if any(character.isspace() for character in intent):
        raise ValueError("intent labels must use underscores, not whitespace")
    if any(not part for part in intent.split("_")):
        raise ValueError("intent labels cannot contain empty underscore fields")
    words = intent.replace("_", " ")
    if not words.strip():
        raise ValueError("intent produced an empty canonical address")
    return CANONICAL_PREFIX + words


def registered_permutation() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PERMUTATION_SEED)
    permutation = torch.randperm(12, generator=generator, dtype=torch.int64)
    if tuple(permutation.tolist()) != EXPECTED_PERMUTATION:
        raise AssertionError("registered label permutation changed")
    if tensor_sha256(permutation) != EXPECTED_PERMUTATION_SHA256:
        raise AssertionError("registered label permutation hash changed")
    return permutation


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def route_from_scores(
    scores: torch.Tensor,
    key_labels: list[str],
    independent_scores: np.ndarray,
) -> dict[str, Any]:
    scores = _matrix("scores", scores)
    if scores.shape[1] != len(key_labels):
        raise ValueError("key labels do not match score columns")
    if len(set(key_labels)) != len(key_labels) or len(key_labels) < 2:
        raise ValueError("canonical routing requires unique labels for >=2 keys")
    maximum_scores, slots = torch.max(scores, dim=1)
    predicted = [key_labels[index] for index in slots.tolist()]
    sorted_scores = torch.sort(scores, dim=1, descending=True).values
    margins = sorted_scores[:, 0] - sorted_scores[:, 1]

    numpy_scores = np.asarray(independent_scores, dtype=np.float64)
    if numpy_scores.shape != tuple(scores.shape):
        raise ValueError("independent score matrix shape changed")
    if not np.isfinite(numpy_scores).all():
        raise ValueError("independent score matrix contains non-finite values")
    numpy_slots = np.argmax(numpy_scores, axis=1)
    independent_error = float(
        np.max(np.abs(numpy_scores - scores.numpy()), initial=0.0)
    )
    if independent_error > 1e-10:
        raise AssertionError("independent NumPy score matrix changed")
    if not np.array_equal(numpy_slots, slots.numpy()):
        raise AssertionError("independent NumPy scorer changed a prediction")
    return {
        "scores": scores,
        "scores_sha256": tensor_sha256(scores),
        "slots": slots,
        "maximum_scores": maximum_scores,
        "margins": margins,
        "predicted": predicted,
        "independent_max_abs_error": independent_error,
    }


def hlm_route(
    query_values: torch.Tensor,
    key_values: torch.Tensor,
    key_labels: list[str],
) -> dict[str, Any]:
    query_input = _matrix("HLM query values", query_values)
    key_input = _matrix("HLM key values", key_values)
    query = unit_rows(query_input)
    keys = unit_rows(key_input)
    if query.shape[1] != keys.shape[1]:
        raise ValueError("HLM query and key dimensions differ")
    scores = torch.relu(query @ keys.T).pow(5)
    query_numpy = query_input.numpy()
    key_numpy = key_input.numpy()
    query_numpy = query_numpy / np.linalg.norm(query_numpy, axis=1, keepdims=True)
    key_numpy = key_numpy / np.linalg.norm(key_numpy, axis=1, keepdims=True)
    independent_scores = np.maximum(query_numpy @ key_numpy.T, 0.0) ** 5
    return route_from_scores(scores, key_labels, independent_scores)


def minilm_route(
    query_values: torch.Tensor,
    key_values: torch.Tensor,
    key_labels: list[str],
) -> dict[str, Any]:
    query_input = _matrix("MiniLM query values", query_values)
    key_input = _matrix("MiniLM key values", key_values)
    query = unit_rows(query_input)
    keys = unit_rows(key_input)
    if query.shape[1] != keys.shape[1]:
        raise ValueError("MiniLM query and key dimensions differ")
    query_numpy = query_input.numpy()
    key_numpy = key_input.numpy()
    query_numpy = query_numpy / np.linalg.norm(query_numpy, axis=1, keepdims=True)
    key_numpy = key_numpy / np.linalg.norm(key_numpy, axis=1, keepdims=True)
    independent_scores = query_numpy @ key_numpy.T
    return route_from_scores(query @ keys.T, key_labels, independent_scores)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.to(device="cpu", dtype=torch.float64)
    return {
        "0.05": float(torch.quantile(values, 0.05).item()),
        "0.5": float(torch.quantile(values, 0.5).item()),
        "0.95": float(torch.quantile(values, 0.95).item()),
    }


def threshold_from_negatives(route: dict[str, Any]) -> dict[str, Any]:
    threshold = higher_quantile(route["maximum_scores"], NEGATIVE_QUANTILE)
    admitted = route["maximum_scores"].to(dtype=torch.float64) > threshold
    return {
        "tau_absolute": threshold,
        "negative_quantile": NEGATIVE_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "comparison": "strict_greater_than",
        "negative_count": int(len(admitted)),
        "negative_admitted_count": int(admitted.sum().item()),
        "negative_admission_rate": float(admitted.double().mean().item()),
    }


def _supported_summary(
    route: dict[str, Any],
    truth: list[str],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(truth) != len(route["predicted"]):
        raise ValueError("truth labels do not match predictions")
    correct = torch.tensor(
        [prediction == label for prediction, label in zip(route["predicted"], truth, strict=True)],
        dtype=torch.bool,
    )
    admitted = route["maximum_scores"].to(dtype=torch.float64) > threshold
    correct_admitted = correct & admitted
    per_intent: dict[str, Any] = {}
    for intent in sorted(set(truth)):
        mask = torch.tensor([label == intent for label in truth], dtype=torch.bool)
        count = int(mask.sum().item())
        intent_admitted = int(admitted[mask].sum().item())
        intent_correct_admitted = int(correct_admitted[mask].sum().item())
        per_intent[intent] = {
            "count": count,
            "accuracy": float(correct[mask].double().mean().item()),
            "coverage": intent_admitted / float(count),
            "correct_admission_rate": intent_correct_admitted / float(count),
            "admitted_count": intent_admitted,
            "correct_admitted_count": intent_correct_admitted,
        }
    admitted_count = int(admitted.sum().item())
    correct_admitted_count = int(correct_admitted.sum().item())
    predicted_counts = collections.Counter(route["predicted"])
    summary = {
        "row_count": len(truth),
        "top1_accuracy": float(correct.double().mean().item()),
        "macro_top1_accuracy": float(
            np.mean([item["accuracy"] for item in per_intent.values()], dtype=np.float64)
        ),
        "coverage": admitted_count / float(len(truth)),
        "macro_coverage": float(
            np.mean([item["coverage"] for item in per_intent.values()], dtype=np.float64)
        ),
        "correct_admission_rate": correct_admitted_count / float(len(truth)),
        "macro_correct_admission_rate": float(
            np.mean(
                [item["correct_admission_rate"] for item in per_intent.values()],
                dtype=np.float64,
            )
        ),
        "selective_accuracy": (
            correct_admitted_count / float(admitted_count) if admitted_count else 0.0
        ),
        "admitted_count": admitted_count,
        "correct_admitted_count": correct_admitted_count,
        "maximum_predicted_intent_share": max(predicted_counts.values())
        / float(len(truth)),
        "maximum_predicted_intent": sorted(
            predicted_counts,
            key=lambda label: (-predicted_counts[label], label),
        )[0],
        "score_quantiles": _quantiles(route["maximum_scores"]),
        "margin_quantiles": _quantiles(route["margins"]),
        "scores_sha256": route["scores_sha256"],
        "independent_max_abs_error": route["independent_max_abs_error"],
        "per_intent": per_intent,
    }
    arrays = {
        "predicted": route["predicted"],
        "maximum_scores": route["maximum_scores"].to(dtype=torch.float64),
        "margins": route["margins"].to(dtype=torch.float64),
        "correct": correct,
        "admitted": admitted,
    }
    return summary, arrays


def _negative_summary(
    route: dict[str, Any], threshold: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    admitted = route["maximum_scores"].to(dtype=torch.float64) > threshold
    summary = {
        "row_count": int(len(admitted)),
        "admitted_count": int(admitted.sum().item()),
        "false_admission_rate": float(admitted.double().mean().item()),
        "score_quantiles": _quantiles(route["maximum_scores"]),
        "margin_quantiles": _quantiles(route["margins"]),
        "scores_sha256": route["scores_sha256"],
        "independent_max_abs_error": route["independent_max_abs_error"],
    }
    arrays = {
        "predicted": route["predicted"],
        "maximum_scores": route["maximum_scores"].to(dtype=torch.float64),
        "margins": route["margins"].to(dtype=torch.float64),
        "admitted": admitted,
    }
    return summary, arrays


def evaluate_arm(
    calibration_positive: dict[str, Any],
    calibration_truth: list[str],
    calibration_negative: dict[str, Any],
    audit_positive: dict[str, Any],
    audit_truth: list[str],
    audit_negative: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold = threshold_from_negatives(calibration_negative)
    tau = threshold["tau_absolute"]
    calibration_supported, calibration_supported_arrays = _supported_summary(
        calibration_positive, calibration_truth, tau
    )
    calibration_off_support, calibration_negative_arrays = _negative_summary(
        calibration_negative, tau
    )
    audit_supported, audit_supported_arrays = _supported_summary(
        audit_positive, audit_truth, tau
    )
    audit_off_support, audit_negative_arrays = _negative_summary(audit_negative, tau)
    return (
        {
            "threshold": threshold,
            "calibration_supported": calibration_supported,
            "calibration_off_support": calibration_off_support,
            "audit_supported": audit_supported,
            "audit_off_support": audit_off_support,
        },
        {
            "calibration_supported": calibration_supported_arrays,
            "calibration_off_support": calibration_negative_arrays,
            "audit_supported": audit_supported_arrays,
            "audit_off_support": audit_negative_arrays,
        },
    )


def source_verdict(arms: dict[str, dict[str, Any]]) -> tuple[str, dict[str, bool], dict[str, float]]:
    candidate = arms[PRIMARY_ARM]
    supported = candidate["audit_supported"]
    negative = candidate["audit_off_support"]
    shuffled = arms[SHUFFLED_ARM]["audit_supported"]
    minilm = arms[MINILM_ARM]["audit_supported"]
    comparisons = {
        "top1_gain_over_shuffled": supported["macro_top1_accuracy"]
        - shuffled["macro_top1_accuracy"],
        "correct_admission_gain_over_shuffled": supported[
            "macro_correct_admission_rate"
        ]
        - shuffled["macro_correct_admission_rate"],
        "top1_drop_vs_minilm": minilm["macro_top1_accuracy"]
        - supported["macro_top1_accuracy"],
    }
    bars = {
        "macro_top1_accuracy": supported["macro_top1_accuracy"]
        >= SOURCE_BARS["macro_top1_accuracy"],
        "macro_coverage": supported["macro_coverage"]
        >= SOURCE_BARS["macro_coverage"],
        "macro_correct_admission_rate": supported["macro_correct_admission_rate"]
        >= SOURCE_BARS["macro_correct_admission_rate"],
        "selective_accuracy": supported["selective_accuracy"]
        >= SOURCE_BARS["selective_accuracy"],
        "off_support_false_admission_rate": negative["false_admission_rate"]
        <= SOURCE_BARS["off_support_false_admission_rate"],
        "maximum_predicted_intent_share": supported[
            "maximum_predicted_intent_share"
        ]
        <= SOURCE_BARS["maximum_predicted_intent_share"],
        "top1_gain_over_shuffled": comparisons["top1_gain_over_shuffled"]
        >= SOURCE_BARS["top1_gain_over_shuffled"],
        "correct_admission_gain_over_shuffled": comparisons[
            "correct_admission_gain_over_shuffled"
        ]
        >= SOURCE_BARS["correct_admission_gain_over_shuffled"],
        "top1_noninferiority_to_minilm": comparisons["top1_drop_vs_minilm"]
        <= SOURCE_BARS["top1_drop_vs_minilm"],
    }
    return (
        "SOURCE_SCREEN_LIVE" if all(bars.values()) else "CLOSED_SOURCE_ONLY",
        bars,
        comparisons,
    )


__all__ = [
    "all_finite",
    "canonical_text",
    "evaluate_arm",
    "hlm_route",
    "minilm_route",
    "registered_permutation",
    "route_from_scores",
    "source_verdict",
    "threshold_from_negatives",
]
