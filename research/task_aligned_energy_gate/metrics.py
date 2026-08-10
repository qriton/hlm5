"""Locked estimators for the Banking77 task-aligned energy gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def intent_accuracies(
    predictions: Sequence[int], targets: Sequence[int], intents: Sequence[str]
) -> dict[str, float]:
    if not (len(predictions) == len(targets) == len(intents)):
        raise ValueError("prediction, target, and intent lengths differ")
    grouped: dict[str, list[float]] = {}
    for prediction, target, intent in zip(predictions, targets, intents, strict=True):
        grouped.setdefault(intent, []).append(float(prediction == target))
    if not grouped:
        raise ValueError("at least one row is required")
    return {
        intent: float(np.mean(values, dtype=np.float64))
        for intent, values in sorted(grouped.items())
    }


def macro_accuracy(per_intent: dict[str, float]) -> float:
    if not per_intent:
        raise ValueError("at least one intent is required")
    return float(np.mean(list(per_intent.values()), dtype=np.float64))


def row_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("equal non-empty prediction and target populations required")
    return float(
        np.mean(
            np.asarray(predictions, dtype=np.int64)
            == np.asarray(targets, dtype=np.int64),
            dtype=np.float64,
        )
    )


def paired_intent_bootstrap(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    intents = sorted(candidate)
    if intents != sorted(baseline) or not intents or resamples <= 0:
        raise ValueError("paired non-empty intent sets and positive resamples required")
    differences = np.asarray(
        [candidate[intent] - baseline[intent] for intent in intents],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(intents), size=(resamples, len(intents)))
    sampled = np.mean(differences[indices], axis=1, dtype=np.float64)
    low, high = np.quantile(sampled, [0.025, 0.975], method="linear")
    return {
        "point": float(np.mean(differences, dtype=np.float64)),
        "low": float(low),
        "high": float(high),
    }


def maximum_intent_prediction_share(predictions: Sequence[int]) -> float:
    if not predictions:
        raise ValueError("at least one prediction is required")
    return float(max(Counter(predictions).values()) / len(predictions))


def select_highest_macro(values: dict[str, float]) -> str:
    if not values:
        raise ValueError("at least one candidate is required")
    return min(values, key=lambda name: (-values[name], name))


def select_shrinkage(values: dict[float, float]) -> float:
    if not values:
        raise ValueError("at least one shrinkage is required")
    return min(values, key=lambda alpha: (-values[alpha], alpha))


def transition_counts(
    candidate: Sequence[int], baseline: Sequence[int], targets: Sequence[int]
) -> dict[str, int]:
    if not (len(candidate) == len(baseline) == len(targets)):
        raise ValueError("transition populations differ")
    counts = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
        "changed_predictions": 0,
    }
    for new, old, target in zip(candidate, baseline, targets, strict=True):
        old_correct = old == target
        new_correct = new == target
        if old_correct and new_correct:
            counts["correct_to_correct"] += 1
        elif old_correct:
            counts["correct_to_wrong"] += 1
        elif new_correct:
            counts["wrong_to_correct"] += 1
        else:
            counts["wrong_to_wrong"] += 1
        counts["changed_predictions"] += int(new != old)
    return counts


def classify_development_verdict(
    validity: dict[str, bool],
    alignment: dict[str, bool],
    dynamics: dict[str, bool],
    *,
    interrupted: bool = False,
) -> str:
    if not all(validity.values()):
        return "HARNESS_INVALID"
    if interrupted:
        return "INCONCLUSIVE"
    alignment_pass = all(alignment.values())
    dynamics_pass = alignment_pass and all(dynamics.values())
    if dynamics_pass:
        return "PASS_BOTH_TO_TEST"
    if alignment_pass:
        return "PASS_ALIGNMENT_TO_TEST"
    return "STOP_BEFORE_TEST"
