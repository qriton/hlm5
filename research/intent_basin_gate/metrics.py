"""Locked estimators for the HLM5 I1 intent-basin gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def intent_accuracies(
    predictions: Sequence[int],
    targets: Sequence[int],
    intent_names: Sequence[str],
) -> dict[str, float]:
    """Return accuracy for every represented intent."""

    if not (len(predictions) == len(targets) == len(intent_names)):
        raise ValueError("predictions, targets, and intent names must have equal length")
    grouped: dict[str, list[float]] = {}
    for prediction, target, intent in zip(
        predictions, targets, intent_names, strict=True
    ):
        grouped.setdefault(intent, []).append(float(prediction == target))
    if not grouped:
        raise ValueError("at least one row is required")
    return {
        intent: float(np.mean(values, dtype=np.float64))
        for intent, values in sorted(grouped.items())
    }


def macro_accuracy(per_intent: dict[str, float]) -> float:
    """Arithmetic mean over equally weighted intent accuracies."""

    if not per_intent:
        raise ValueError("at least one intent is required")
    return float(np.mean(list(per_intent.values()), dtype=np.float64))


def paired_intent_bootstrap(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_810,
) -> dict[str, float]:
    """Cluster bootstrap over intent-level paired differences."""

    intents = sorted(candidate)
    if intents != sorted(baseline):
        raise ValueError("candidate and baseline intent sets must match")
    if not intents or resamples <= 0:
        raise ValueError("non-empty intents and positive resamples are required")
    differences = np.asarray(
        [candidate[intent] - baseline[intent] for intent in intents],
        dtype=np.float64,
    )
    point = float(np.mean(differences, dtype=np.float64))
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(intents),
        size=(resamples, len(intents)),
        endpoint=False,
    )
    sampled = np.mean(differences[indices], axis=1, dtype=np.float64)
    low, high = np.quantile(sampled, [0.025, 0.975], method="linear")
    return {"point": point, "low": float(low), "high": float(high)}


def average_rank_auc(
    supported_scores: Sequence[float],
    oos_scores: Sequence[float],
) -> float:
    """Tie-aware probability that a supported score exceeds an OOS score."""

    positive = np.asarray(supported_scores, dtype=np.float64)
    negative = np.asarray(oos_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both score populations must be non-empty")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    rank_sum = float(ranks[: positive.size].sum(dtype=np.float64))
    auc = (
        rank_sum - positive.size * (positive.size + 1) / 2.0
    ) / (positive.size * negative.size)
    return float(auc)


def maximum_intent_prediction_share(predictions: Sequence[int]) -> float:
    """Largest fraction assigned to one predicted intent index."""

    if not predictions:
        raise ValueError("at least one prediction is required")
    values, counts = np.unique(np.asarray(predictions, dtype=np.int64), return_counts=True)
    if values.size == 0:
        raise ValueError("at least one prediction is required")
    return float(counts.max() / len(predictions))


def select_static_control(control_macros: dict[str, float]) -> str:
    """Select highest macro accuracy, breaking ties by control name."""

    if not control_macros:
        raise ValueError("at least one static control is required")
    return min(control_macros, key=lambda name: (-control_macros[name], name))


def classify_development_verdict(
    validity: dict[str, bool],
    scientific: dict[str, bool],
    *,
    interrupted: bool = False,
) -> str:
    """Apply validity precedence and the registered development vocabulary."""

    if not all(validity.values()):
        return "HARNESS_INVALID"
    if interrupted:
        return "INCONCLUSIVE"
    return "PASS_TO_TEST" if all(scientific.values()) else "STOP_BEFORE_TEST"


def classify_test_verdict(
    validity: dict[str, bool],
    scientific: dict[str, bool],
    *,
    interrupted: bool = False,
) -> str:
    """Apply validity precedence and the registered test vocabulary."""

    if not all(validity.values()):
        return "HARNESS_INVALID"
    if interrupted:
        return "INCONCLUSIVE"
    return "PASS" if all(scientific.values()) else "FAIL"
