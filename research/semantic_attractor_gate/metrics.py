"""Locked estimators for the HLM5 S1 semantic attractor gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def relation_accuracies(
    predictions: Sequence[int],
    targets: Sequence[int],
    relations: Sequence[str],
) -> dict[str, float]:
    if not (len(predictions) == len(targets) == len(relations)):
        raise ValueError("predictions, targets, and relations must have equal length")
    grouped: dict[str, list[float]] = {}
    for prediction, target, relation in zip(predictions, targets, relations, strict=True):
        grouped.setdefault(relation, []).append(float(prediction == target))
    if not grouped:
        raise ValueError("at least one row is required")
    return {
        relation: float(np.mean(values, dtype=np.float64))
        for relation, values in sorted(grouped.items())
    }


def macro_accuracy(per_relation: dict[str, float]) -> float:
    if not per_relation:
        raise ValueError("at least one relation is required")
    return float(np.mean(list(per_relation.values()), dtype=np.float64))


def paired_relation_bootstrap(
    candidate: dict[str, float],
    baseline: dict[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 20_260_810,
) -> dict[str, float]:
    relations = sorted(candidate)
    if relations != sorted(baseline):
        raise ValueError("candidate and baseline relation sets must match")
    if not relations or resamples <= 0:
        raise ValueError("non-empty relations and positive resamples are required")
    differences = np.asarray(
        [candidate[r] - baseline[r] for r in relations],
        dtype=np.float64,
    )
    point = float(np.mean(differences, dtype=np.float64))
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(relations),
        size=(resamples, len(relations)),
        endpoint=False,
    )
    sampled = np.mean(differences[indices], axis=1, dtype=np.float64)
    low, high = np.quantile(sampled, [0.025, 0.975], method="linear")
    return {"point": point, "low": float(low), "high": float(high)}


def average_rank_auc(
    supported_scores: Sequence[float],
    off_support_scores: Sequence[float],
) -> float:
    positive = np.asarray(supported_scores, dtype=np.float64)
    negative = np.asarray(off_support_scores, dtype=np.float64)
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


def maximum_relation_prediction_share(
    predicted_slots: Sequence[int],
    slot_relations: Sequence[str],
) -> float:
    if not predicted_slots:
        raise ValueError("at least one prediction is required")
    counts: dict[str, int] = {}
    for slot in predicted_slots:
        if not 0 <= slot < len(slot_relations):
            raise IndexError(f"predicted slot {slot} is outside the store")
        relation = slot_relations[slot]
        counts[relation] = counts.get(relation, 0) + 1
    return max(counts.values()) / len(predicted_slots)


def classify_verdict(
    validity: dict[str, bool],
    scientific: dict[str, bool],
    *,
    interrupted: bool = False,
) -> str:
    if not all(validity.values()):
        return "HARNESS_INVALID"
    if interrupted:
        return "INCONCLUSIVE"
    return "PASS" if all(scientific.values()) else "FAIL"
