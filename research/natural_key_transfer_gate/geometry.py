"""Float64 key geometry, HLM5 routing, and registered NKT-1 metrics."""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from .contract import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEGREE,
    DEPLOYED_TAU_COS,
    PASS_BARS,
    SHRINKAGE,
)


ARM_NAMES = (
    "raw",
    "centered",
    "blind_zca",
    "shuffled_within",
    "source_fisher",
)


@dataclass(frozen=True)
class KeyOperators:
    mean: torch.Tensor
    transforms: dict[str, torch.Tensor | None]
    metadata: dict[str, Any]


def _require_matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    header = f"{array.dtype.str}|{array.shape}|".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def label_sha256(labels: Iterable[str]) -> str:
    payload = "\0".join(labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonicalize_eigenvectors(vectors: torch.Tensor) -> torch.Tensor:
    vectors = vectors.clone()
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        pivot = int(torch.argmax(torch.abs(vector)).item())
        if float(vector[pivot].item()) < 0.0:
            vectors[:, column].neg_()
    return vectors


def population_covariance(residuals: torch.Tensor) -> torch.Tensor:
    residuals = _require_matrix("residuals", residuals)
    return (residuals.T @ residuals) / float(residuals.shape[0])


def within_residuals(
    values: torch.Tensor,
    labels: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    values = _require_matrix("values", values)
    if len(labels) != values.shape[0]:
        raise ValueError("labels do not match values")
    class_means: dict[str, torch.Tensor] = {}
    residuals = torch.empty_like(values)
    for label in sorted(set(labels)):
        indices = [index for index, item in enumerate(labels) if item == label]
        if not indices:
            raise AssertionError("empty label group")
        index_tensor = torch.tensor(indices, dtype=torch.long)
        mean = values[index_tensor].mean(dim=0)
        class_means[label] = mean
        residuals[index_tensor] = values[index_tensor] - mean
    return residuals, class_means


def fit_inverse_root(
    covariance: torch.Tensor,
    shrinkage: float = SHRINKAGE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    covariance = _require_matrix("covariance", covariance)
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not 0.0 < shrinkage <= 1.0:
        raise ValueError("shrinkage must be in (0, 1]")
    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    trace_mean = torch.trace(covariance) / float(dimension)
    if not float(trace_mean.item()) > 0.0:
        raise ValueError("covariance trace mean must be positive")
    identity = torch.eye(dimension, dtype=torch.float64)
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * identity
    eigenvalues, eigenvectors = torch.linalg.eigh(shrunk)
    eigenvectors = canonicalize_eigenvectors(eigenvectors)
    if not float(eigenvalues.min().item()) > 0.0:
        raise ValueError("shrunk covariance is not positive definite")
    inverse_root = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = 0.5 * (inverse_root + inverse_root.T)
    identity_error = float(
        torch.max(torch.abs(inverse_root @ shrunk @ inverse_root - identity)).item()
    )
    metadata = {
        "dimension": dimension,
        "shrinkage": shrinkage,
        "trace_mean": float(trace_mean.item()),
        "eigenvalue_min": float(eigenvalues.min().item()),
        "eigenvalue_median": float(torch.quantile(eigenvalues, 0.5).item()),
        "eigenvalue_max": float(eigenvalues.max().item()),
        "condition_number": float((eigenvalues.max() / eigenvalues.min()).item()),
        "inverse_identity_max_abs_error": identity_error,
        "inverse_root_sha256": tensor_sha256(inverse_root),
        "eigenvectors_sha256": tensor_sha256(eigenvectors),
    }
    return inverse_root, metadata


def fit_key_operators(
    source_values: torch.Tensor,
    source_labels: list[str],
    shuffled_labels: list[str],
) -> KeyOperators:
    values = _require_matrix("source_values", source_values)
    if len(source_labels) != values.shape[0]:
        raise ValueError("source labels do not match source values")
    if len(shuffled_labels) != values.shape[0]:
        raise ValueError("shuffled labels do not match source values")
    if collections.Counter(source_labels) != collections.Counter(shuffled_labels):
        raise ValueError("shuffled label multiset changed")

    mean = values.mean(dim=0)
    global_covariance = population_covariance(values - mean)
    within, class_means = within_residuals(values, source_labels)
    shuffled, shuffled_means = within_residuals(values, shuffled_labels)
    within_covariance = population_covariance(within)
    shuffled_covariance = population_covariance(shuffled)

    blind_transform, blind_metadata = fit_inverse_root(global_covariance)
    shuffled_transform, shuffled_metadata = fit_inverse_root(shuffled_covariance)
    fisher_transform, fisher_metadata = fit_inverse_root(within_covariance)
    identity = torch.eye(values.shape[1], dtype=torch.float64)
    transforms = {
        "raw": None,
        "centered": identity,
        "blind_zca": blind_transform,
        "shuffled_within": shuffled_transform,
        "source_fisher": fisher_transform,
    }
    metadata = {
        "source_count": values.shape[0],
        "source_intent_count": len(class_means),
        "shuffled_intent_count": len(shuffled_means),
        "source_labels_sha256": label_sha256(source_labels),
        "shuffled_labels_sha256": label_sha256(shuffled_labels),
        "mean_sha256": tensor_sha256(mean),
        "blind_zca": blind_metadata,
        "shuffled_within": shuffled_metadata,
        "source_fisher": fisher_metadata,
    }
    return KeyOperators(mean=mean, transforms=transforms, metadata=metadata)


def unit_rows(values: torch.Tensor) -> torch.Tensor:
    values = _require_matrix("values", values)
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if not bool((norms > 0.0).all()):
        raise ValueError("zero row cannot be normalized")
    return values / norms


def apply_arm(
    values: torch.Tensor,
    operators: KeyOperators,
    arm: str,
) -> torch.Tensor:
    values = _require_matrix("values", values)
    if arm not in ARM_NAMES:
        raise KeyError(arm)
    if arm == "raw":
        transformed = values
    else:
        transformed = values - operators.mean
        transform = operators.transforms[arm]
        if transform is None:
            raise AssertionError(f"arm {arm} has no transform")
        transformed = transformed @ transform
    return unit_rows(transformed)


def route_scores(
    query_keys: torch.Tensor,
    support_keys: torch.Tensor,
) -> dict[str, Any]:
    query_keys = _require_matrix("query_keys", query_keys)
    support_keys = _require_matrix("support_keys", support_keys)
    if query_keys.shape[1] != support_keys.shape[1]:
        raise ValueError("query and support dimensions differ")
    cosine = query_keys @ support_keys.T
    scores = torch.relu(cosine).pow(DEGREE)
    maximum_scores, slots = torch.max(scores, dim=1)
    maximum_cosines, cosine_slots = torch.max(cosine, dim=1)
    abstentions = maximum_scores == 0.0
    positive = maximum_cosines > 0.0
    if not torch.equal(slots[positive], cosine_slots[positive]):
        raise AssertionError("degree-five and cosine readers disagree")

    numpy_cosine = query_keys.numpy() @ support_keys.numpy().T
    numpy_scores = np.maximum(numpy_cosine, 0.0) ** DEGREE
    numpy_slots = np.argmax(numpy_scores, axis=1)
    maximum_score_error = float(
        np.max(np.abs(numpy_scores - scores.numpy()), initial=0.0)
    )
    if not np.array_equal(numpy_slots, slots.numpy()):
        raise AssertionError("independent NumPy scorer changed a prediction")
    if maximum_score_error > 1e-10:
        raise AssertionError(
            f"independent score error {maximum_score_error} exceeds 1e-10"
        )
    return {
        "cosine": cosine,
        "scores": scores,
        "slots": slots,
        "maximum_scores": maximum_scores,
        "maximum_cosines": maximum_cosines,
        "abstentions": abstentions,
        "independent_max_abs_error": maximum_score_error,
        "scores_sha256": tensor_sha256(scores),
    }


def binary_auc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    positive = positive.detach().to(dtype=torch.float64, device="cpu").numpy()
    negative = negative.detach().to(dtype=torch.float64, device="cpu").numpy()
    if (
        positive.ndim != 1
        or negative.ndim != 1
        or not len(positive)
        or not len(negative)
    ):
        raise ValueError("AUROC requires non-empty one-dimensional samples")
    values = np.concatenate([negative, positive])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_ranks = ranks[len(negative) :]
    u_statistic = positive_ranks.sum() - len(positive) * (len(positive) + 1) / 2.0
    return float(u_statistic / (len(positive) * len(negative)))


def support_geometry(
    support_keys: torch.Tensor,
    support_labels: list[str],
) -> dict[str, Any]:
    support_keys = _require_matrix("support_keys", support_keys)
    if len(support_labels) != support_keys.shape[0]:
        raise ValueError("support labels do not match support keys")
    gram = support_keys @ support_keys.T
    eigenvalues = torch.linalg.eigvalsh(0.5 * (gram + gram.T))
    minimum = float(eigenvalues.min().item())
    maximum = float(eigenvalues.max().item())
    condition = maximum / minimum if minimum > 0.0 else None
    within: list[float] = []
    between: list[float] = []
    for left in range(len(support_labels)):
        for right in range(left + 1, len(support_labels)):
            value = float(gram[left, right].item())
            if support_labels[left] == support_labels[right]:
                within.append(value)
            else:
                between.append(value)
    return {
        "gram_sha256": tensor_sha256(gram),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "condition_number": condition,
        "within_intent_cosine_mean": float(np.mean(within, dtype=np.float64)),
        "within_intent_cosine_median": float(np.median(within)),
        "between_intent_cosine_mean": float(np.mean(between, dtype=np.float64)),
        "between_intent_cosine_median": float(np.median(between)),
    }


def evaluate_arm(
    query_keys: torch.Tensor,
    support_keys: torch.Tensor,
    query_labels: list[str],
    support_labels: list[str],
    off_support_keys: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(query_labels) != query_keys.shape[0]:
        raise ValueError("query labels do not match query keys")
    if len(support_labels) != support_keys.shape[0]:
        raise ValueError("support labels do not match support keys")
    routed = route_scores(query_keys, support_keys)
    off_support = route_scores(off_support_keys, support_keys)

    predictions: list[str | None] = []
    rows: list[dict[str, Any]] = []
    true_margins: list[float] = []
    for index, true_label in enumerate(query_labels):
        abstained = bool(routed["abstentions"][index].item())
        slot = int(routed["slots"][index].item())
        predicted = None if abstained else support_labels[slot]
        predictions.append(predicted)
        true_mask = torch.tensor(
            [label == true_label for label in support_labels], dtype=torch.bool
        )
        false_mask = ~true_mask
        true_cosine = float(routed["cosine"][index, true_mask].max().item())
        false_cosine = float(routed["cosine"][index, false_mask].max().item())
        margin = true_cosine - false_cosine
        true_margins.append(margin)
        rows.append(
            {
                "true_intent": true_label,
                "predicted_intent": predicted,
                "correct": predicted == true_label,
                "abstained": abstained,
                "top_slot": slot,
                "top_cosine": float(routed["maximum_cosines"][index].item()),
                "top_degree5_score": float(routed["maximum_scores"][index].item()),
                "true_vs_best_false_cosine_margin": margin,
                "admitted_at_fixed_threshold": bool(
                    routed["maximum_scores"][index].item() >= DEPLOYED_TAU_COS**DEGREE
                ),
            }
        )

    observed_intents = sorted(set(query_labels))
    per_intent = {}
    for intent in observed_intents:
        indices = [index for index, label in enumerate(query_labels) if label == intent]
        per_intent[intent] = float(
            np.mean(
                [predictions[index] == query_labels[index] for index in indices],
                dtype=np.float64,
            )
        )
    prediction_counts = collections.Counter(
        prediction for prediction in predictions if prediction is not None
    )
    max_share = max(prediction_counts.values(), default=0) / float(len(predictions))
    metrics = {
        "row_accuracy": float(
            np.mean(
                [
                    prediction == truth
                    for prediction, truth in zip(predictions, query_labels)
                ],
                dtype=np.float64,
            )
        ),
        "macro_accuracy": float(np.mean(list(per_intent.values()), dtype=np.float64)),
        "per_intent_accuracy": per_intent,
        "observed_intents": observed_intents,
        "abstentions": int(sum(prediction is None for prediction in predictions)),
        "maximum_predicted_intent_share": max_share,
        "true_vs_best_false_cosine_margin_mean": float(
            np.mean(true_margins, dtype=np.float64)
        ),
        "true_vs_best_false_cosine_margin_median": float(np.median(true_margins)),
        "fixed_threshold_admission_fraction": float(
            np.mean(
                [row["admitted_at_fixed_threshold"] for row in rows],
                dtype=np.float64,
            )
        ),
        "supported_off_support_auroc": binary_auc(
            routed["maximum_scores"], off_support["maximum_scores"]
        ),
        "supported_top_score_quantiles": {
            str(q): float(
                np.quantile(routed["maximum_scores"].numpy(), q, method="linear")
            )
            for q in (0.05, 0.5, 0.95)
        },
        "off_support_top_score_quantiles": {
            str(q): float(
                np.quantile(off_support["maximum_scores"].numpy(), q, method="linear")
            )
            for q in (0.05, 0.5, 0.95)
        },
        "independent_max_abs_error": max(
            routed["independent_max_abs_error"],
            off_support["independent_max_abs_error"],
        ),
        "query_scores_sha256": routed["scores_sha256"],
        "off_support_scores_sha256": off_support["scores_sha256"],
        "support_geometry": support_geometry(support_keys, support_labels),
    }
    return metrics, rows


def paired_intent_bootstrap(
    candidate: dict[str, float],
    comparator: dict[str, float],
) -> dict[str, Any]:
    intents = sorted(candidate)
    if intents != sorted(comparator):
        raise ValueError("paired intent sets differ")
    differences = np.asarray(
        [candidate[intent] - comparator[intent] for intent in intents],
        dtype=np.float64,
    )
    point = float(np.mean(differences, dtype=np.float64))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_RESAMPLES, len(differences)),
        endpoint=False,
    )
    samples = np.mean(differences[indices], axis=1, dtype=np.float64)
    interval = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "intents": intents,
        "point": point,
        "interval": [float(interval[0]), float(interval[1])],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "improved": int(np.sum(differences > 0.0)),
        "tied": int(np.sum(differences == 0.0)),
        "worsened": int(np.sum(differences < 0.0)),
    }


def development_verdict(
    arms: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    candidate = arms["source_fisher"]
    raw = arms["raw"]
    bars = {
        "candidate_minus_raw": comparisons["raw"]["point"]
        >= PASS_BARS["candidate_minus_raw"],
        "candidate_minus_raw_interval": comparisons["raw"]["interval"][0]
        > PASS_BARS["interval_lower_strictly_greater_than"],
        "candidate_minus_blind_zca": comparisons["blind_zca"]["point"]
        >= PASS_BARS["candidate_minus_blind_zca"],
        "candidate_minus_blind_zca_interval": comparisons["blind_zca"]["interval"][0]
        > PASS_BARS["interval_lower_strictly_greater_than"],
        "candidate_minus_shuffled_within": comparisons["shuffled_within"]["point"]
        >= PASS_BARS["candidate_minus_shuffled_within"],
        "candidate_minus_shuffled_within_interval": comparisons["shuffled_within"][
            "interval"
        ][0]
        > PASS_BARS["interval_lower_strictly_greater_than"],
        "candidate_max_predicted_intent_share": candidate[
            "maximum_predicted_intent_share"
        ]
        <= PASS_BARS["candidate_max_predicted_intent_share"],
        "candidate_auroc_noninferiority": candidate["supported_off_support_auroc"]
        >= raw["supported_off_support_auroc"]
        - PASS_BARS["candidate_auroc_drop_vs_raw"],
    }
    return ("PASS_TO_TEST" if all(bars.values()) else "STOP_BEFORE_TEST"), bars
