"""Float64 key operators, source calibration, and registered SCA-1 metrics."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from research.natural_key_transfer_gate.geometry import (
    canonicalize_eigenvectors,
    population_covariance,
    route_scores,
    support_geometry,
    tensor_sha256,
    unit_rows,
)

from .contract import (
    ARM_NAMES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CALIBRATION_NEGATIVE_QUANTILE,
    CALIBRATION_QUANTILE_METHOD,
    CANDIDATE_ARM,
    PASS_BARS,
    RANDOM_BASIS_SEED,
    SHRINKAGE,
)


@dataclass(frozen=True)
class KeyOperators:
    mean: torch.Tensor
    transforms: dict[str, torch.Tensor | None]
    metadata: dict[str, Any]


def registered_random_basis(dimension: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if dimension <= 0:
        raise ValueError("random-basis dimension must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_BASIS_SEED)
    gaussian = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    basis, upper = torch.linalg.qr(gaussian, mode="complete")
    signs = torch.where(
        torch.diag(upper) < 0.0,
        -torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    basis = basis * signs.unsqueeze(0)
    identity = torch.eye(dimension, dtype=torch.float64)
    metadata = {
        "dimension": dimension,
        "seed": RANDOM_BASIS_SEED,
        "generator_device": "cpu",
        "dtype": "torch.float64",
        "construction": "standard_normal_complete_qr_diag_r_nonnegative",
        "basis_sha256": tensor_sha256(basis),
        "orthogonality_max_abs_error": float(
            torch.max(torch.abs(basis.T @ basis - identity)).item()
        ),
    }
    return basis, metadata


def _matrix(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _operator_metadata(
    name: str,
    transform: torch.Tensor,
    covariance: torch.Tensor,
) -> dict[str, Any]:
    identity = torch.eye(transform.shape[0], dtype=torch.float64)
    error = float(
        torch.max(torch.abs(transform @ covariance @ transform - identity)).item()
    )
    singular_values = torch.linalg.svdvals(transform)
    return {
        "name": name,
        "dimension": int(transform.shape[0]),
        "transform_sha256": tensor_sha256(transform),
        "singular_values_sha256": tensor_sha256(singular_values),
        "singular_value_min": float(singular_values.min().item()),
        "singular_value_median": float(torch.quantile(singular_values, 0.5).item()),
        "singular_value_max": float(singular_values.max().item()),
        "inverse_identity_max_abs_error": error,
    }


def fit_key_operators(source_values: torch.Tensor) -> KeyOperators:
    values = _matrix("source_values", source_values)
    mean = values.mean(dim=0)
    covariance = population_covariance(values - mean)
    dimension = covariance.shape[0]
    trace_mean = torch.trace(covariance) / float(dimension)
    if not float(trace_mean.item()) > 0.0:
        raise ValueError("source covariance trace must be positive")
    identity = torch.eye(dimension, dtype=torch.float64)
    shrunk = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * trace_mean * identity
    shrunk = 0.5 * (shrunk + shrunk.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(shrunk)
    eigenvectors = canonicalize_eigenvectors(eigenvectors)
    if not float(eigenvalues.min().item()) > 0.0:
        raise ValueError("shrunk covariance is not positive definite")
    inverse_scales = torch.rsqrt(eigenvalues)

    blind = eigenvectors @ torch.diag(inverse_scales) @ eigenvectors.T
    blind = 0.5 * (blind + blind.T)
    diagonal = torch.diag(torch.rsqrt(torch.diag(shrunk)))

    random_basis, random_basis_metadata = registered_random_basis(dimension)
    random_transform = random_basis @ torch.diag(inverse_scales) @ random_basis.T
    random_transform = 0.5 * (random_transform + random_transform.T)

    blind_singular = torch.linalg.svdvals(blind)
    random_singular = torch.linalg.svdvals(random_transform)
    spectrum_max_abs_error = float(
        torch.max(torch.abs(blind_singular - random_singular)).item()
    )
    spectrum_max_rel_error = float(
        torch.max(
            torch.abs(blind_singular - random_singular)
            / torch.clamp(
                torch.abs(blind_singular), min=torch.finfo(torch.float64).tiny
            )
        ).item()
    )

    transforms: dict[str, torch.Tensor | None] = {
        "raw": None,
        "centered": identity,
        "diagonal_std": diagonal,
        "random_basis_zca": random_transform,
        "blind_zca": blind,
    }
    metadata = {
        "source_count": int(values.shape[0]),
        "dimension": dimension,
        "shrinkage": SHRINKAGE,
        "mean_sha256": tensor_sha256(mean),
        "covariance_sha256": tensor_sha256(covariance),
        "shrunk_covariance_sha256": tensor_sha256(shrunk),
        "trace_mean": float(trace_mean.item()),
        "eigenvalue_min": float(eigenvalues.min().item()),
        "eigenvalue_median": float(torch.quantile(eigenvalues, 0.5).item()),
        "eigenvalue_max": float(eigenvalues.max().item()),
        "condition_number": float((eigenvalues.max() / eigenvalues.min()).item()),
        "eigenvectors_sha256": tensor_sha256(eigenvectors),
        "random_basis_seed": RANDOM_BASIS_SEED,
        "random_basis_sha256": tensor_sha256(random_basis),
        "random_basis_orthogonality_max_abs_error": random_basis_metadata[
            "orthogonality_max_abs_error"
        ],
        "basis_hashes_differ": tensor_sha256(eigenvectors)
        != tensor_sha256(random_basis),
        "blind_zca": _operator_metadata("blind_zca", blind, shrunk),
        "diagonal_std": {
            "name": "diagonal_std",
            "transform_sha256": tensor_sha256(diagonal),
            "diagonal_min": float(torch.diag(diagonal).min().item()),
            "diagonal_median": float(torch.quantile(torch.diag(diagonal), 0.5).item()),
            "diagonal_max": float(torch.diag(diagonal).max().item()),
        },
        "random_basis_zca": _operator_metadata(
            "random_basis_zca", random_transform, shrunk
        ),
        "matched_spectrum_max_abs_error": spectrum_max_abs_error,
        "matched_spectrum_max_rel_error": spectrum_max_rel_error,
    }
    return KeyOperators(mean=mean, transforms=transforms, metadata=metadata)


def apply_arm(
    values: torch.Tensor,
    operators: KeyOperators,
    arm: str,
) -> torch.Tensor:
    values = _matrix("values", values)
    if arm not in ARM_NAMES:
        raise KeyError(arm)
    if arm == "raw":
        transformed = values
    else:
        transform = operators.transforms[arm]
        if transform is None:
            raise AssertionError(f"arm {arm} has no transform")
        transformed = (values - operators.mean) @ transform
    return unit_rows(transformed)


def calibration_threshold(maximum_negative_scores: torch.Tensor) -> float:
    values = (
        maximum_negative_scores.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .numpy()
    )
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("calibration negatives must be finite and one-dimensional")
    return float(
        np.quantile(
            values,
            CALIBRATION_NEGATIVE_QUANTILE,
            method=CALIBRATION_QUANTILE_METHOD,
        )
    )


def admission_mask(maximum_scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return maximum_scores.detach().to(device="cpu", dtype=torch.float64) > threshold


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    return {
        "0.05": float(torch.quantile(values, 0.05).item()),
        "0.5": float(torch.quantile(values, 0.5).item()),
        "0.95": float(torch.quantile(values, 0.95).item()),
    }


def summarize_supported(
    routed: dict[str, Any],
    query_labels: list[str],
    support_labels: list[str],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(query_labels) != len(routed["slots"]):
        raise ValueError("query labels do not match routed rows")
    predicted = [support_labels[int(slot)] for slot in routed["slots"].tolist()]
    correct = torch.tensor(
        [guess == truth for guess, truth in zip(predicted, query_labels, strict=True)],
        dtype=torch.bool,
    )
    admitted = admission_mask(routed["maximum_scores"], threshold)
    correct_admitted = correct & admitted

    per_intent: dict[str, dict[str, float | int]] = {}
    for intent in sorted(set(query_labels)):
        mask = torch.tensor(
            [label == intent for label in query_labels], dtype=torch.bool
        )
        count = int(mask.sum().item())
        per_intent[intent] = {
            "count": count,
            "accuracy": float(correct[mask].to(dtype=torch.float64).mean().item()),
            "coverage": float(admitted[mask].to(dtype=torch.float64).mean().item()),
            "correct_admission_rate": float(
                correct_admitted[mask].to(dtype=torch.float64).mean().item()
            ),
        }

    predicted_counts = collections.Counter(predicted)
    admitted_count = int(admitted.sum().item())
    true_best = torch.empty(len(query_labels), dtype=torch.float64)
    false_best = torch.empty(len(query_labels), dtype=torch.float64)
    cosine = routed["cosine"].to(dtype=torch.float64, device="cpu")
    for index, truth in enumerate(query_labels):
        truth_mask = torch.tensor(
            [label == truth for label in support_labels], dtype=torch.bool
        )
        if not bool(truth_mask.any()) or not bool((~truth_mask).any()):
            raise ValueError("supported summary needs true and false support slots")
        true_best[index] = torch.max(cosine[index, truth_mask])
        false_best[index] = torch.max(cosine[index, ~truth_mask])
    margin = true_best - false_best

    metrics = {
        "row_count": len(query_labels),
        "threshold": threshold,
        "top1_accuracy": float(correct.to(dtype=torch.float64).mean().item()),
        "coverage": float(admitted.to(dtype=torch.float64).mean().item()),
        "correct_admission_rate": float(
            correct_admitted.to(dtype=torch.float64).mean().item()
        ),
        "macro_top1_accuracy": float(
            np.mean([item["accuracy"] for item in per_intent.values()])
        ),
        "macro_coverage": float(
            np.mean([item["coverage"] for item in per_intent.values()])
        ),
        "macro_correct_admission_rate": float(
            np.mean([item["correct_admission_rate"] for item in per_intent.values()])
        ),
        "selective_accuracy": (
            float(correct[admitted].to(dtype=torch.float64).mean().item())
            if admitted_count
            else 0.0
        ),
        "admitted_count": admitted_count,
        "correct_admitted_count": int(correct_admitted.sum().item()),
        "maximum_predicted_intent_share": max(predicted_counts.values())
        / float(len(predicted)),
        "maximum_predicted_intent": sorted(
            predicted_counts, key=lambda label: (-predicted_counts[label], label)
        )[0],
        "maximum_score_quantiles": _quantiles(routed["maximum_scores"]),
        "true_vs_best_false_cosine_margin_mean": float(margin.mean().item()),
        "true_vs_best_false_cosine_margin_median": float(
            torch.quantile(margin, 0.5).item()
        ),
        "per_intent": per_intent,
        "independent_max_abs_error": routed["independent_max_abs_error"],
        "scores_sha256": routed["scores_sha256"],
    }
    arrays = {
        "predicted": predicted,
        "correct": correct,
        "admitted": admitted,
        "correct_admitted": correct_admitted,
        "maximum_scores": routed["maximum_scores"].to(dtype=torch.float64),
        "maximum_cosines": routed["maximum_cosines"].to(dtype=torch.float64),
    }
    return metrics, arrays


def summarize_off_support(
    routed: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    admitted = admission_mask(routed["maximum_scores"], threshold)
    metrics = {
        "row_count": int(len(admitted)),
        "threshold": threshold,
        "false_admission_rate": float(admitted.to(dtype=torch.float64).mean().item()),
        "admitted_count": int(admitted.sum().item()),
        "maximum_score_quantiles": _quantiles(routed["maximum_scores"]),
        "independent_max_abs_error": routed["independent_max_abs_error"],
        "scores_sha256": routed["scores_sha256"],
    }
    arrays = {
        "admitted": admitted,
        "maximum_scores": routed["maximum_scores"].to(dtype=torch.float64),
        "maximum_cosines": routed["maximum_cosines"].to(dtype=torch.float64),
        "slots": routed["slots"].to(dtype=torch.long),
    }
    return metrics, arrays


def evaluate_arm(
    operators: KeyOperators,
    arm: str,
    calibration_support_values: torch.Tensor,
    calibration_support_labels: list[str],
    calibration_positive_values: torch.Tensor,
    calibration_positive_labels: list[str],
    calibration_negative_values: torch.Tensor,
    target_support_values: torch.Tensor,
    target_support_labels: list[str],
    target_values: torch.Tensor,
    target_labels: list[str],
    off_support_values: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_support = apply_arm(calibration_support_values, operators, arm)
    calibration_positive = apply_arm(calibration_positive_values, operators, arm)
    calibration_negative = apply_arm(calibration_negative_values, operators, arm)
    target_support = apply_arm(target_support_values, operators, arm)
    target = apply_arm(target_values, operators, arm)
    off_support = apply_arm(off_support_values, operators, arm)

    calibration_negative_route = route_scores(calibration_negative, calibration_support)
    threshold = calibration_threshold(calibration_negative_route["maximum_scores"])
    calibration_negative_metrics, calibration_negative_arrays = summarize_off_support(
        calibration_negative_route, threshold
    )
    calibration_positive_metrics, calibration_positive_arrays = summarize_supported(
        route_scores(calibration_positive, calibration_support),
        calibration_positive_labels,
        calibration_support_labels,
        threshold,
    )
    target_metrics, target_arrays = summarize_supported(
        route_scores(target, target_support),
        target_labels,
        target_support_labels,
        threshold,
    )
    off_support_metrics, off_support_arrays = summarize_off_support(
        route_scores(off_support, target_support), threshold
    )
    metrics = {
        "arm": arm,
        "threshold": threshold,
        "calibration_negative": calibration_negative_metrics,
        "calibration_positive": calibration_positive_metrics,
        "target": target_metrics,
        "off_support_audit": off_support_metrics,
        "calibration_support_geometry": support_geometry(
            calibration_support, calibration_support_labels
        ),
        "target_support_geometry": support_geometry(
            target_support, target_support_labels
        ),
    }
    arrays = {
        "calibration_negative": calibration_negative_arrays,
        "calibration_positive": calibration_positive_arrays,
        "target": target_arrays,
        "off_support_audit": off_support_arrays,
        "calibration_support_keys": calibration_support,
        "target_support_keys": target_support,
        "target_keys": target,
    }
    return metrics, arrays


def paired_intent_bootstrap(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
) -> dict[str, Any]:
    candidate_per_intent = candidate["target"]["per_intent"]
    comparator_per_intent = comparator["target"]["per_intent"]
    intents = sorted(candidate_per_intent)
    if intents != sorted(comparator_per_intent):
        raise ValueError("candidate and comparator intent sets differ")
    differences = np.asarray(
        [
            candidate_per_intent[intent]["correct_admission_rate"]
            - comparator_per_intent[intent]["correct_admission_rate"]
            for intent in intents
        ],
        dtype=np.float64,
    )
    point = float(np.mean(differences, dtype=np.float64))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(intents),
        size=(BOOTSTRAP_RESAMPLES, len(intents)),
        endpoint=False,
    )
    samples = np.mean(differences[indices], axis=1, dtype=np.float64)
    interval = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "metric": "macro_correct_admission_rate",
        "intents": intents,
        "point": point,
        "interval": [float(interval[0]), float(interval[1])],
        "improved": int(np.sum(differences > 0.0)),
        "tied": int(np.sum(differences == 0.0)),
        "worsened": int(np.sum(differences < 0.0)),
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
    }


def development_verdict(
    arms: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    candidate = arms[CANDIDATE_ARM]
    raw = arms["raw"]
    bars = {
        "candidate_correct_admission_macro": candidate["target"][
            "macro_correct_admission_rate"
        ]
        >= PASS_BARS["candidate_correct_admission_macro"],
        "candidate_minus_raw_correct_admission_macro": comparisons["raw"]["point"]
        >= PASS_BARS["candidate_minus_raw_correct_admission_macro"],
        "candidate_minus_raw_interval": comparisons["raw"]["interval"][0]
        > PASS_BARS["paired_interval_lower_strictly_greater_than"],
        "candidate_minus_diagonal_correct_admission_macro": comparisons["diagonal_std"][
            "point"
        ]
        >= PASS_BARS["candidate_minus_diagonal_correct_admission_macro"],
        "candidate_minus_diagonal_interval": comparisons["diagonal_std"]["interval"][0]
        > PASS_BARS["paired_interval_lower_strictly_greater_than"],
        "candidate_minus_random_basis_correct_admission_macro": comparisons[
            "random_basis_zca"
        ]["point"]
        >= PASS_BARS["candidate_minus_random_basis_correct_admission_macro"],
        "candidate_minus_random_basis_interval": comparisons["random_basis_zca"][
            "interval"
        ][0]
        > PASS_BARS["paired_interval_lower_strictly_greater_than"],
        "candidate_target_coverage_macro": candidate["target"]["macro_coverage"]
        >= PASS_BARS["candidate_target_coverage_macro"],
        "candidate_selective_accuracy": candidate["target"]["selective_accuracy"]
        >= PASS_BARS["candidate_selective_accuracy"],
        "candidate_off_support_false_admission_rate": candidate["off_support_audit"][
            "false_admission_rate"
        ]
        <= PASS_BARS["candidate_off_support_false_admission_rate"],
        "candidate_max_predicted_intent_share": candidate["target"][
            "maximum_predicted_intent_share"
        ]
        <= PASS_BARS["candidate_max_predicted_intent_share"],
        "candidate_top1_noninferiority": candidate["target"]["macro_top1_accuracy"]
        >= raw["target"]["macro_top1_accuracy"]
        - PASS_BARS["candidate_top1_accuracy_drop_vs_raw"],
    }
    return ("PASS_TO_TEST" if all(bars.values()) else "STOP_BEFORE_TEST"), bars
