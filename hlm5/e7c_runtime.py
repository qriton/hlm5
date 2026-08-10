"""Fixed ZCA direction and dual-arm runtime for E7c confirmation."""

from __future__ import annotations

import math
from typing import Any

import torch

from .e7_contract import tensor_sha256
from .e7b_runtime import (
    GEOMETRIC_REACHABLE,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    TARGET_CHUNK,
    finite_or_none,
    grid_dose,
    native_rows,
)


RIDGE_FRACTION = 0.01


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / (rows.norm(dim=1, keepdim=True) + 1e-12)


@torch.inference_mode()
def zca_basis(
    head_native: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build the frozen float32 covariance basis and float64 ZCA operator."""
    head32 = head_native.to(torch.float32)
    mean32 = head32.mean(dim=0)
    covariance32 = head32.T @ head32 / head32.shape[0]
    covariance32.sub_(torch.outer(mean32, mean32))
    covariance32 = 0.5 * (covariance32 + covariance32.T)
    eigenvalues32, eigenvectors32 = torch.linalg.eigh(covariance32)
    eigenvalues32 = eigenvalues32.clamp_min(0.0)
    positive = eigenvalues32[eigenvalues32 > 0]
    if positive.numel() == 0:
        raise RuntimeError("readout covariance has no positive eigenvalues")
    median_positive = float(positive.median())
    ridge = RIDGE_FRACTION * median_positive
    if not math.isfinite(ridge) or ridge <= 0:
        raise RuntimeError(f"invalid ZCA ridge: {ridge}")
    eigenvectors64 = eigenvectors32.to(torch.float64)
    scale64 = (eigenvalues32.to(torch.float64) + ridge).rsqrt()
    operator64 = (eigenvectors64 * scale64) @ eigenvectors64.T
    if not all(
        torch.isfinite(tensor).all()
        for tensor in (mean32, eigenvalues32, eigenvectors32, operator64)
    ):
        raise RuntimeError("ZCA basis contains nonfinite values")
    return mean32, eigenvalues32, eigenvectors32, operator64, ridge


def basis_record(
    mean32: torch.Tensor,
    eigenvalues32: torch.Tensor,
    eigenvectors32: torch.Tensor,
    operator64: torch.Tensor,
    ridge: float,
) -> dict[str, Any]:
    return {
        "mean_sha256": tensor_sha256(mean32),
        "eigenvalues_sha256": tensor_sha256(eigenvalues32),
        "eigenvectors_sha256": tensor_sha256(eigenvectors32),
        "operator_sha256": tensor_sha256(operator64),
        "median_positive_eigenvalue": ridge / RIDGE_FRACTION,
        "ridge_fraction": RIDGE_FRACTION,
        "ridge": ridge,
        "operator_dtype": "float64",
        "operator_shape": list(operator64.shape),
    }


@torch.inference_mode()
def raw_and_zca_directions(
    head_native: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return raw and frozen raw-row ZCA-half directions in float64."""
    head64 = head_native.to(torch.float64)
    mean32, eigenvalues32, eigenvectors32, operator64, ridge = zca_basis(head_native)
    source = head64[target_ids]
    raw = normalize_rows(source)
    eigenvectors64 = eigenvectors32.to(torch.float64)
    scale64 = (eigenvalues32.to(torch.float64) + ridge).rsqrt()
    zca = normalize_rows((source @ eigenvectors64 * scale64) @ eigenvectors64.T)
    if not torch.isfinite(raw).all() or not torch.isfinite(zca).all():
        raise RuntimeError("E7c direction construction produced nonfinite values")
    record = basis_record(mean32, eigenvalues32, eigenvectors32, operator64, ridge)
    record.update(
        {
            "raw_directions_sha256": tensor_sha256(raw),
            "zca_directions_sha256": tensor_sha256(zca),
            "direction_dtype": "float64",
            "direction_shape": list(raw.shape),
        }
    )
    return raw, zca, record


@torch.inference_mode()
def geometry_for_directions(
    head64: torch.Tensor,
    hidden_native: torch.Tensor,
    target_ids: torch.Tensor,
    directions: torch.Tensor,
) -> list[dict[str, Any]]:
    """Classify exact full-vocabulary affine geometry for fixed directions."""
    if head64.dtype != torch.float64 or directions.dtype != torch.float64:
        raise TypeError("E7c exact geometry requires float64 head and directions")
    if hidden_native.dtype != torch.bfloat16 or target_ids.dtype != torch.long:
        raise TypeError("E7c requires BF16 hidden and long target ids")
    if directions.shape != (target_ids.numel(), head64.shape[1]):
        raise ValueError("E7c candidate direction shape mismatch")
    base = head64 @ hidden_native.to(torch.float64)
    rows: list[dict[str, Any]] = []
    for start in range(0, target_ids.numel(), TARGET_CHUNK):
        end = min(start + TARGET_CHUNK, target_ids.numel())
        tids = target_ids[start:end]
        projected = head64 @ directions[start:end].T
        local = torch.arange(end - start, device=head64.device)
        diagonal = projected[tids, local]
        slopes = diagonal.unsqueeze(0) - projected
        intercepts = base[tids].unsqueeze(0) - base.unsqueeze(1)
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
            raise RuntimeError("E7c lower bound is nonfinite")
        if torch.isnan(upper).any() or torch.isneginf(upper).any():
            raise RuntimeError("E7c upper bound has an invalid nonfinite value")
        for offset, column in enumerate(range(start, end)):
            lower_value = float(lower[offset])
            upper_value = float(upper[offset])
            row: dict[str, Any] = {
                "pool_index": column,
                "target_id": int(target_ids[column]),
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
    if len(rows) != target_ids.numel():
        raise RuntimeError("E7c did not classify every target")
    return rows


@torch.inference_mode()
def evaluate_arm(
    model: Any,
    head64: torch.Tensor,
    hidden_native: torch.Tensor,
    target_ids: torch.Tensor,
    directions: torch.Tensor,
) -> list[dict[str, Any]]:
    geometry = geometry_for_directions(head64, hidden_native, target_ids, directions)
    return native_rows(model, hidden_native, directions, geometry)


__all__ = [
    "RIDGE_FRACTION",
    "basis_record",
    "evaluate_arm",
    "geometry_for_directions",
    "normalize_rows",
    "raw_and_zca_directions",
    "zca_basis",
]
