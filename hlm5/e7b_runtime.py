"""Exact geometry and native-head helpers for the registered E7b study."""

from __future__ import annotations

import math
import platform
from typing import Any

import torch
import transformers

from .e7_contract import MODEL_VOCAB_SIZE, tensor_sha256
from .e7_runtime import native_head_logits


TARGET_CHUNK = 40
DOSE_GRID = 120
DOSE_GRID_CHUNK = 8
HEAD_BATCH = 16
NATIVE_ADMIT = "ADMIT_NATIVE"
REFUSE_HARD = "REFUSE_HARD_BLOCKER"
REFUSE_INTERVAL = "REFUSE_EMPTY_INTERVAL"
REFUSE_TIE = "REFUSE_NATIVE_TIE"
REFUSE_WRONG = "REFUSE_NATIVE_WRONG_ARGMAX"
GEOMETRIC_REACHABLE = "GEOMETRIC_REACHABLE"


def configure_native_runtime() -> dict[str, Any]:
    """Pin and report every registered CUDA arithmetic switch."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    return native_runtime_record()


def native_runtime_record() -> dict[str, Any]:
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast_enabled": torch.is_autocast_enabled(),
    }


def environment_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("registered E7b environment requires CUDA")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "device_count": torch.cuda.device_count(),
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@torch.inference_mode()
def build_slope_matrix(
    head64: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce E7's float64 unit-row directions and slope matrix."""
    if head64.dtype != torch.float64 or target_ids.dtype != torch.long:
        raise TypeError("E7b geometry requires float64 head and long target ids")
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
    if not torch.isfinite(directions).all() or not torch.isfinite(slopes_cpu).all():
        raise RuntimeError("E7b slope construction produced a nonfinite value")
    return directions, slopes_cpu


@torch.inference_mode()
def grid_dose(
    intercept: torch.Tensor,
    slope: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Use E7's exact 120-point open-interval dose grid."""
    if intercept.dtype != torch.float64 or slope.dtype != torch.float64:
        raise TypeError("E7b grid inputs must be float64")
    cap = lower + max(50.0, 5.0 * lower)
    high = min(upper, cap) if math.isfinite(upper) else cap
    if not math.isfinite(lower) or not math.isfinite(high) or not lower < high:
        raise RuntimeError(f"invalid E7b dose interval: ({lower}, {high})")
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
    if not torch.isfinite(betas).all() or not torch.isfinite(worst).all():
        raise RuntimeError("E7b dose grid produced a nonfinite value")
    index = int(worst.argmax())
    beta = float(betas[index])
    margin = float(worst[index])
    if not lower < beta or (math.isfinite(upper) and not beta < upper):
        raise RuntimeError("E7b dose is outside the exact open interval")
    if margin <= 0:
        raise RuntimeError("reachable E7b row has no positive grid margin")
    return beta, margin


@torch.inference_mode()
def geometry_rows(
    head64: torch.Tensor,
    hidden_native: torch.Tensor,
    target_ids: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Classify all targets exhaustively under the exact E7 geometry."""
    if head64.dtype != torch.float64:
        raise TypeError("E7b head must be float64 at the certificate boundary")
    if hidden_native.dtype != torch.bfloat16:
        raise TypeError("E7b prompt hidden must be native bfloat16")
    hidden64 = hidden_native.to(torch.float64)
    if not torch.isfinite(head64).all() or not torch.isfinite(hidden64).all():
        raise RuntimeError("E7b certificate boundary contains nonfinite values")
    directions, slopes_cpu = build_slope_matrix(head64, target_ids)
    base = head64 @ hidden64
    if not torch.isfinite(base).all():
        raise RuntimeError("E7b base logits are nonfinite")

    rows: list[dict[str, Any]] = []
    for start in range(0, target_ids.numel(), TARGET_CHUNK):
        end = min(start + TARGET_CHUNK, target_ids.numel())
        tids = target_ids[start:end]
        slopes = slopes_cpu[:, start:end].to(head64.device)
        intercepts = base[tids].unsqueeze(0) - base.unsqueeze(1)
        local_columns = torch.arange(end - start, device=head64.device)
        intercepts[tids, local_columns] = float("inf")
        positive = slopes > 0
        negative = slopes < 0
        hard_matrix = (slopes <= 0) & (intercepts <= 0)
        hard = hard_matrix.any(dim=0)
        ratios = -intercepts / slopes
        lower = torch.clamp(
            torch.where(positive, ratios, float("-inf")).amax(dim=0),
            min=0.0,
        )
        upper = torch.where(negative, ratios, float("inf")).amin(dim=0)
        if not torch.isfinite(lower).all():
            raise RuntimeError("E7b lower bound is nonfinite")
        if torch.isnan(upper).any() or torch.isneginf(upper).any():
            raise RuntimeError("E7b upper bound has an invalid nonfinite value")

        for local, column in enumerate(range(start, end)):
            target_id = int(target_ids[column])
            lower_value = float(lower[local])
            upper_value = float(upper[local])
            row: dict[str, Any] = {
                "pool_index": column,
                "target_id": target_id,
                "L": lower_value,
                "U": finite_or_none(upper_value),
                "hard_blocker": bool(hard[local]),
                "hard_blocker_id": None,
                "beta": None,
                "float64_margin": None,
            }
            if row["hard_blocker"]:
                blocker_ids = torch.nonzero(
                    hard_matrix[:, local], as_tuple=False
                ).flatten()
                row["hard_blocker_id"] = int(blocker_ids[0])
                row["geometry_class"] = REFUSE_HARD
                row["decision"] = REFUSE_HARD
            elif not lower_value < upper_value:
                row["geometry_class"] = REFUSE_INTERVAL
                row["decision"] = REFUSE_INTERVAL
            else:
                intercept = intercepts[:, local]
                slope = slopes[:, local]
                beta, margin = grid_dose(
                    intercept,
                    slope,
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
        raise RuntimeError("E7b geometry did not classify every target")
    return rows, directions, slopes_cpu


def native_decision(
    *,
    target_id: int,
    prediction: int,
    native_margin: float,
) -> str:
    """Apply the registered strict native-head admission rule."""
    if not math.isfinite(native_margin):
        raise ValueError("native margin must be finite")
    if native_margin == 0.0:
        return REFUSE_TIE
    if native_margin < 0.0 or prediction != target_id:
        return REFUSE_WRONG
    return NATIVE_ADMIT


def _competitor(native: torch.Tensor, target_id: int) -> tuple[int, torch.Tensor]:
    masked = native.clone()
    masked[target_id] = float("-inf")
    competitor_id = int(masked.argmax())
    return competitor_id, native[competitor_id]


@torch.inference_mode()
def native_rows(
    model: Any,
    hidden_native: torch.Tensor,
    directions: torch.Tensor,
    geometry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate the full reachable set in the frozen sorted 16-row batches."""
    if hidden_native.dtype != torch.bfloat16:
        raise TypeError("E7b native admission requires a bfloat16 hidden")
    reachable_indices = [
        index
        for index, row in enumerate(geometry)
        if row["geometry_class"] == GEOMETRIC_REACHABLE
    ]
    if reachable_indices != sorted(
        reachable_indices, key=lambda index: geometry[index]["target_id"]
    ):
        raise RuntimeError("E7b reachable rows are not in sorted target-id order")
    output = [dict(row) for row in geometry]
    modified: list[torch.Tensor] = []
    for index in reachable_indices:
        row = geometry[index]
        beta = row["beta"]
        if not isinstance(beta, float):
            raise RuntimeError("reachable E7b row has no frozen beta")
        edited = (
            hidden_native.to(torch.float64) + beta * directions[index]
        ).to(torch.bfloat16)
        modified.append(edited)

    for start in range(0, len(modified), HEAD_BATCH):
        end = min(start + HEAD_BATCH, len(modified))
        hidden_batch = torch.stack(modified[start:end]).contiguous()
        if hidden_batch.dtype != torch.bfloat16 or hidden_batch.shape[1:] != (
            hidden_native.numel(),
        ):
            raise RuntimeError("E7b native hidden batch contract failed")
        with torch.autocast(device_type="cuda", enabled=False):
            logits_batch = native_head_logits(model, hidden_batch)
        if logits_batch.dtype != torch.bfloat16 or logits_batch.shape != (
            end - start,
            MODEL_VOCAB_SIZE,
        ):
            raise RuntimeError("E7b ordinary head returned the wrong shape or dtype")
        if not torch.isfinite(logits_batch).all():
            raise RuntimeError("E7b ordinary head returned nonfinite logits")

        for offset, reachable_position in enumerate(range(start, end)):
            row_index = reachable_indices[reachable_position]
            row = output[row_index]
            target_id = row["target_id"]
            native = logits_batch[offset].detach().contiguous()
            competitor_id, competitor_logit = _competitor(native, target_id)
            prediction = int(native.argmax())
            target_logit = float(native[target_id].float())
            competitor_value = float(competitor_logit.float())
            margin = float(
                native[target_id].float() - competitor_logit.float()
            )
            decision = native_decision(
                target_id=target_id,
                prediction=prediction,
                native_margin=margin,
            )
            strict = bool(native[target_id] > competitor_logit)
            if (decision == NATIVE_ADMIT) != (strict and prediction == target_id):
                raise RuntimeError("E7b strict native decision is inconsistent")
            row.update(
                {
                    "native_checked": True,
                    "native_target_logit": target_logit,
                    "native_competitor_id": competitor_id,
                    "native_competitor_logit": competitor_value,
                    "native_margin": margin,
                    "native_prediction": prediction,
                    "decision": decision,
                    "native_logits_sha256": tensor_sha256(native),
                    "native_logits_dtype": "bfloat16",
                    "native_logits_shape": [MODEL_VOCAB_SIZE],
                    "native_batch_start": start,
                    "native_batch_size": end - start,
                    "native_batch_offset": offset,
                }
            )
    return output


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        REFUSE_HARD: 0,
        REFUSE_INTERVAL: 0,
        REFUSE_TIE: 0,
        REFUSE_WRONG: 0,
        NATIVE_ADMIT: 0,
    }
    for row in rows:
        decision = row.get("decision")
        if decision not in counts:
            raise RuntimeError(f"unknown E7b decision: {decision}")
        counts[decision] += 1
    geometric = counts[REFUSE_TIE] + counts[REFUSE_WRONG] + counts[NATIVE_ADMIT]
    return {
        "target_count": len(rows),
        "geometrically_reachable": geometric,
        "native_admitted": counts[NATIVE_ADMIT],
        "geometric_coverage": geometric / len(rows) if rows else 0.0,
        "native_coverage": counts[NATIVE_ADMIT] / len(rows) if rows else 0.0,
        "retained_fraction": counts[NATIVE_ADMIT] / geometric if geometric else 0.0,
        "decision_counts": counts,
    }
