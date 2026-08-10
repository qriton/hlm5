"""Screen fixed readout preconditioners on the already-spent E7b pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from hlm5.e7_contract import (  # noqa: E402
    MODEL_REVISION,
    sha256_file,
    snapshot_path,
    tensor_sha256,
)
from hlm5.e7_runtime import (  # noqa: E402
    assert_model_invariants,
    final_hidden_batch,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_contract import (  # noqa: E402
    ADMISSION_PATH,
    PROMPT,
    PROMPT_SHA256,
    RESULT_PATH,
    VERDICT_PATH,
    load_admission,
    load_json_object,
    load_result,
    scientific_sha256,
    stable_json_sha256,
)
from hlm5.e7b_runtime import (  # noqa: E402
    GEOMETRIC_REACHABLE,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    TARGET_CHUNK,
    configure_native_runtime,
    environment_record,
    finite_or_none,
    grid_dose,
    native_rows,
    summarize_rows,
)
from hlm5.io import RESULTS_DIR  # noqa: E402


ADMISSION_SHA256 = "60249139741d79c3995205bc09c04e0f68067155098abff7d4c1d9d08e828b39"
RESULT_SHA256 = "9a5232b33f8828195794cb137ee2ec83e10ec8af524418abe3608cf0ac2d4d11"
VERDICT_SHA256 = "5f5e35f71d432d2df0287c7dec1350e8b53c3dae7bea0f025634536a3a6fd016"
SPLIT_SALT = "hlm5-e7c-spent-split-v1"
EXPECTED_SPLIT_COUNTS = {"development": 583, "validation": 617}
RIDGE_FRACTION = 0.01
REPORT_PATH = RESULTS_DIR / "e7c_spent_geometry_screen.json"
SCREEN_SOURCE_PATHS = (
    "docs/e7c-spent-geometry-screen-protocol-2026-08-10.md",
    "scripts/analyze_e7c_spent_geometry.py",
    "tests/test_e7c_spent_geometry.py",
)

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "raw_unit", "kind": "identity", "centered": False},
    {"name": "centered_unit", "kind": "identity", "centered": True},
    {
        "name": "diag_whiten_centered_r1e2",
        "kind": "diagonal",
        "centered": True,
        "power": 0.5,
    },
    {
        "name": "zca_half_raw_r1e2",
        "kind": "spectral",
        "centered": False,
        "power": 0.5,
    },
    {
        "name": "zca_half_centered_r1e2",
        "kind": "spectral",
        "centered": True,
        "power": 0.5,
    },
    {
        "name": "mahalanobis_raw_r1e2",
        "kind": "spectral",
        "centered": False,
        "power": 1.0,
    },
    {
        "name": "mahalanobis_centered_r1e2",
        "kind": "spectral",
        "centered": True,
        "power": 1.0,
    },
)

GEOMETRY_FIELDS = (
    "target_id",
    "L",
    "U",
    "beta",
    "float64_margin",
    "geometry_class",
    "hard_blocker",
    "hard_blocker_id",
)


def split_name(target_id: int) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{target_id}".encode("utf-8")).digest()
    return "development" if digest[0] & 1 == 0 else "validation"


def split_counts(target_ids: list[int]) -> dict[str, int]:
    counts = {"development": 0, "validation": 0}
    for target_id in target_ids:
        counts[split_name(target_id)] += 1
    return counts


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT_BOOTSTRAP,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    return rows / (rows.norm(dim=1, keepdim=True) + 1e-12)


@torch.inference_mode()
def covariance_basis(
    head_native: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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
        raise RuntimeError(f"invalid covariance ridge: {ridge}")
    return mean32, eigenvalues32, eigenvectors32, ridge


@torch.inference_mode()
def directions_for_candidate(
    head64: torch.Tensor,
    target_ids: torch.Tensor,
    spec: dict[str, Any],
    *,
    mean32: torch.Tensor,
    eigenvalues32: torch.Tensor,
    eigenvectors32: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    source = head64[target_ids]
    mean64 = mean32.to(torch.float64)
    if spec["centered"]:
        source = source - mean64
    kind = spec["kind"]
    if kind == "identity":
        transformed = source
    elif kind == "diagonal":
        diagonal = (
            eigenvectors32.square() @ eigenvalues32
        ).to(torch.float64)
        transformed = source * (diagonal + ridge).pow(-float(spec["power"]))
    elif kind == "spectral":
        eigenvectors64 = eigenvectors32.to(torch.float64)
        scale = (eigenvalues32.to(torch.float64) + ridge).pow(
            -float(spec["power"])
        )
        transformed = (source @ eigenvectors64 * scale) @ eigenvectors64.T
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    directions = normalize_rows(transformed)
    if not torch.isfinite(directions).all():
        raise RuntimeError(f"candidate {spec['name']} produced nonfinite directions")
    return directions


@torch.inference_mode()
def geometry_for_directions(
    head64: torch.Tensor,
    hidden_native: torch.Tensor,
    target_ids: torch.Tensor,
    directions: torch.Tensor,
) -> list[dict[str, Any]]:
    if head64.dtype != torch.float64 or directions.dtype != torch.float64:
        raise TypeError("E7c exact geometry requires float64 head and directions")
    if hidden_native.dtype != torch.bfloat16 or target_ids.dtype != torch.long:
        raise TypeError("E7c requires BF16 hidden and long target ids")
    if directions.shape != (target_ids.numel(), head64.shape[1]):
        raise ValueError("candidate direction shape mismatch")
    base = head64 @ hidden_native.to(torch.float64)
    rows: list[dict[str, Any]] = []
    for start in range(0, target_ids.numel(), TARGET_CHUNK):
        end = min(start + TARGET_CHUNK, target_ids.numel())
        tids = target_ids[start:end]
        projected = head64 @ directions[start:end].T
        local_columns = torch.arange(end - start, device=head64.device)
        diagonal = projected[tids, local_columns]
        slopes = diagonal.unsqueeze(0) - projected
        intercepts = base[tids].unsqueeze(0) - base.unsqueeze(1)
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
            raise RuntimeError("E7c lower bound is nonfinite")
        if torch.isnan(upper).any() or torch.isneginf(upper).any():
            raise RuntimeError("E7c upper bound has an invalid nonfinite value")
        for local, column in enumerate(range(start, end)):
            lower_value = float(lower[local])
            upper_value = float(upper[local])
            row: dict[str, Any] = {
                "pool_index": column,
                "target_id": int(target_ids[column]),
                "L": lower_value,
                "U": finite_or_none(upper_value),
                "hard_blocker": bool(hard[local]),
                "hard_blocker_id": None,
                "beta": None,
                "float64_margin": None,
            }
            if row["hard_blocker"]:
                blockers = torch.nonzero(
                    hard_matrix[:, local], as_tuple=False
                ).flatten()
                row["hard_blocker_id"] = int(blockers[0])
                row["geometry_class"] = REFUSE_HARD
                row["decision"] = REFUSE_HARD
            elif not lower_value < upper_value:
                row["geometry_class"] = REFUSE_INTERVAL
                row["decision"] = REFUSE_INTERVAL
            else:
                beta, margin = grid_dose(
                    intercepts[:, local],
                    slopes[:, local],
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


def geometry_summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if split_name(row["target_id"]) == split]
    counts = {GEOMETRIC_REACHABLE: 0, REFUSE_HARD: 0, REFUSE_INTERVAL: 0}
    for row in selected:
        counts[row["geometry_class"]] += 1
    return {
        "target_count": len(selected),
        "geometrically_reachable": counts[GEOMETRIC_REACHABLE],
        "geometric_fraction": counts[GEOMETRIC_REACHABLE] / len(selected),
        "decision_counts": counts,
    }


def geometry_signature(rows: list[dict[str, Any]]) -> str:
    payload = [{field: row.get(field) for field in GEOMETRY_FIELDS} for row in rows]
    return stable_json_sha256(payload)


def assert_raw_reproduction(
    computed: list[dict[str, Any]],
    frozen: list[dict[str, Any]],
) -> None:
    if len(computed) != len(frozen):
        raise RuntimeError("raw geometry row count differs from E7b")
    for index, (actual, expected) in enumerate(zip(computed, frozen, strict=True)):
        for field in GEOMETRY_FIELDS:
            if actual.get(field) != expected.get(field):
                raise RuntimeError(
                    "raw E7b geometry reproduction failed at "
                    f"row {index}, field {field}: "
                    f"{actual.get(field)!r} != {expected.get(field)!r}"
                )


def choose_candidate(candidate_rows: list[dict[str, Any]]) -> str:
    if not candidate_rows:
        raise ValueError("candidate summaries are empty")
    best = max(
        enumerate(candidate_rows),
        key=lambda item: (
            item[1]["development"]["geometrically_reachable"],
            -item[0],
        ),
    )
    return str(best[1]["name"])


def validation_passes(summary: dict[str, Any]) -> bool:
    geometric = int(summary["geometrically_reachable"])
    admitted = int(summary["native_admitted"])
    target_count = int(summary["target_count"])
    return 5 * geometric >= 4 * target_count and 100 * admitted >= 99 * geometric


def native_split_summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if split_name(row["target_id"]) == split]
    summary = summarize_rows(selected)
    summary["ready_bar_passed"] = validation_passes(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if REPORT_PATH.exists():
        raise FileExistsError(f"spent geometry report already exists: {REPORT_PATH}")
    dirty = git_output("status", "--porcelain", "--", *SCREEN_SOURCE_PATHS)
    if dirty:
        raise RuntimeError("E7c screen sources must be committed:\n" + dirty)
    implementation_commit = git_output("rev-parse", "HEAD")
    source_sha256 = {
        relative: sha256_file(REPO_ROOT_BOOTSTRAP / relative)
        for relative in SCREEN_SOURCE_PATHS
    }
    if sha256_file(ADMISSION_PATH) != ADMISSION_SHA256:
        raise RuntimeError("E7b admission hash differs from the frozen spent input")
    if sha256_file(RESULT_PATH) != RESULT_SHA256:
        raise RuntimeError("E7b result hash differs from the frozen spent input")
    if sha256_file(VERDICT_PATH) != VERDICT_SHA256:
        raise RuntimeError("E7b verdict hash differs from the frozen spent input")
    admission, _ = load_admission()
    result, _ = load_result()
    verdict = load_json_object(VERDICT_PATH)
    if verdict.get("verdict") != "FAIL_3B_NATIVE_ADMISSION":
        raise RuntimeError("E7b spent input is not the implementation-valid failure")

    frozen_rows = admission["scientific"]["rows"]
    target_list = [int(row["target_id"]) for row in frozen_rows]
    if split_counts(target_list) != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("E7c spent split count drift")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E7c spent geometry screen requires CUDA")
    torch.manual_seed(0)
    native_runtime = configure_native_runtime()
    started = time.time()
    tokenizer = load_pinned_tokenizer(snapshot_path())
    model = load_pinned_model(snapshot_path(), device)
    invariants = assert_model_invariants(model)
    hidden = final_hidden_batch(model, tokenizer, [PROMPT], device, batch_size=1)[0]
    head_native = model.get_output_embeddings().weight.detach()
    head64 = head_native.to(torch.float64)
    target_ids = torch.tensor(target_list, dtype=torch.long, device=device)

    mean32, eigenvalues32, eigenvectors32, ridge = covariance_basis(head_native)
    basis = {
        "mean_sha256": tensor_sha256(mean32),
        "eigenvalues_sha256": tensor_sha256(eigenvalues32),
        "eigenvectors_sha256": tensor_sha256(eigenvectors32),
        "median_positive_eigenvalue": ridge / RIDGE_FRACTION,
        "ridge_fraction": RIDGE_FRACTION,
        "ridge": ridge,
    }

    raw_spec = CANDIDATE_SPECS[0]
    raw_directions = directions_for_candidate(
        head64,
        target_ids,
        raw_spec,
        mean32=mean32,
        eigenvalues32=eigenvalues32,
        eigenvectors32=eigenvectors32,
        ridge=ridge,
    )
    raw_rows = geometry_for_directions(head64, hidden, target_ids, raw_directions)
    assert_raw_reproduction(raw_rows, frozen_rows)
    raw_geometry_sha256 = geometry_signature(raw_rows)
    del raw_directions, raw_rows

    development_indices = [
        index
        for index, target_id in enumerate(target_list)
        if split_name(target_id) == "development"
    ]
    development_index_tensor = torch.tensor(
        development_indices, dtype=torch.long, device=device
    )
    development_target_ids = target_ids[development_index_tensor]

    candidate_summaries: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        directions = directions_for_candidate(
            head64,
            development_target_ids,
            spec,
            mean32=mean32,
            eigenvalues32=eigenvalues32,
            eigenvectors32=eigenvectors32,
            ridge=ridge,
        )
        rows = geometry_for_directions(
            head64, hidden, development_target_ids, directions
        )
        candidate_summaries.append(
            {
                "name": spec["name"],
                "spec": spec,
                "directions_sha256": tensor_sha256(directions),
                "geometry_sha256": geometry_signature(rows),
                "development": geometry_summary(rows, "development"),
                "validation_evaluated_before_selection": False,
            }
        )
        del directions, rows

    selected_name = choose_candidate(candidate_summaries)
    selected_spec = next(
        spec for spec in CANDIDATE_SPECS if spec["name"] == selected_name
    )
    selected_directions = directions_for_candidate(
        head64,
        target_ids,
        selected_spec,
        mean32=mean32,
        eigenvalues32=eigenvalues32,
        eigenvectors32=eigenvectors32,
        ridge=ridge,
    )
    selected_geometry = geometry_for_directions(
        head64, hidden, target_ids, selected_directions
    )
    selected_rows = native_rows(
        model, hidden, selected_directions, selected_geometry
    )
    for row, frozen in zip(selected_rows, frozen_rows, strict=True):
        row["target_text"] = frozen["target_text"]

    baseline_reachable = {
        row["target_id"]
        for row in frozen_rows
        if row["geometry_class"] == GEOMETRIC_REACHABLE
    }
    selected_reachable = {
        row["target_id"]
        for row in selected_rows
        if row["geometry_class"] == GEOMETRIC_REACHABLE
    }
    development = geometry_summary(selected_geometry, "development")
    validation = native_split_summary(selected_rows, "validation")
    overall = summarize_rows(selected_rows)
    ready = validation_passes(validation)
    scientific = {
        "scope": "post-outcome screen on the already-spent E7b pool",
        "model_revision": MODEL_REVISION,
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "spent_input_sha256": {
            "admission": ADMISSION_SHA256,
            "result": RESULT_SHA256,
            "verdict": VERDICT_SHA256,
        },
        "split_salt": SPLIT_SALT,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "candidate_specs": list(CANDIDATE_SPECS),
        "basis": basis,
        "raw_reproduction_geometry_sha256": raw_geometry_sha256,
        "candidate_development_summaries": candidate_summaries,
        "selected_candidate": selected_name,
        "selected_development": development,
        "selected_validation": validation,
        "selected_overall": overall,
        "gained_reachable_target_ids": sorted(selected_reachable - baseline_reachable),
        "lost_reachable_target_ids": sorted(baseline_reachable - selected_reachable),
        "selected_rows": selected_rows,
        "ready_for_fresh_pool_preregistration": ready,
        "claim_boundary": (
            "Development evidence on the spent E7b pool only; a pass permits "
            "fresh-pool preregistration, not promotion."
        ),
    }
    payload = {
        "schema": "hlm5-e7c-spent-geometry-screen-v1",
        "status": "COMPLETE",
        "implementation_commit": implementation_commit,
        "source_sha256": source_sha256,
        "environment": environment_record(device),
        "native_runtime": native_runtime,
        "model_invariants": invariants,
        "e7b_result_scientific_sha256": result["scientific_sha256"],
        "scientific": scientific,
        "scientific_sha256": scientific_sha256(scientific),
        "runtime_seconds": time.time() - started,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"COMPLETE spent geometry screen: {REPORT_PATH}")
    print(
        f"selected={selected_name} "
        f"development={development['geometrically_reachable']}/"
        f"{development['target_count']} "
        f"validation={validation['geometrically_reachable']}/"
        f"{validation['target_count']} "
        f"native={validation['native_admitted']}/"
        f"{validation['geometrically_reachable']} ready={ready}"
    )


if __name__ == "__main__":
    main()
