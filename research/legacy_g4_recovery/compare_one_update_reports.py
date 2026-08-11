"""Classify the first exact divergence between two one-update probe launches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGES = ("loaded", "sampled", "forward", "backward", "clipped", "stepped")
CLASSIFICATION = {
    "loaded": "LOAD_STATE_DIVERGENCE",
    "sampled": "SAMPLING_OR_RNG_DIVERGENCE",
    "forward": "FORWARD_DIVERGENCE",
    "backward": "BACKWARD_REDUCTION_DIVERGENCE",
    "clipped": "CLIP_DIVERGENCE",
    "stepped": "OPTIMIZER_STEP_DIVERGENCE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-world", type=int, default=96)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(directory: Path, world: int) -> list[dict[str, Any]]:
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if complete.get("rank_reports") != world:
        raise RuntimeError(
            f"completion record mismatch at {directory}: {complete.get('rank_reports')}")
    reports = []
    for rank in range(world):
        path = directory / f"rank{rank:03d}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": "hlm5-g4-one-update-determinism-rank-v1",
            "rank": rank,
            "world_size": world,
            "source_step": 522000,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise RuntimeError(
                    f"report field mismatch path={path} {key}={report.get(key)!r}")
        if set(report.get("stages", {})) != set(STAGES):
            raise RuntimeError(f"stage coverage mismatch at {path}")
        reports.append(report)
    return reports


def manifest_sha256(directory: Path, world: int) -> str:
    digest = hashlib.sha256()
    for rank in range(world):
        path = directory / f"rank{rank:03d}.json"
        digest.update(f"rank{rank:03d}:{sha256(path)}\n".encode("ascii"))
    digest.update(f"complete:{sha256(directory / 'complete.json')}\n".encode("ascii"))
    return digest.hexdigest()


def differing_fields(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(differing_fields(left[key], right[key], path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                differing_fields(left_item, right_item, f"{prefix}[{index}]")
            )
        return differences
    return [] if type(left) is type(right) and left == right else [prefix]


def classify(
    reports_a: list[dict[str, Any]], reports_b: list[dict[str, Any]]
) -> dict[str, Any]:
    topology_mismatches = []
    per_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    first_stage_by_rank: dict[str, str | None] = {}

    for left, right in zip(reports_a, reports_b):
        rank = left["rank"]
        topology_fields = (
            "rank",
            "world_size",
            "local_rank",
            "hostname",
            "device_name",
            "torch",
            "cuda",
            "cudnn",
            "source_step",
            "scheduled_lr",
        )
        topology_diff = [
            key for key in topology_fields if type(left.get(key)) is not type(right.get(key))
            or left.get(key) != right.get(key)
        ]
        if topology_diff:
            topology_mismatches.append({"rank": rank, "fields": topology_diff})

        first_stage = None
        for stage in STAGES:
            differences = differing_fields(left["stages"][stage], right["stages"][stage])
            if differences:
                per_stage[stage].append(
                    {"rank": rank, "fields": differences[:32], "field_count": len(differences)}
                )
                if first_stage is None:
                    first_stage = stage
        first_stage_by_rank[str(rank)] = first_stage

    first_divergent_stage = next(
        (stage for stage in STAGES if per_stage[stage]),
        None,
    )
    if topology_mismatches:
        status = "IMPLEMENTATION_INVALID"
        classification = "TOPOLOGY_MISMATCH"
    elif first_divergent_stage is None:
        status = "VALID_DIAGNOSIS"
        classification = "ONE_UPDATE_EXACT"
    else:
        status = "VALID_DIAGNOSIS"
        classification = CLASSIFICATION[first_divergent_stage]

    return {
        "status": status,
        "classification": classification,
        "first_divergent_stage": first_divergent_stage,
        "topology_mismatches": topology_mismatches,
        "divergent_rank_count_by_stage": {
            stage: len(per_stage[stage]) for stage in STAGES
        },
        "first_stage_by_rank": first_stage_by_rank,
        "difference_samples_by_stage": {
            stage: per_stage[stage][:32] for stage in STAGES if per_stage[stage]
        },
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    reports_a = load_reports(args.run_a, args.expected_world)
    reports_b = load_reports(args.run_b, args.expected_world)
    diagnosis = classify(reports_a, reports_b)
    result = {
        "schema": "hlm5-g4-one-update-determinism-result-v1",
        "world_size": args.expected_world,
        "source_step": 522000,
        "run_a_manifest_sha256": manifest_sha256(args.run_a, args.expected_world),
        "run_b_manifest_sha256": manifest_sha256(args.run_b, args.expected_world),
        **diagnosis,
    }
    write_new_json(args.result, result)
    print(
        f"ONE_UPDATE_DIAGNOSIS status={result['status']} "
        f"classification={result['classification']} "
        f"first_stage={result['first_divergent_stage']}",
        flush=True,
    )
    return 0 if result["status"] == "VALID_DIAGNOSIS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
