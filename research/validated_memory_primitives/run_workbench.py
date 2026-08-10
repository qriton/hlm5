"""Replay and summarize the two validated HLM5 memory primitives.

This is the self-contained demo front door for two bounded research results:

1. exact deterministic-address lifecycle operations; and
2. support-masked multi-support retrieval on a synthetic orthogonal task.

Both HLM5 readers are required to remain bit-identical to matched caches.  The
workbench deliberately does not claim semantic routing, language-model gain,
an HLM-specific retrieval advantage, or authorization to scale.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "snapshot_manifest.json"
OUTPUT = ROOT / "outputs" / "validated_memory_primitives.json"
EXPECTED_MANIFEST_SHA256 = (
    "a47653ed3aac7c8b8760bedbcf00d9a32339c37e6be4e4d5998fcb0545b46110"
)
EXPECTED_EXACT_STATUS = "CACHE_EQUIVALENT_REFERENCE"
EXPECTED_EXACT_SCIENTIFIC_SHA256 = (
    "a69593454521f693c21ba7195d33e2e59dc0aafc8242cce0e886aba2554de907"
)
EXPECTED_MULTI_STATUS = "NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT"
EXPECTED_MULTI_SCIENTIFIC_SHA256 = (
    "07a410e232745067feaee9862804e6c173c3a50b0d00de9aa079a2044a1829c6"
)


class WorkbenchError(RuntimeError):
    """Raised when frozen provenance or a replay contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _snapshot_path(relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise WorkbenchError(f"unsafe snapshot path: {relative}")
    path = ROOT / relative_path
    if path.is_symlink() or not path.is_file():
        raise WorkbenchError(f"snapshot artifact is missing or linked: {relative}")
    try:
        path.resolve().relative_to(ROOT)
    except ValueError as exc:
        raise WorkbenchError(f"snapshot artifact escapes package: {relative}") from exc
    return path


def verify_snapshot() -> dict[str, Any]:
    if MANIFEST.is_symlink() or not MANIFEST.is_file():
        raise WorkbenchError("snapshot manifest is missing or linked")
    actual_manifest_hash = sha256_file(MANIFEST)
    if actual_manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise WorkbenchError(
            "snapshot manifest changed: "
            f"{actual_manifest_hash} != {EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema") != "hlm5-validated-memory-primitives-snapshot-v1":
        raise WorkbenchError("unexpected snapshot manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise WorkbenchError("snapshot manifest has no pinned files")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise WorkbenchError("snapshot manifest has a malformed file pin")
        actual = sha256_file(_snapshot_path(relative))
        if actual != expected:
            raise WorkbenchError(
                f"snapshot artifact changed: {relative}: {actual} != {expected}"
            )
    return manifest


def _load_json(relative: str) -> dict[str, Any]:
    value = json.loads(_snapshot_path(relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkbenchError(f"expected JSON object: {relative}")
    return value


def verify_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    exact = _load_json("baselines/results/exact_address_lifecycle_reference.json")
    multi = _load_json(
        "baselines/results/multi_support_kernel_equivalence_reference.json"
    )

    if exact.get("status") != EXPECTED_EXACT_STATUS:
        raise WorkbenchError("exact-address result status changed")
    if exact.get("scientific_sha256") != EXPECTED_EXACT_SCIENTIFIC_SHA256:
        raise WorkbenchError("exact-address scientific digest changed")
    if not exact.get("all_phases_degree5_cache_bit_exact"):
        raise WorkbenchError("exact-address reader is not cache-equivalent")
    if not all(exact.get("lifecycle", {}).values()):
        raise WorkbenchError("exact-address lifecycle is incomplete")

    if multi.get("status") != EXPECTED_MULTI_STATUS:
        raise WorkbenchError("multi-support result status changed")
    if multi.get("scientific_sha256") != EXPECTED_MULTI_SCIENTIFIC_SHA256:
        raise WorkbenchError("multi-support scientific digest changed")
    aggregate = multi.get("aggregate", {})
    if not aggregate.get("all_hlm_matched_degree5_fields_bit_exact"):
        raise WorkbenchError("multi-support reader is not cache-equivalent")
    boundary = multi.get("claim_boundary", {})
    required_boundary = {
        "multi_support_function": True,
        "hlm_specific_retrieval_advantage": False,
        "semantic_routing": False,
        "language_model_gain": False,
        "training_or_scaling_authorized": False,
    }
    if boundary != required_boundary:
        raise WorkbenchError("multi-support claim boundary changed")
    return exact, multi


def _load_module(name: str, relative: str) -> ModuleType:
    path = _snapshot_path(relative)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise WorkbenchError(f"could not load reference module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay_references() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (ROOT, ROOT / "baselines", ROOT / "demos"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    exact_module = _load_module(
        "hlm5_demo_exact_address_reference",
        "demos/exact_address_lifecycle_reference.py",
    )
    multi_module = _load_module(
        "hlm5_demo_multi_support_reference",
        "demos/multi_support_kernel_equivalence_reference.py",
    )
    exact = exact_module.run_reference()
    multi = multi_module.run_reference()
    return exact, multi


def _require_replay_matches(
    frozen: dict[str, Any], replayed: dict[str, Any], label: str
) -> None:
    if replayed != frozen:
        raise WorkbenchError(
            f"{label} replay differs from the frozen registered result"
        )


def build_report(
    manifest: dict[str, Any],
    exact: dict[str, Any],
    multi: dict[str, Any],
) -> dict[str, Any]:
    create = multi["aggregate"]["phases"]["create"]
    report: dict[str, Any] = {
        "schema": "hlm5-validated-memory-primitives-workbench-v1",
        "status": "VALIDATED_CACHE_EQUIVALENT_PRIMITIVES",
        "scope": "Two deterministic CPU references; no checkpoint or LM forward.",
        "provenance": {
            "snapshot_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "snapshot_date": manifest["snapshot_date"],
            "source_repository": manifest["source_repository"],
            "registered_runtime": manifest["registered_runtime"],
            "exact_scientific_sha256": exact["scientific_sha256"],
            "multi_support_scientific_sha256": multi["scientific_sha256"],
        },
        "working": {
            "exact_address_lifecycle": {
                "status": exact["status"],
                "operations": exact["lifecycle"],
                "matched_degree5_cache_bit_exact": exact[
                    "all_phases_degree5_cache_bit_exact"
                ],
            },
            "bounded_multi_support": {
                "status": multi["status"],
                "hlm5_degree5_accuracy": create["hlm5_degree5"]["accuracy"],
                "matched_degree5_cache_accuracy": create[
                    "matched_degree5_cache"
                ]["accuracy"],
                "cosine_soft_cache_accuracy": create["cosine_soft_cache"][
                    "accuracy"
                ],
                "hard_top1_accuracy": create["hard_top1_cache"]["accuracy"],
                "shuffled_value_null_accuracy": multi["aggregate"]["phases"][
                    "shuffled_value_null"
                ]["accuracy"],
                "one_share_deactivated_accuracy": multi["aggregate"]["phases"][
                    "deactivate_one_share"
                ]["hlm5_degree5"]["accuracy"],
                "off_support_outputs_bit_zero": multi["aggregate"]["phases"][
                    "off_support"
                ]["all_reader_outputs_bit_zero"],
                "matched_degree5_cache_bit_exact": multi["aggregate"][
                    "all_hlm_matched_degree5_fields_bit_exact"
                ],
            },
        },
        "use_now": [
            "Governed exact-address create/edit/deactivate/reactivate/delete lifecycle.",
            "Explicit support-gated nonlinear multi-support composition where the task fits the measured boundary.",
            "Matched-cache implementation as the simpler production reference unless another HLM-specific property is measured.",
        ],
        "do_not_claim": [
            "HLM-specific retrieval advantage",
            "semantic or paraphrase routing",
            "language-model quality gain",
            "robustness across the full mixture range",
            "training or scaling authorization",
        ],
    }
    report["scientific_sha256"] = scientific_sha256(report)
    return report


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise WorkbenchError("refusing to write workbench output through a link")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def render_summary(report: dict[str, Any]) -> str:
    exact = report["working"]["exact_address_lifecycle"]
    multi = report["working"]["bounded_multi_support"]
    return "\n".join(
        [
            "HLM5 validated memory primitives",
            "================================",
            (
                "[VALID] Exact-address lifecycle: create, edit, deactivate, "
                "serialize/restore, reactivate, delete, and off-support gate."
            ),
            f"        Degree-5 reader == matched cache bit-for-bit: {exact['matched_degree5_cache_bit_exact']}",
            (
                "[VALID, NARROW] Multi-support composition: "
                f"degree-5={multi['hlm5_degree5_accuracy']:.0%}, "
                f"cosine-soft={multi['cosine_soft_cache_accuracy']:.0%}, "
                f"hard-top1={multi['hard_top1_accuracy']:.0%}."
            ),
            f"        Degree-5 reader == matched degree-5 cache bit-for-bit: {multi['matched_degree5_cache_bit_exact']}",
            "",
            "USE NOW: governed lifecycle; bounded support-gated composition; matched cache as the production reference.",
            "DO NOT CLAIM: HLM-specific advantage, semantic routing, LM gain, robustness, or scaling authorization.",
            f"Evidence digest: {report['scientific_sha256']}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify frozen hashes and evidence without recomputing the references",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="do not update the generated combined report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = verify_snapshot()
    frozen_exact, frozen_multi = verify_frozen_evidence()
    if args.verify_only:
        print(
            "Snapshot and frozen evidence verified: "
            f"{len(manifest['files'])} artifacts, exact + multi-support."
        )
        return 0

    replayed_exact, replayed_multi = replay_references()
    _require_replay_matches(frozen_exact, replayed_exact, "exact-address")
    _require_replay_matches(frozen_multi, replayed_multi, "multi-support")
    report = build_report(manifest, replayed_exact, replayed_multi)
    if not args.no_write:
        atomic_write_json(OUTPUT, report)
    print(render_summary(report))
    if not args.no_write:
        print(f"Combined report: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
