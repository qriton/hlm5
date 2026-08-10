"""Fail-closed provenance contract for HLM5 E11 cross-node portability."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .e7_contract import (
    MODEL_FILE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    REPO_ROOT,
    require_file_hash,
    sha256_file,
    snapshot_path,
    stable_json_sha256,
    verify_snapshot,
)
from .e7b_contract import EXPECTED_NATIVE_RUNTIME
from .e8_contract import EXPECTED_ENVIRONMENT, expected_model_invariants
from .e10_contract import selected_population
from .e11_runtime import E10_NODE


PROTOCOL_COMMIT = "8543e5a36c7220836e2ac75ed279b0694b598ae6"
PROTOCOL_PATH = "docs/e11-3b-cross-node-portability-prereg-2026-08-11.md"
PROTOCOL_SHA256 = "6a5bfd08ac942fd98f0a5bdaabf83bb8c3bdec48d241ffe33a7801e5f4173da1"

E10_EVIDENCE_COMMIT = "5e37272ecfcfbc9edb40ebe52d9f19c41b83a105"
E10_DIR = "results/e10_3b_evidence"
E10_PREFLIGHT_SHA256 = (
    "cf4820e4049b94e0abdd009ce361ce5081ebf41ca0ae49bb28e946e5b0247a0f"
)
E10_BUNDLE_SHA256 = "d8b437041c5f902691c1f9e801ef57cdc5bedb1ea7d07a22b77d41aa9c07a3c7"
E10_ADMISSION_SHA256 = (
    "2c8b3a763a1209eb2098b679fa1fcaae9b5361d6c09da53b80dd6ac82c7dea86"
)
E10_RESULT_SHA256 = "824a118add90b163114372e22f55d31d7503a3f90ae4579e9eccf7cf16adaf3e"
E10_VERDICT_SHA256 = "9e3aa70a01a8f43e39836fa93479beea17b6d1e8535c96c113c1209f6b48ce56"
E10_TIER_SHA256 = "25f698e5b4569dd5c9556906bdeb7043ee32df2d57ae070b19efecc336c621fd"
E10_BUNDLE_MANIFEST_SHA256 = (
    "6cd6abb5ebb275b1a6cd4af46f719c63f58814f1a2ee441667506277d47bbceb"
)
E10_SELECTED_SHA256 = "b5ae24760e283e63e34459affb9173617311380ab02433dbb08feebe6abdf2de"
E10_SELECTED_CASE_SHA256 = (
    "e73c67a8a95f15e71844cf8450f924bea9452627030283013f85d633baafad09"
)

SEED = 0
PREFLIGHT_SCHEMA = "hlm5-e11-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e11-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e11-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e11-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e11-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e11-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57",
    "hlm5/e7_contract.py": "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d",
    "hlm5/e7_runtime.py": "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc",
    "hlm5/e7b_contract.py": "606d95ca4972d1febadf6348c441449574f1c33eb2f3b7b71b3f79df27c31119",
    "hlm5/e7b_runtime.py": "bf6e7b0abfe83d0da0f146ae72dc69d52b0469fcc311689c535a7628ea2fdbc2",
    "hlm5/e8_contract.py": "633106728f658ce469c22e7f3d51fc013b38b90af24f5a506c6deb721166fd80",
    "hlm5/e9_contract.py": "bf345b9e09ee60672d94a582f2dcea60343a5889bd6410f60af81023cffc5fc7",
    "hlm5/e9_runtime.py": "7c805950e3eb0618953c4e24d64629c5b427e774a2395d525384a6425643b3f7",
    "hlm5/e10_contract.py": "4d893d9cdc07d3556db7ab25d367b27580ea065ddaba2b2c6e88f365c5447dca",
    "hlm5/e10_runtime.py": "09dbb2a71426f21b9c935acb22d15e71f4c80767b59e43cc472f4639516015ef",
    "hlm5/memory.py": "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973",
    "hlm5/public_adapter.py": "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d",
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/e11_contract.py",
    "hlm5/e11_runtime.py",
    "scripts/preflight_e11_3b.py",
    "scripts/register_e11_3b.py",
    "scripts/admit_e11_3b.py",
    "scripts/replay_e11_3b.py",
    "scripts/verify_e11_3b.py",
    "tests/test_e11_3b.py",
)
REGISTERED_COMMANDS = tuple(
    f"python scripts/{name}_e11_3b.py"
    for name in ("preflight", "register", "admit", "replay", "verify")
)

STAGING_DIR = REPO_ROOT / "results/e11_3b_staging"
PREFLIGHT_PATH = STAGING_DIR / "preflight.json"
EXECUTION_RECEIPT_PATH = STAGING_DIR / "execution_receipt.json"
ATTEMPT_PATH = STAGING_DIR / "attempt.json"
ADMISSION_PATH = STAGING_DIR / "admission.json"
RESULT_PATH = STAGING_DIR / "result.json"
VERDICT_PATH = STAGING_DIR / "verdict.json"
NAMED_OUTPUTS = (
    PREFLIGHT_PATH,
    EXECUTION_RECEIPT_PATH,
    ATTEMPT_PATH,
    ADMISSION_PATH,
    RESULT_PATH,
    VERDICT_PATH,
)
BUNDLE_PATH = REPO_ROOT / E10_DIR / "adapter_1024.pt"


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_create_json(path: Path, value: Any) -> None:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"E11 output must stay under {staging}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{resolved.name}.", dir=resolved.parent
    )
    temp = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, resolved)
        except FileExistsError as error:
            raise FileExistsError(f"E11 output already exists: {resolved}") from error
    finally:
        temp.unlink(missing_ok=True)


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E11 outputs already exist: {present}")


def scientific_sha256(value: dict[str, Any]) -> str:
    return stable_json_sha256(value)


def verify_protocol() -> None:
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("registered E11 protocol unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E11 protocol")


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E11 dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def frozen_e10() -> dict[str, Any]:
    expected = {
        "preflight.json": E10_PREFLIGHT_SHA256,
        "adapter_1024.pt": E10_BUNDLE_SHA256,
        "admission.json": E10_ADMISSION_SHA256,
        "result.json": E10_RESULT_SHA256,
        "verdict.json": E10_VERDICT_SHA256,
    }
    for name, digest in expected.items():
        require_file_hash(REPO_ROOT / E10_DIR / name, digest, f"E10 {name}")
    verdict = load_json_object(REPO_ROOT / E10_DIR / "verdict.json")
    admission = load_json_object(REPO_ROOT / E10_DIR / "admission.json")
    if verdict.get("verdict") != "PASS_3B_CAPACITY_1024":
        raise RuntimeError("bound E10 verdict is not PASS_3B_CAPACITY_1024")
    measurement = admission["scientific"]["measurement"]
    tier = measurement["tiers"]["1024"]
    bundle = measurement["bundle"]
    selected = measurement["direct"]["direct_summary"]["selected_indices"]
    selected_cases = measurement["direct"]["direct_summary"]["selected_case_ids"]
    checks = {
        "tier": (tier, E10_TIER_SHA256),
        "bundle": (bundle, E10_BUNDLE_MANIFEST_SHA256),
        "selected": (selected, E10_SELECTED_SHA256),
        "selected_cases": (selected_cases, E10_SELECTED_CASE_SHA256),
    }
    for name, (value, digest) in checks.items():
        if stable_json_sha256(value) != digest:
            raise RuntimeError(f"bound E10 {name} object changed")
    return {
        "tier": tier,
        "bundle": bundle,
        "selected_indices": selected,
        "selected_case_ids": selected_cases,
        "raw_sha256": expected,
    }


def registered_source_sha256(*, require_clean: bool = False) -> dict[str, str]:
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E11 sources must be committed:\n" + dirty)
    missing = [p for p in REGISTERED_SOURCE_PATHS if not (REPO_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"registered E11 sources missing: {missing}")
    return {path: sha256_file(REPO_ROOT / path) for path in REGISTERED_SOURCE_PATHS}


def environment_failures(value: Any, *, expected_node: str | None = None) -> list[str]:
    if not isinstance(value, dict):
        return ["E11 environment missing"]
    failures = [
        f"E11 environment mismatch: {name}"
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if value.get(name) != expected
    ]
    node = value.get("node")
    if not isinstance(node, str) or not node:
        failures.append("E11 physical node missing")
    elif node == E10_NODE:
        failures.append("E11 physical node equals E10 node")
    elif expected_node is not None and node != expected_node:
        failures.append("E11 physical node differs from preflight")
    if set(value) != set(EXPECTED_ENVIRONMENT) | {"node"}:
        failures.append("E11 environment fields are not exhaustive")
    return failures


def structural_failures(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return ["E11 measurement missing"]
    failures: list[str] = []
    if set(measurement) != {
        "memory_receipt",
        "positive_scan",
        "locality_scan",
        "rollback",
    }:
        failures.append("E11 measurement fields differ from E10 tier")
    rows = measurement.get("positive_scan", {}).get("rows")
    locality = measurement.get("locality_scan", {}).get("rows")
    if not isinstance(rows, list) or len(rows) != 1024:
        failures.append("E11 positive denominator mismatch")
    if not isinstance(locality, list) or len(locality) != 12_288:
        failures.append("E11 locality denominator mismatch")
    memory = measurement.get("memory_receipt", {})
    if memory.get("active_count") != 1024 or memory.get("active_slots") != list(
        range(1024)
    ):
        failures.append("E11 active prefix mismatch")
    rollback = measurement.get("rollback", {})
    if rollback.get("post_active_count") != 0:
        failures.append("E11 rollback did not clear memory")
    return failures


def _source_receipt(payload: dict[str, Any]) -> None:
    if payload.get("source_sha256") != registered_source_sha256():
        raise RuntimeError("registered E11 source drift")
    if payload.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E11 dependency drift")


def load_preflight() -> tuple[dict[str, Any], str]:
    payload = load_json_object(PREFLIGHT_PATH)
    evidence = frozen_e10()
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "e10_evidence_commit": E10_EVIDENCE_COMMIT,
        "e10_raw_sha256": evidence["raw_sha256"],
        "e10_bundle_manifest": evidence["bundle"],
        "commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e11_3b.py -q",
        "tests_returncode": 0,
        "e11_outcomes_computed": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E11 preflight mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E11 model invariant mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E11 arithmetic mismatch")
    failures = environment_failures(payload.get("environment"))
    if failures:
        raise RuntimeError("invalid E11 environment: " + "; ".join(failures))
    commit = payload.get("implementation_commit")
    if not isinstance(commit, str) or git_output("rev-parse", commit) != commit:
        raise RuntimeError("E11 implementation commit unavailable")
    _source_receipt(payload)
    verify_protocol()
    verify_dependencies()
    verify_snapshot(snapshot_path())
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    payload = load_json_object(EXECUTION_RECEIPT_PATH)
    expected = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "preflight_sha256": preflight_sha,
        "preflight_file_sha256_recomputed": preflight_sha,
        "source_sha256": preflight["source_sha256"],
        "dependency_sha256": DEPENDENCY_SHA256,
        "e10_raw_sha256": preflight["e10_raw_sha256"],
        "e10_bundle_manifest": preflight["e10_bundle_manifest"],
        "environment": preflight["environment"],
        "native_runtime": preflight["native_runtime"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e11_3b.py -q",
        "tests_returncode": 0,
        "e11_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E11 execution receipt mismatch: {name}")
    _source_receipt(payload)
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, receipt_sha = load_execution_receipt()
    payload = load_json_object(ATTEMPT_PATH)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "status": "PORTABILITY_ATTEMPT_SPENT",
        "execution_receipt_sha256": receipt_sha,
        "implementation_commit": receipt["implementation_commit"],
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "bundle_sha256": E10_BUNDLE_SHA256,
        "pre_admission_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E11 attempt mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def verify_scientific(payload: dict[str, Any], label: str) -> None:
    value = payload.get("scientific")
    if not isinstance(value, dict) or payload.get(
        "scientific_sha256"
    ) != scientific_sha256(value):
        raise RuntimeError(f"{label} scientific hash mismatch")


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, receipt_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    payload = load_json_object(ADMISSION_PATH)
    expected = {
        "schema": ADMISSION_SCHEMA,
        "status": "COMPLETE",
        "attempt_sha256": attempt_sha,
        "execution_receipt_sha256": receipt_sha,
        "implementation_commit": receipt["implementation_commit"],
        "native_runtime": receipt["native_runtime"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E11 admission provenance mismatch: {name}")
    verify_scientific(payload, "E11 admission")
    if payload["scientific"].get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E11 admission model invariant mismatch")
    failures = structural_failures(payload["scientific"].get("measurement"))
    if failures:
        raise RuntimeError("invalid E11 admission: " + "; ".join(failures))
    return payload, sha256_file(ADMISSION_PATH)


def result_validity_failures(
    result: dict[str, Any], admission: dict[str, Any]
) -> list[str]:
    try:
        verify_scientific(result, "E11 result")
    except Exception as error:
        return [str(error)]
    science = result["scientific"]
    failures = []
    if science.get("admission_scientific_sha256") != admission["scientific_sha256"]:
        failures.append("E11 replay binds a different admission")
    if science.get("measurement") != admission["scientific"].get("measurement"):
        failures.append("E11 replay differs from admission")
    if science.get("measurement_exact_match") is not True:
        failures.append("E11 replay exact-match flag failed")
    return failures


def load_result() -> tuple[dict[str, Any], str]:
    receipt, _receipt_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    payload = load_json_object(RESULT_PATH)
    expected = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE",
        "attempt_sha256": attempt_sha,
        "admission_sha256": admission_sha,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "native_runtime": receipt["native_runtime"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E11 result provenance mismatch: {name}")
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E11 result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


def scientific_failures(result: dict[str, Any]) -> list[str]:
    measurement = result["scientific"]["measurement"]
    expected = frozen_e10()["tier"]
    return (
        []
        if measurement == expected
        else ["cross-node E11 tier differs from frozen E10 tier"]
    )


def verdict_scientific_payload(
    *,
    verdict: str,
    result: dict[str, Any] | None,
    validity_failures: list[str],
    scientific_failures_value: list[str],
) -> dict[str, Any]:
    measurement = (
        None if result is None else result.get("scientific", {}).get("measurement")
    )
    expected = frozen_e10()["tier"]
    return {
        "verdict": verdict,
        "cross_node_exact_match": measurement == expected
        if measurement is not None
        else None,
        "positive_summary": None
        if measurement is None
        else measurement["positive_scan"]["summary"],
        "locality_summary": None
        if measurement is None
        else measurement["locality_scan"]["summary"],
        "rollback_summary": None
        if measurement is None
        else measurement["rollback"]["summary"],
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Exact portability of the frozen E10 1024-slot adapter between two "
            "registered Leonardo A100 nodes under one pinned software/runtime stack."
        ),
    }


__all__ = [
    "ADMISSION_PATH",
    "ADMISSION_SCHEMA",
    "ATTEMPT_PATH",
    "ATTEMPT_SCHEMA",
    "BUNDLE_PATH",
    "DEPENDENCY_SHA256",
    "EXECUTION_RECEIPT_PATH",
    "EXECUTION_SCHEMA",
    "NAMED_OUTPUTS",
    "PREFLIGHT_PATH",
    "PREFLIGHT_SCHEMA",
    "PROTOCOL_COMMIT",
    "PROTOCOL_SHA256",
    "REGISTERED_COMMANDS",
    "RESULT_PATH",
    "RESULT_SCHEMA",
    "SEED",
    "VERDICT_PATH",
    "VERDICT_SCHEMA",
    "assert_outputs_absent",
    "atomic_create_json",
    "environment_failures",
    "frozen_e10",
    "git_output",
    "load_admission",
    "load_attempt",
    "load_execution_receipt",
    "load_json_object",
    "load_preflight",
    "load_result",
    "registered_source_sha256",
    "result_validity_failures",
    "scientific_failures",
    "scientific_sha256",
    "selected_population",
    "structural_failures",
    "verdict_scientific_payload",
    "verify_dependencies",
    "verify_protocol",
]
