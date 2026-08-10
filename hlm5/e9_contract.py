"""Fail-closed provenance and verdict contract for the E9 locality scan."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .e7_contract import (
    MODEL_FILE_SHA256,
    MODEL_HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    REPO_ROOT,
    RESULTS_DIR,
    git_output,
    require_file_hash,
    sha256_file,
    snapshot_path,
    stable_json_sha256,
    verify_snapshot,
)
from .e7b_contract import EXPECTED_NATIVE_RUNTIME
from .e8_contract import (
    COUNTERFACT_PATH,
    COUNTERFACT_SHA256,
    EXPECTED_ENVIRONMENT,
    expected_model_invariants,
)
from .e8_runtime import GATE_THRESHOLD
from .e9_runtime import (
    GATE_BATCH,
    KINDS,
    MODEL_FORWARD_BATCH,
    QUERY_CASE_COUNT,
    QUERY_CASE_START,
    QUERY_CASE_STOP,
    QUERY_COUNT,
    load_query_rows,
    query_prompt_order,
    query_specs,
    summarize_query_scan,
)


PROTOCOL_COMMIT = "b16c93199fac5d47c4f29cc64cac147517ee62d1"
PROTOCOL_PATH = "docs/e9-wide-locality-prereg-2026-08-11.md"
PROTOCOL_SHA256 = "acb1ebe21c8eda33d4c2fa6b3b2b80e4643af69048e638afe9de98b0a8a185b3"

E8_ADMISSION_PATH = "results/e8_3b_evidence/admission.json"
E8_ADMISSION_SHA256 = "96d6dd8f8528d38de52d559df86f4228b508de4487cf9ea3efd50b63adccc7b0"
E8_RESULT_PATH = "results/e8_3b_evidence/result.json"
E8_RESULT_SHA256 = "ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6"
E8_VERDICT_PATH = "results/e8_3b_evidence/verdict.json"
E8_VERDICT_SHA256 = "5e376c4c065ecf7e7602d3e81984e6425175876bc763ffd2a427684a6b9d7deb"
E8_IMPLEMENTATION_COMMIT = "3ee23ba8a1451f3231254fe7f7f63b4e3422468d"
E8_DIRECT_SHA256 = "417eb04f694821d23da067209f9f29f1e0443d0ec96fcf3d5fd4572151c272c3"
E8_MEMORY_SHA256 = "cddc691e44f87a2fceec5153763211b4b112adda0ee8ac1f347c748206b612c1"
E8_KEY_DRIFT_FIELDS = frozenset(
    {"active_keys_sha256", "key_mean_sha256", "key_transform_sha256"}
)

QUERY_ROW_SHA256 = "1902f1aa83e7214121e9d12db75c499b659d0077706b8373020748111643c7fa"
QUERY_CASE_ID_SHA256 = (
    "c8a495c828e3fa4da443f7021a28cf587a978b51441326f62b3ec9fd5fc1d579"
)
QUERY_PROMPT_SHA256 = "7f9c199bb1a402fa95bd46ffa879d01a7e36acb3f9921eb2ac2043b3343ff07a"
QUERY_KIND_SHA256 = {
    "exact": "ae8350d284f8bbf1db507bb6c9c939504e0a2c74bb14a44ebf0b6bdc44c5358c",
    "paraphrase": "d55b79a22f0c70e0d903cbded95e6ad688a16a4c16eec49f3dbe101be6b0256a",
    "neighborhood": "0b65a870be5f2b67bccabd0f4c1c299171a265d3b873ad3d8d29bc8e9827fdb8",
}
SEED = 0

PREFLIGHT_SCHEMA = "hlm5-e9-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e9-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e9-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e9-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e9-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e9-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57",
    "hlm5/memory.py": "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973",
    "hlm5/model.py": "2f810bde52bbf3e0da9f52640bf36ca924cd665bd6a7036155f0581494b8857c",
    "hlm5/io.py": "5886ce8331ba42ca0ee75dd67746665be58d174e5f3416ad661f81c7d7685439",
    "hlm5/certify.py": "a13544eb242cb1f827496a46a6d943723ad51f35b827fe47d2da641e26a1c210",
    "hlm5/edit_audit.py": "930d51619bb240e13a15412f6db92b0cf58a87e17a405dc7110bbaaa2e8e8930",
    "hlm5/key_value.py": "a0bacdfcba80397bfacefa7c7762bcb202fb91363da0e6fa902f9c707c70140a",
    "hlm5/public_adapter.py": "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d",
    "hlm5/e7_contract.py": "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d",
    "hlm5/e7_runtime.py": "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc",
    "hlm5/e7b_contract.py": "606d95ca4972d1febadf6348c441449574f1c33eb2f3b7b71b3f79df27c31119",
    "hlm5/e7b_runtime.py": "bf6e7b0abfe83d0da0f146ae72dc69d52b0469fcc311689c535a7628ea2fdbc2",
    "hlm5/e7c_runtime.py": "55b7daf0d1d65303db5a3331ddc9b3db701e512ea45f2823a2dbfece35a58d23",
    "hlm5/e8_contract.py": "633106728f658ce469c22e7f3d51fc013b38b90af24f5a506c6deb721166fd80",
    "hlm5/e8_runtime.py": "dc78a1d7bfea929a66b01d69cc0fdf9b7f7634643d918c6d182c8854fb205be4",
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/e9_contract.py",
    "hlm5/e9_runtime.py",
    "scripts/preflight_e9_3b.py",
    "scripts/register_e9_3b.py",
    "scripts/admit_e9_3b.py",
    "scripts/replay_e9_3b.py",
    "scripts/verify_e9_3b.py",
    "tests/test_e9_3b.py",
)
REGISTERED_COMMANDS = (
    "python scripts/preflight_e9_3b.py",
    "python scripts/register_e9_3b.py",
    "python scripts/admit_e9_3b.py",
    "python scripts/replay_e9_3b.py",
    "python scripts/verify_e9_3b.py",
)

STAGING_DIR = RESULTS_DIR / "e9_3b_staging"
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


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_staging_path(path: Path) -> Path:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"E9 output must stay under {staging}: {resolved}")
    return resolved


def atomic_create_json(path: Path, value: Any) -> None:
    resolved = _require_staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, resolved)
        except FileExistsError as error:
            raise FileExistsError(f"E9 output already exists: {resolved}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink()


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E9 outputs already exist: {present}")


def scientific_sha256(scientific: dict[str, Any]) -> str:
    return stable_json_sha256(scientific)


def verify_scientific_hash(payload: dict[str, Any], label: str) -> None:
    scientific = payload.get("scientific")
    if not isinstance(scientific, dict):
        raise RuntimeError(f"{label} has no scientific object")
    if payload.get("scientific_sha256") != scientific_sha256(scientific):
        raise RuntimeError(f"{label} scientific hash mismatch")


def verify_protocol() -> None:
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("registered E9 protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E9 protocol")


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E9 dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def frozen_e8_measurement() -> dict[str, Any]:
    return load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"]["measurement"]


def verify_bound_inputs() -> dict[str, Any]:
    require_file_hash(
        REPO_ROOT / E8_ADMISSION_PATH, E8_ADMISSION_SHA256, "E8 admission"
    )
    require_file_hash(REPO_ROOT / E8_RESULT_PATH, E8_RESULT_SHA256, "E8 result")
    require_file_hash(REPO_ROOT / E8_VERDICT_PATH, E8_VERDICT_SHA256, "E8 verdict")
    require_file_hash(
        REPO_ROOT / COUNTERFACT_PATH, COUNTERFACT_SHA256, "CounterFact data"
    )
    admission = load_json_object(REPO_ROOT / E8_ADMISSION_PATH)
    result = load_json_object(REPO_ROOT / E8_RESULT_PATH)
    verdict = load_json_object(REPO_ROOT / E8_VERDICT_PATH)
    if admission.get("status") != "COMPLETE" or result.get("status") != "COMPLETE":
        raise RuntimeError("bound E8 evidence is incomplete")
    if verdict.get("verdict") != "PASS_3B_FULL_PATH_ZCA":
        raise RuntimeError("bound E8 verdict is not PASS_3B_FULL_PATH_ZCA")
    measurement = result["scientific"]["measurement"]
    candidate = measurement["arms"]["candidate"]
    if stable_json_sha256(candidate["direct_rows"]) != E8_DIRECT_SHA256:
        raise RuntimeError("bound E8 candidate direct rows changed")
    if stable_json_sha256(candidate["memory_receipt"]) != E8_MEMORY_SHA256:
        raise RuntimeError("bound E8 candidate memory receipt changed")
    if admission["scientific"]["measurement"] != measurement:
        raise RuntimeError("bound E8 admission and replay measurement differ")
    return {
        "e8_implementation_commit": E8_IMPLEMENTATION_COMMIT,
        "e8_admission_sha256": E8_ADMISSION_SHA256,
        "e8_result_sha256": E8_RESULT_SHA256,
        "e8_verdict_sha256": E8_VERDICT_SHA256,
        "e8_verdict": "PASS_3B_FULL_PATH_ZCA",
        "e8_direct_sha256": E8_DIRECT_SHA256,
        "e8_memory_sha256": E8_MEMORY_SHA256,
        "counterfact_sha256": COUNTERFACT_SHA256,
    }


def selected_population() -> dict[str, Any]:
    e8_prompts = frozen_e8_measurement()["prompt_order"]
    query_rows = load_query_rows(REPO_ROOT / COUNTERFACT_PATH, e8_prompts)
    prompts = query_prompt_order(query_rows)
    case_ids = [row["case_id"] for row in query_rows]
    kind_hashes = {
        kind: stable_json_sha256([row[kind] for row in query_rows]) for kind in KINDS
    }
    if stable_json_sha256(query_rows) != QUERY_ROW_SHA256:
        raise RuntimeError("E9 selected-row hash mismatch")
    if stable_json_sha256(case_ids) != QUERY_CASE_ID_SHA256:
        raise RuntimeError("E9 case-id hash mismatch")
    if stable_json_sha256(prompts) != QUERY_PROMPT_SHA256:
        raise RuntimeError("E9 prompt-order hash mismatch")
    if kind_hashes != QUERY_KIND_SHA256:
        raise RuntimeError("E9 kind-list hash mismatch")
    if set(prompts) & set(e8_prompts):
        raise RuntimeError("E9 query population overlaps E8 prompts")
    return {
        "query_rows": query_rows,
        "prompt_order": prompts,
        "case_ids": case_ids,
        "kind_sha256": kind_hashes,
    }


def registered_source_sha256(*, require_clean: bool = False) -> dict[str, str]:
    missing = [p for p in REGISTERED_SOURCE_PATHS if not (REPO_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"registered E9 sources missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E9 sources must be committed:\n" + dirty)
    return {p: sha256_file(REPO_ROOT / p) for p in REGISTERED_SOURCE_PATHS}


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("source_sha256") != registered_source_sha256():
        raise RuntimeError("registered E9 source drift")
    if receipt.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E9 dependency receipt mismatch")


def _verify_implementation_ancestry(commit: Any) -> None:
    if not isinstance(commit, str) or git_output("rev-parse", commit) != commit:
        raise RuntimeError("registered E9 implementation commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E9 preflight")
    payload = load_json_object(PREFLIGHT_PATH)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "query_case_start": QUERY_CASE_START,
        "query_case_stop": QUERY_CASE_STOP,
        "query_case_count": QUERY_CASE_COUNT,
        "query_count": QUERY_COUNT,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_case_id_sha256": QUERY_CASE_ID_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "query_kind_sha256": QUERY_KIND_SHA256,
        "model_forward_batch": MODEL_FORWARD_BATCH,
        "gate_batch": GATE_BATCH,
        "bound_inputs": verify_bound_inputs(),
        "e9_query_outcomes_computed": False,
        "query_population_materialized_without_model": QUERY_COUNT,
        "commands": list(REGISTERED_COMMANDS),
        "preflight_test_command": "python -m pytest tests/test_e9_3b.py -q",
        "preflight_tests_returncode": 0,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E9 preflight field mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E9 preflight model invariants mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E9 preflight native arithmetic mismatch")
    environment_failures = environment_validity_failures(payload.get("environment"))
    if environment_failures:
        raise RuntimeError(
            "invalid E9 preflight environment: " + "; ".join(environment_failures)
        )
    frozen = frozen_e8_measurement()
    if (
        payload.get("basis_and_directions")
        != load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"][
            "basis_and_directions"
        ]
    ):
        raise RuntimeError("E9 preflight basis differs from E8")
    memory_failures = e8_memory_compatibility_failures(
        payload.get("e8_memory_receipt"),
        frozen["arms"]["candidate"]["memory_receipt"],
    )
    memory_failures.extend(
        key_whitening_validity_failures(
            payload.get("key_whitening"), payload.get("e8_memory_receipt", {})
        )
    )
    if memory_failures:
        raise RuntimeError("invalid E9 preflight memory: " + "; ".join(memory_failures))
    anchor_failures = _anchor_failures(payload.get("e8_anchor_scan"), frozen["cases"])
    if anchor_failures:
        raise RuntimeError(
            "invalid E9 preflight anchors: " + "; ".join(anchor_failures)
        )
    _verify_implementation_ancestry(payload.get("implementation_commit"))
    _verify_source_receipt(payload)
    verify_protocol()
    verify_dependencies()
    verify_snapshot(snapshot_path())
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    if not EXECUTION_RECEIPT_PATH.is_file():
        raise FileNotFoundError("missing E9 execution receipt")
    payload = load_json_object(EXECUTION_RECEIPT_PATH)
    expected = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "preflight_sha256": preflight_sha,
        "preflight_file_sha256_recomputed": preflight_sha,
        "model_file_sha256": MODEL_FILE_SHA256,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "basis_and_directions": preflight["basis_and_directions"],
        "e8_memory_receipt": preflight["e8_memory_receipt"],
        "key_whitening": preflight["key_whitening"],
        "e8_anchor_scan": preflight["e8_anchor_scan"],
        "bound_inputs": preflight["bound_inputs"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "e9_query_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
        "test_command": "python -m pytest tests/test_e9_3b.py -q",
        "tests_returncode": 0,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E9 execution receipt field mismatch: {name}")
    if payload.get("environment") != preflight.get("environment"):
        raise RuntimeError("E9 execution environment differs from preflight")
    if payload.get("native_runtime") != preflight.get("native_runtime"):
        raise RuntimeError("E9 native runtime differs from preflight")
    _verify_source_receipt(payload)
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    if not ATTEMPT_PATH.is_file():
        raise FileNotFoundError("missing E9 burn marker")
    payload = load_json_object(ATTEMPT_PATH)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": execution_sha,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "pre_admission_outputs_absent": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E9 attempt field mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def environment_validity_failures(
    value: Any, *, expected_node: str | None = None
) -> list[str]:
    """Validate the E8 runtime fields plus E9's physical-node binding."""
    if not isinstance(value, dict):
        return ["E9 environment record missing"]
    failures = [
        f"E9 environment field mismatch: {name}"
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if value.get(name) != expected
    ]
    node = value.get("node")
    if not isinstance(node, str) or not node:
        failures.append("E9 physical node is missing")
    elif expected_node is not None and node != expected_node:
        failures.append("E9 physical node differs from preflight")
    if set(value) != set(EXPECTED_ENVIRONMENT) | {"node"}:
        failures.append("E9 environment fields are not exhaustive")
    return failures


def e8_memory_compatibility_failures(
    actual: Any, expected: dict[str, Any] | None = None
) -> list[str]:
    """Require exact E8 memory provenance outside the three key hashes."""
    if not isinstance(actual, dict):
        return ["E9 reconstructed memory receipt missing"]
    if expected is None:
        expected = frozen_e8_measurement()["arms"]["candidate"]["memory_receipt"]
    failures: list[str] = []
    if set(actual) != set(expected):
        failures.append("E9 reconstructed memory receipt fields differ from E8")
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if name in E8_KEY_DRIFT_FIELDS:
            if not _valid_hash(actual_value):
                failures.append(f"E9 reconstructed memory has invalid {name}")
        elif actual_value != expected_value:
            failures.append(f"E9 reconstructed memory differs from E8: {name}")
    return failures


def key_whitening_validity_failures(
    record: Any, memory_receipt: dict[str, Any]
) -> list[str]:
    """Validate the pre-outcome-frozen E9 key operator and its receipt links."""
    if not isinstance(record, dict):
        return ["E9 key-whitening receipt missing"]
    expected = frozen_e8_measurement()["key_whitening"]
    drift_fields = {"mean_sha256", "transform_sha256"}
    failures: list[str] = []
    if set(record) != set(expected):
        failures.append("E9 key-whitening receipt fields differ from E8")
    for name, expected_value in expected.items():
        actual_value = record.get(name)
        if name in drift_fields:
            if not _valid_hash(actual_value):
                failures.append(f"E9 key-whitening receipt has invalid {name}")
        elif actual_value != expected_value:
            failures.append(f"E9 key-whitening receipt differs from E8: {name}")
    if record.get("mean_sha256") != memory_receipt.get("key_mean_sha256"):
        failures.append("E9 key-mean hashes disagree across receipts")
    if record.get("transform_sha256") != memory_receipt.get("key_transform_sha256"):
        failures.append("E9 key-transform hashes disagree across receipts")
    return failures


def _anchor_failures(anchor: Any, e8_cases: list[dict[str, Any]]) -> list[str]:
    if not isinstance(anchor, dict):
        return ["E8 anchor scan missing"]
    rows = anchor.get("rows")
    failures: list[str] = []
    if not isinstance(rows, list) or len(rows) != 64:
        return ["E8 anchor row count mismatch"]
    for index, (row, case) in enumerate(zip(rows, e8_cases, strict=True)):
        prefix = f"E8 anchor row {index}"
        score = row.get("actual_score")
        if (
            row.get("case_index") != index
            or row.get("case_id") != case["case_id"]
            or row.get("own_slot") != index
            or row.get("selected_slot") != index
            or row.get("gate_threshold") != GATE_THRESHOLD
        ):
            failures.append(prefix + " identity or slot mismatch")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            failures.append(prefix + " invalid score")
        elif row.get("gate_open") is not (float(score) >= GATE_THRESHOLD):
            failures.append(prefix + " gate decision mismatch")
        if (
            row.get("gate_open") is not True
            or row.get("delta_zero") is not False
            or row.get("hidden_bit_identical") is not False
        ):
            failures.append(prefix + " failed non-vacuity control")
        for name in (
            "prompt_sha256",
            "delta_sha256",
            "native_hidden_sha256",
            "adapted_hidden_sha256",
        ):
            if not _valid_hash(row.get(name)):
                failures.append(prefix + f" missing {name}")
        if row.get("native_hidden_sha256") == row.get("adapted_hidden_sha256"):
            failures.append(prefix + " changed-hidden hash is not distinct")
    expected_summary = {
        "anchor_count": 64,
        "gate_open_count": 64,
        "own_slot_count": 64,
        "nonzero_delta_count": 64,
    }
    if anchor.get("summary") != expected_summary:
        failures.append("E8 anchor summary mismatch")
    return failures


def _scan_failures(scan: Any, query_rows: list[dict[str, Any]]) -> list[str]:
    if not isinstance(scan, dict):
        return ["E9 query scan missing"]
    rows = scan.get("rows")
    failures: list[str] = []
    specs = query_specs(query_rows)
    if not isinstance(rows, list) or len(rows) != QUERY_COUNT:
        return ["E9 query scan row count mismatch"]
    e8_cases = frozen_e8_measurement()["cases"]
    labels = frozen_e8_measurement()["arms"]["candidate"]["memory_receipt"][
        "active_labels"
    ]
    for index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        prefix = f"E9 query row {index}"
        for field in ("pool_index", "case_index", "case_id", "kind", "prompt"):
            if row.get(field) != spec[field]:
                failures.append(prefix + f" field mismatch: {field}")
        if row.get("prompt_sha256") != stable_json_sha256(spec["prompt"]):
            failures.append(prefix + " prompt hash mismatch")
        if row.get("native_hidden_dtype") != "bfloat16" or row.get(
            "native_hidden_shape"
        ) != [MODEL_HIDDEN_SIZE]:
            failures.append(prefix + " native hidden tensor mismatch")
        score = row.get("selected_score")
        slot = row.get("selected_slot")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
        ):
            failures.append(prefix + " selected score is invalid")
        if not isinstance(slot, int) or not 0 <= slot < 64:
            failures.append(prefix + " selected slot is inactive")
        elif (
            row.get("selected_e8_case_id") != e8_cases[slot]["case_id"]
            or row.get("selected_label") != labels[slot]
        ):
            failures.append(prefix + " selected slot mapping mismatch")
        expected_gate = bool(
            isinstance(score, (int, float))
            and math.isfinite(float(score))
            and float(score) >= GATE_THRESHOLD
        )
        if (
            row.get("gate_threshold") != GATE_THRESHOLD
            or row.get("gate_open") is not expected_gate
        ):
            failures.append(prefix + " gate decision mismatch")
        start = (index // GATE_BATCH) * GATE_BATCH
        if (
            row.get("gate_batch_start") != start
            or row.get("gate_batch_size") != GATE_BATCH
            or row.get("gate_batch_offset") != index - start
        ):
            failures.append(prefix + " gate batch mismatch")
        if row.get("delta_dtype") != "bfloat16" or row.get("delta_shape") != [
            MODEL_HIDDEN_SIZE
        ]:
            failures.append(prefix + " delta tensor mismatch")
        for name in (
            "native_hidden_sha256",
            "delta_sha256",
            "adapted_hidden_sha256",
        ):
            if not _valid_hash(row.get(name)):
                failures.append(prefix + f" missing {name}")
        if expected_gate:
            if (
                row.get("delta_zero") is not False
                or row.get("hidden_bit_identical") is not False
                or row.get("native_hidden_sha256") == row.get("adapted_hidden_sha256")
            ):
                failures.append(prefix + " open gate did not change the hidden")
        elif (
            row.get("delta_zero") is not True
            or row.get("hidden_bit_identical") is not True
            or row.get("native_hidden_sha256") != row.get("adapted_hidden_sha256")
        ):
            failures.append(prefix + " closed gate changed the hidden")
    try:
        expected_summary = summarize_query_scan(rows)
    except Exception as error:
        failures.append(f"E9 query summary exception: {error}")
    else:
        if scan.get("summary") != expected_summary:
            failures.append("E9 query summary mismatch")
    return failures


def measurement_validity_failures(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return ["E9 measurement missing"]
    failures: list[str] = []
    population = selected_population()
    expected_metadata = {
        "query_case_start": QUERY_CASE_START,
        "query_case_stop": QUERY_CASE_STOP,
        "query_case_count": QUERY_CASE_COUNT,
        "query_count": QUERY_COUNT,
        "query_row_sha256": QUERY_ROW_SHA256,
        "query_case_id_sha256": QUERY_CASE_ID_SHA256,
        "query_prompt_sha256": QUERY_PROMPT_SHA256,
        "query_kind_sha256": QUERY_KIND_SHA256,
        "model_forward_batch": MODEL_FORWARD_BATCH,
        "model_forward_batch_count": QUERY_COUNT // MODEL_FORWARD_BATCH,
        "gate_batch": GATE_BATCH,
        "gate_batch_count": QUERY_COUNT // GATE_BATCH,
    }
    if measurement.get("population") != expected_metadata:
        failures.append("E9 population metadata mismatch")
    frozen = frozen_e8_measurement()
    memory_receipt = measurement.get("e8_memory_receipt")
    failures.extend(
        e8_memory_compatibility_failures(
            memory_receipt, frozen["arms"]["candidate"]["memory_receipt"]
        )
    )
    failures.extend(
        key_whitening_validity_failures(
            measurement.get("key_whitening"),
            memory_receipt if isinstance(memory_receipt, dict) else {},
        )
    )
    failures.extend(
        _anchor_failures(measurement.get("e8_anchor_scan"), frozen["cases"])
    )
    failures.extend(
        _scan_failures(measurement.get("query_scan"), population["query_rows"])
    )
    return failures


def admission_validity_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        failures.append("admission schema or status mismatch")
    try:
        verify_scientific_hash(payload, "E9 admission")
    except Exception as error:
        return failures + [str(error)]
    scientific = payload["scientific"]
    if scientific.get("model_invariants") != expected_model_invariants():
        failures.append("admission model invariants mismatch")
    frozen_basis = load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"][
        "basis_and_directions"
    ]
    if scientific.get("basis_and_directions") != frozen_basis:
        failures.append("admission basis differs from E8")
    failures.extend(measurement_validity_failures(scientific.get("measurement")))
    return failures


def verify_admission_provenance(
    payload: dict[str, Any],
    *,
    receipt: dict[str, Any],
    execution_sha: str,
    attempt_sha: str,
) -> None:
    expected = {
        "attempt_sha256": attempt_sha,
        "execution_receipt_sha256": execution_sha,
        "implementation_commit": receipt["implementation_commit"],
        "native_runtime": receipt["native_runtime"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E9 admission provenance mismatch: {name}")
    if payload.get("scientific", {}).get("basis_and_directions") != receipt.get(
        "basis_and_directions"
    ):
        raise RuntimeError("E9 admission basis differs from registration")
    measurement = payload.get("scientific", {}).get("measurement", {})
    if measurement.get("e8_memory_receipt") != receipt.get("e8_memory_receipt"):
        raise RuntimeError("E9 admission memory differs from registration")
    if measurement.get("key_whitening") != receipt.get("key_whitening"):
        raise RuntimeError("E9 admission key whitening differs from registration")


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    if not ADMISSION_PATH.is_file():
        raise FileNotFoundError("missing E9 admission")
    payload = load_json_object(ADMISSION_PATH)
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E9 admission schema or status mismatch")
    verify_admission_provenance(
        payload, receipt=receipt, execution_sha=execution_sha, attempt_sha=attempt_sha
    )
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("invalid E9 admission: " + "; ".join(failures))
    return payload, sha256_file(ADMISSION_PATH)


def result_validity_failures(
    result: dict[str, Any], admission: dict[str, Any]
) -> list[str]:
    try:
        verify_scientific_hash(result, "E9 result")
    except Exception as error:
        return [str(error)]
    scientific = result["scientific"]
    frozen = admission["scientific"]
    failures: list[str] = []
    if scientific.get("admission_scientific_sha256") != admission.get(
        "scientific_sha256"
    ):
        failures.append("result binds a different admission")
    for name in ("model_invariants", "basis_and_directions", "measurement"):
        if scientific.get(name) != frozen.get(name):
            failures.append(f"new-process replay differs from admission: {name}")
    for name in ("measurement_exact_match", "all_tensor_hashes_match"):
        if scientific.get(name) is not True:
            failures.append(f"result replay bar failed: {name}")
    return failures


def verify_result_provenance(
    payload: dict[str, Any],
    *,
    receipt: dict[str, Any],
    admission: dict[str, Any],
    admission_sha: str,
    attempt_sha: str,
) -> None:
    expected = {
        "attempt_sha256": attempt_sha,
        "admission_sha256": admission_sha,
        "admission_scientific_sha256": admission["scientific_sha256"],
        "native_runtime": receipt["native_runtime"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E9 result provenance mismatch: {name}")


def load_result() -> tuple[dict[str, Any], str]:
    receipt, _execution_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    if not RESULT_PATH.is_file():
        raise FileNotFoundError("missing E9 replay result")
    payload = load_json_object(RESULT_PATH)
    if payload.get("schema") != RESULT_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E9 result schema or status mismatch")
    verify_result_provenance(
        payload,
        receipt=receipt,
        admission=admission,
        admission_sha=admission_sha,
        attempt_sha=attempt_sha,
    )
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E9 result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


def scientific_failures(result: dict[str, Any]) -> list[str]:
    summary = (
        result.get("scientific", {})
        .get("measurement", {})
        .get("query_scan", {})
        .get("summary", {})
    )
    failures: list[str] = []
    if summary.get("query_count") != QUERY_COUNT:
        failures.append("E9 scientific denominator is not 12,288")
    if summary.get("gate_open_count") != 0:
        failures.append("one or more disjoint E9 queries opened the gate")
    by_kind = summary.get("by_kind", {})
    for kind in KINDS:
        if by_kind.get(kind, {}).get("query_count") != QUERY_CASE_COUNT:
            failures.append(f"E9 {kind} denominator is not 4,096")
        if by_kind.get(kind, {}).get("gate_open_count") != 0:
            failures.append(f"one or more E9 {kind} queries opened the gate")
    return failures


def verdict_scientific_payload(
    *,
    verdict: str,
    result: dict[str, Any] | None,
    validity_failures: list[str],
    scientific_failures_value: list[str],
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "summary": (
            None
            if result is None
            else result.get("scientific", {})
            .get("measurement", {})
            .get("query_scan", {})
            .get("summary")
        ),
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Pinned frozen SmolLM3-3B-Base, registered A100/BF16 physical node "
            "and runtime, pre-outcome-frozen 64-slot E8-recipe exact-key adapter, "
            "and deterministic 12,288-query CounterFact locality pool."
        ),
    }


__all__ = [
    "ADMISSION_PATH",
    "ATTEMPT_PATH",
    "EXECUTION_RECEIPT_PATH",
    "PREFLIGHT_PATH",
    "RESULT_PATH",
    "VERDICT_PATH",
    "atomic_create_json",
    "scientific_sha256",
]
