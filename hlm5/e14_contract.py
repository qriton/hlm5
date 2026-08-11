"""Fail-closed provenance and scientific contract for HLM5 E14."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

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
from .e9_contract import selected_population as selected_e9_population
from .e10_contract import selected_population as selected_e10_population
from .e12_contract import selected_population as selected_e12_population
from .e13_contract import selected_population as selected_e13_population
from .e14_runtime import (
    ACTIVE_COUNT,
    ARMS,
    COSINE,
    EVAL_CANDIDATE_COUNT,
    HLM,
    IDENTITY,
    LOCALITY_COUNT,
    TRAIN_COUNT,
    fact_record,
    load_records,
    locality_prompt_order,
    locality_record,
    select_locality,
    select_train_and_eval,
    scientific_summary,
    training_config,
)


PROTOCOL_COMMIT = "ed4082bf765b7420204798049a156d65ca9226b8"
PROTOCOL_PATH = "docs/e14-3b-frozen-router-training-prereg-2026-08-11.md"
PROTOCOL_SHA256 = "96528a33993e9877f9243c5b732fc2943f3bac03762e2d0129ae87f0459d2c3b"
COUNTERFACT_PATH = "scripts/data/counterfact.json"
COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
E8_RESULT_PATH = "results/e8_3b_evidence/result.json"
E13_VERDICT_PATH = "results/e13_3b_evidence/verdict.json"
E13_VERDICT_SHA256 = "e430b6dd13d478048681aa7a41478c11d7c7538be5a4efbb7fd29b53e1938389"

TRAIN_EXPECTED = {
    "count": 256,
    "first_case_id": 2_508,
    "last_case_id": 3_415,
    "row_sha256": "db43b7e28da0e4616f6163b5049cb7895c6392adf3c54c5ae957258425d4e000",
    "case_id_sha256": "b0657598f2cde50b0c82d11f61136e6cf20b785dd87da15c952d072e02004bae",
    "exact_sha256": "1b8e26504cc94a36051b5e9a2c4cd40dd95702d22b1ea1d83934b670eeb9bb18",
    "paraphrase_sha256": "05f489e481bec9b6f19011dae37758721c63fcb3dd36e8ede38b1c289d26cf22",
    "target_id_sha256": "8312204e023f35f59f86d0b62028d186debd8de474ebd3860a71c6be299cabef",
    "target_sha256": "51062ff76e1ef5995a162665ec96f2a9961e8c6080791162cd38e421651a9e9d",
}
EVAL_EXPECTED = {
    "count": 1_200,
    "first_case_id": 3_420,
    "last_case_id": 8_678,
    "row_sha256": "229c2921e1643802a9a95ffc9c028f765c252b43f9c89dd0799e35a836f2fe19",
    "case_id_sha256": "b7e37abf71506f490a9803541c00d488ef5b3130125126233e8c000e68437fd4",
    "exact_sha256": "e1182b38ad54e72fc4612adae4fc2b770f5eb3d8f775984931a5b5b57071cd1c",
    "paraphrase_sha256": "e40c50970378a8b35e68a8353686bbbf825c51507dc557e901caff87627f3162",
    "target_id_sha256": "4cdaa7264a671b1e5927453c497d20934085b5701a28ebbbea6bd0c6bb464dbb",
    "target_sha256": "612386afed50c5af1986256010952873d9a95005a2302f984ff0b3fa82efd8e6",
}
LOCALITY_EXPECTED = {
    "case_count": 998,
    "prompt_count": 2_994,
    "first_case_id": 405,
    "last_case_id": 21_917,
    "row_sha256": "c647ea97e4d846c82b2d815a3704ee20b26970a510da96d5ac518b6f61d51019",
    "case_id_sha256": "ae512eaf6d80bcd8ce0acd65cb499378baa9442c8f33b080ae260f952b52c353",
    "prompt_order_sha256": "625c6dbd760800728c5593e2f56ed5a34e9cccc38519bd934a4aa4b454f19ecc",
    "kind_sha256": {
        "exact": "3111bc89715ab167ee77bc4662fe74c72b51102a40e5fc9d227bb745d497047c",
        "paraphrase": "bf6acfc6bfd237e9b63714df1c3d960501495d6bb0a27cb0036c3bebfbc3a0e9",
        "neighborhood": "280911964a0c5aeda81fb3dd1a34a4b8852a8ad229ea22918397330bc641d311",
    },
}

PREFLIGHT_SCHEMA = "hlm5-e14-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e14-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e14-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e14-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e14-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e14-3b-verdict-v1"

REGISTERED_SOURCE_PATHS = (
    "hlm5/e14_contract.py",
    "hlm5/e14_runtime.py",
    "scripts/preflight_e14_3b.py",
    "scripts/register_e14_3b.py",
    "scripts/admit_e14_3b.py",
    "scripts/replay_e14_3b.py",
    "scripts/verify_e14_3b.py",
    "tests/test_e14_3b.py",
    "slurm/e14_3b_leonardo.slurm",
)
DEPENDENCY_PATHS = (
    "hlm5/__init__.py",
    "hlm5/certify.py",
    "hlm5/e7_contract.py",
    "hlm5/e7_runtime.py",
    "hlm5/e7b_contract.py",
    "hlm5/e7b_runtime.py",
    "hlm5/e7c_runtime.py",
    "hlm5/e8_contract.py",
    "hlm5/e8_runtime.py",
    "hlm5/e9_contract.py",
    "hlm5/e10_contract.py",
    "hlm5/e12_contract.py",
    "hlm5/e13_contract.py",
    "hlm5/e13_runtime.py",
    "hlm5/io.py",
    "hlm5/key_value.py",
    "hlm5/memory.py",
    "hlm5/model.py",
    "hlm5/public_adapter.py",
)
REGISTERED_COMMANDS = tuple(
    f"python scripts/{name}_e14_3b.py"
    for name in ("preflight", "register", "admit", "replay", "verify")
)
TEST_COMMAND = "python -m pytest tests/test_e14_3b.py -q"

STAGING_DIR = REPO_ROOT / "results" / "e14_3b_staging"
PREFLIGHT_PATH = STAGING_DIR / "preflight.json"
EXECUTION_RECEIPT_PATH = STAGING_DIR / "execution_receipt.json"
ATTEMPT_PATH = STAGING_DIR / "attempt.json"
BUNDLE_PATH = STAGING_DIR / "routers.pt"
ADMISSION_PATH = STAGING_DIR / "admission.json"
RESULT_PATH = STAGING_DIR / "result.json"
VERDICT_PATH = STAGING_DIR / "verdict.json"
NAMED_OUTPUTS = (
    PREFLIGHT_PATH,
    EXECUTION_RECEIPT_PATH,
    ATTEMPT_PATH,
    BUNDLE_PATH,
    ADMISSION_PATH,
    RESULT_PATH,
    VERDICT_PATH,
)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _staging_path(path: Path) -> Path:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"E14 output must stay under {staging}: {resolved}")
    return resolved


def atomic_create_json(path: Path, value: Any) -> None:
    resolved = _staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(resolved)
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    handle, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, resolved)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_create_bundle(path: Path, payload: dict[str, torch.Tensor]) -> str:
    resolved = _staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(resolved)
    handle, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.link(temporary, resolved)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256_file(resolved)


def load_bundle(path: Path, manifest: dict[str, Any]) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or set(value) != set(manifest):
        raise RuntimeError("E14 bundle members differ from manifest")
    from .e7_contract import tensor_sha256

    for name, tensor in value.items():
        actual = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        if actual != manifest[name]:
            raise RuntimeError(f"E14 bundle tensor drift: {name}")
    return value


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError("E14 create-new output already exists: " + ", ".join(present))


def scientific_sha256(value: Any) -> str:
    return stable_json_sha256(value)


def registered_hashes(paths: Iterable[str], *, require_clean: bool = False) -> dict[str, str]:
    values = tuple(paths)
    missing = [path for path in values if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"registered E14 files missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *values)
        if dirty:
            raise RuntimeError("registered E14 files must be committed:\n" + dirty)
    return {path: sha256_file(REPO_ROOT / path) for path in values}


def verify_protocol() -> None:
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E14 protocol")
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("E14 protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def verify_bound_inputs() -> dict[str, Any]:
    require_file_hash(REPO_ROOT / COUNTERFACT_PATH, COUNTERFACT_SHA256, "CounterFact")
    require_file_hash(REPO_ROOT / E13_VERDICT_PATH, E13_VERDICT_SHA256, "E13 verdict")
    verdict = load_json_object(REPO_ROOT / E13_VERDICT_PATH)
    if verdict.get("verdict") != "PASS_3B_READOUT_REPAIR_1024":
        raise RuntimeError("bound E13 verdict is not terminal PASS")
    verify_snapshot(snapshot_path())
    return {
        "counterfact_sha256": COUNTERFACT_SHA256,
        "e13_verdict_sha256": E13_VERDICT_SHA256,
        "e13_verdict": verdict["verdict"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
    }


def selected_population(tokenizer: Any) -> dict[str, Any]:
    e8 = load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"]["measurement"]
    e9 = selected_e9_population()
    e10 = selected_e10_population(tokenizer)
    e12 = selected_e12_population(tokenizer)
    e13 = selected_e13_population(tokenizer)
    prior = (
        list(e8["prompt_order"])
        + list(e9["prompt_order"])
        + list(e10["prompt_order"])
        + list(e12["prompt_order"])
        + list(e12["query_prompt_order"])
        + list(e13["prompt_order"])
        + list(e13["query_prompt_order"])
    )
    records = load_records(REPO_ROOT / COUNTERFACT_PATH)
    train, evaluation = select_train_and_eval(records, tokenizer, prior)
    reserved = [row[name] for row in train + evaluation for name in ("exact", "paraphrase")]
    locality = select_locality(records, prior + reserved)
    train_actual = fact_record(train)
    eval_actual = fact_record(evaluation)
    locality_actual = locality_record(locality)
    if train_actual != TRAIN_EXPECTED:
        raise RuntimeError(f"E14 training population mismatch: {train_actual}")
    if eval_actual != EVAL_EXPECTED:
        raise RuntimeError(f"E14 evaluation population mismatch: {eval_actual}")
    if locality_actual != LOCALITY_EXPECTED:
        raise RuntimeError(f"E14 locality population mismatch: {locality_actual}")
    all_prompts = prior + reserved + locality_prompt_order(locality)
    if len(all_prompts) != len(set(all_prompts)):
        raise RuntimeError("E14 global prompt disjointness failed")
    return {
        "train": train,
        "evaluation": evaluation,
        "locality": locality,
        "train_record": train_actual,
        "evaluation_record": eval_actual,
        "locality_record": locality_actual,
        "calibration_prompts": list(e8["prompt_order"][-18:]),
    }


def environment_failures(value: Any, *, node: str | None = None) -> list[str]:
    if not isinstance(value, dict):
        return ["E14 environment missing"]
    failures = [
        f"E14 environment mismatch: {name}"
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if value.get(name) != expected
    ]
    if not isinstance(value.get("node"), str) or not value["node"]:
        failures.append("E14 node missing")
    elif node is not None and value["node"] != node:
        failures.append("E14 node differs from registration")
    if set(value) != set(EXPECTED_ENVIRONMENT) | {"node"}:
        failures.append("E14 environment fields are not exhaustive")
    return failures


def load_preflight() -> tuple[dict[str, Any], str]:
    payload = load_json_object(PREFLIGHT_PATH)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "train_population": TRAIN_EXPECTED,
        "evaluation_population": EVAL_EXPECTED,
        "locality_population": LOCALITY_EXPECTED,
        "training_config": training_config(),
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": TEST_COMMAND,
        "tests_returncode": 0,
        "outcomes_computed": False,
        "bound_inputs": verify_bound_inputs(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E14 preflight mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E14 model invariant mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E14 native runtime mismatch")
    failures = environment_failures(payload.get("environment"))
    if failures:
        raise RuntimeError("; ".join(failures))
    implementation = payload.get("implementation_commit")
    if not isinstance(implementation, str) or git_output("rev-parse", implementation) != implementation:
        raise RuntimeError("E14 implementation commit unavailable")
    if payload.get("source_sha256") != registered_hashes(REGISTERED_SOURCE_PATHS):
        raise RuntimeError("E14 registered source drift")
    if payload.get("dependency_sha256") != registered_hashes(DEPENDENCY_PATHS):
        raise RuntimeError("E14 dependency drift")
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    payload = load_json_object(EXECUTION_RECEIPT_PATH)
    expected = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "preflight_sha256": preflight_sha,
        "implementation_commit": preflight["implementation_commit"],
        "source_sha256": preflight["source_sha256"],
        "dependency_sha256": preflight["dependency_sha256"],
        "environment": preflight["environment"],
        "native_runtime": preflight["native_runtime"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "pre_attempt_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E14 execution receipt mismatch: {name}")
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, receipt_sha = load_execution_receipt()
    payload = load_json_object(ATTEMPT_PATH)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": receipt_sha,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
        "train_row_sha256": TRAIN_EXPECTED["row_sha256"],
        "evaluation_row_sha256": EVAL_EXPECTED["row_sha256"],
        "locality_row_sha256": LOCALITY_EXPECTED["row_sha256"],
        "pre_admission_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E14 attempt mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def measurement_failures(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return ["E14 measurement missing"]
    failures: list[str] = []
    if measurement.get("population") != {
        "train": TRAIN_EXPECTED,
        "evaluation": EVAL_EXPECTED,
        "locality": LOCALITY_EXPECTED,
    }:
        failures.append("E14 population receipt mismatch")
    training = measurement.get("training", {})
    bundle_tensors = measurement.get("bundle", {}).get("tensors", {})
    key_record = measurement.get("key_whitening", {})
    if key_record.get("fit_count") != TRAIN_COUNT + EVAL_CANDIDATE_COUNT + 18:
        failures.append("E14 whitening fit denominator mismatch")
    if key_record.get("mean_sha256") != bundle_tensors.get("key_mean", {}).get("sha256"):
        failures.append("E14 whitening mean is not bound to bundle")
    if key_record.get("transform_sha256") != bundle_tensors.get("key_transform", {}).get("sha256"):
        failures.append("E14 whitening transform is not bound to bundle")
    for arm in (HLM, COSINE):
        rows = training.get(arm, {}).get("telemetry")
        if not isinstance(rows, list) or [row.get("step") for row in rows] != [0, 1, 50, 100, 200, 400]:
            failures.append(f"E14 {arm} telemetry schedule mismatch")
            continue
        for row in rows:
            for name in ("loss", "cross_entropy", "identity_displacement_mse", "parameter_update_norm"):
                if not math_isfinite(row.get(name)):
                    failures.append(f"E14 {arm} nonfinite {name}")
            if not math_isfinite(row.get("gradient_norm")):
                failures.append(f"E14 {arm} nonfinite gradient")
        if rows[-1].get("parameter_update_norm", 0) <= 0:
            failures.append(f"E14 {arm} router did not update")
        prefix = "hlm" if arm == HLM else "cosine"
        router = training.get(arm, {}).get("router", {})
        if router.get("a_sha256") != bundle_tensors.get(f"{prefix}_a", {}).get("sha256"):
            failures.append(f"E14 {arm} A tensor is not bound to bundle")
        if router.get("b_sha256") != bundle_tensors.get(f"{prefix}_b", {}).get("sha256"):
            failures.append(f"E14 {arm} B tensor is not bound to bundle")
    evaluation = measurement.get("evaluation", {})
    direct = evaluation.get("direct", {})
    direct_rows = direct.get("rows")
    if not isinstance(direct_rows, list) or len(direct_rows) != EVAL_CANDIDATE_COUNT:
        failures.append("E14 direct denominator mismatch")
        direct_rows = []
    admitted_indices: list[int] = []
    for index, row in enumerate(direct_rows):
        if row.get("case_index") != index:
            failures.append(f"E14 direct row {index} index mismatch")
        eligible = row.get("baseline_correct") is False
        if row.get("eligible") is not eligible:
            failures.append(f"E14 direct row {index} eligibility mismatch")
        admitted = bool(eligible and row.get("decision") == "ADMIT_NATIVE")
        if row.get("deployment_admitted") is not admitted:
            failures.append(f"E14 direct row {index} admission mismatch")
        if admitted:
            admitted_indices.append(index)
    expected_selected = admitted_indices[:ACTIVE_COUNT]
    if direct.get("selected_indices") != expected_selected:
        failures.append("E14 selected indices were not the first direct admissions")
    if direct.get("selected_count") != len(expected_selected):
        failures.append("E14 selected count mismatch")
    if direct.get("selected_indices_sha256") != stable_json_sha256(expected_selected):
        failures.append("E14 selected-index hash mismatch")
    if direct.get("selected_count", 0) > ACTIVE_COUNT:
        failures.append("E14 selected count exceeds frozen tier")
    arms = evaluation.get("arms", {})
    if direct.get("selected_count") == ACTIVE_COUNT and set(arms) != set(ARMS):
        failures.append("E14 evaluation arms missing")
    for arm, value in arms.items():
        for kind in ("exact", "paraphrase"):
            rows = value.get(kind, {}).get("rows")
            if not isinstance(rows, list) or len(rows) != ACTIVE_COUNT:
                failures.append(f"E14 {arm} {kind} denominator mismatch")
                continue
            for offset, row in enumerate(rows):
                if row.get("candidate_index") != expected_selected[offset]:
                    failures.append(f"E14 {arm} {kind} row {offset} candidate mismatch")
                if row.get("own_slot") != offset:
                    failures.append(f"E14 {arm} {kind} row {offset} own-slot mismatch")
            expected_summary = {
                "count": len(rows),
                "gate_open_count": sum(row.get("gate_open") is True for row in rows),
                "own_slot_count": sum(
                    row.get("selected_slot") == row.get("own_slot") for row in rows
                ),
                "strict_target_success_count": sum(
                    row.get("strict_target_success") is True for row in rows
                ),
                "nonzero_delta_count": sum(row.get("delta_zero") is False for row in rows),
                "minimum_margin": min(float(row["target_margin"]) for row in rows),
            }
            if value.get(kind, {}).get("summary") != expected_summary:
                failures.append(f"E14 {arm} {kind} summary mismatch")
        local = value.get("locality", {}).get("rows")
        if not isinstance(local, list) or len(local) != LOCALITY_COUNT:
            failures.append(f"E14 {arm} locality denominator mismatch")
        else:
            for index, row in enumerate(local):
                expected_kind = ("exact", "paraphrase", "neighborhood")[index % 3]
                if row.get("pool_index") != index or row.get("kind") != expected_kind:
                    failures.append(f"E14 {arm} locality row {index} order mismatch")
            expected_by_kind = {
                kind: {
                    "count": sum(row.get("kind") == kind for row in local),
                    "gate_open_count": sum(
                        row.get("kind") == kind and row.get("gate_open") is True
                        for row in local
                    ),
                    "nonzero_delta_count": sum(
                        row.get("kind") == kind and row.get("delta_zero") is False
                        for row in local
                    ),
                    "changed_hidden_count": sum(
                        row.get("kind") == kind
                        and row.get("hidden_bit_identical") is False
                        for row in local
                    ),
                }
                for kind in ("exact", "paraphrase", "neighborhood")
            }
            expected_local_summary = {
                "count": len(local),
                "gate_open_count": sum(row.get("gate_open") is True for row in local),
                "nonzero_delta_count": sum(
                    row.get("delta_zero") is False for row in local
                ),
                "changed_hidden_count": sum(
                    row.get("hidden_bit_identical") is False for row in local
                ),
                "by_kind": expected_by_kind,
            }
            if value.get("locality", {}).get("summary") != expected_local_summary:
                failures.append(f"E14 {arm} locality summary mismatch")
        memory = value.get("memory", {})
        if memory.get("physical_slots") != 1_032 or memory.get("active_slots") != ACTIVE_COUNT:
            failures.append(f"E14 {arm} memory shape mismatch")
        rollback = value.get("rollback", {})
        if rollback.get("exact_count") != ACTIVE_COUNT or rollback.get("locality_count") != LOCALITY_COUNT:
            failures.append(f"E14 {arm} rollback denominator mismatch")
    for name in ("trunk_parameter_sha256_before", "trunk_parameter_sha256_after"):
        if not _hash(measurement.get(name)):
            failures.append(f"E14 invalid {name}")
    return failures


def math_isfinite(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, receipt_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    payload = load_json_object(ADMISSION_PATH)
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E14 admission is not COMPLETE")
    if payload.get("attempt_sha256") != attempt_sha:
        raise RuntimeError("E14 admission attempt chain mismatch")
    if payload.get("execution_receipt_sha256") != receipt_sha:
        raise RuntimeError("E14 admission execution-receipt chain mismatch")
    if payload.get("implementation_commit") != receipt["implementation_commit"]:
        raise RuntimeError("E14 admission implementation commit mismatch")
    if payload.get("native_runtime") != receipt["native_runtime"]:
        raise RuntimeError("E14 admission native runtime mismatch")
    if payload.get("scientific_sha256") != scientific_sha256(payload.get("scientific")):
        raise RuntimeError("E14 admission scientific hash mismatch")
    failures = measurement_failures(payload.get("scientific", {}).get("measurement"))
    if failures:
        raise RuntimeError("invalid E14 admission: " + "; ".join(failures))
    bundle = payload["scientific"]["measurement"].get("bundle", {})
    if bundle.get("file_sha256") != sha256_file(BUNDLE_PATH):
        raise RuntimeError("E14 admission bundle file hash mismatch")
    load_bundle(BUNDLE_PATH, bundle.get("tensors", {}))
    return payload, sha256_file(ADMISSION_PATH)


def load_result() -> tuple[dict[str, Any], str]:
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    receipt, _receipt_sha = load_execution_receipt()
    payload = load_json_object(RESULT_PATH)
    if payload.get("schema") != RESULT_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E14 result is not COMPLETE")
    if payload.get("admission_sha256") != admission_sha:
        raise RuntimeError("E14 result admission chain mismatch")
    if payload.get("attempt_sha256") != attempt_sha:
        raise RuntimeError("E14 result attempt chain mismatch")
    if payload.get("admission_scientific_sha256") != admission["scientific_sha256"]:
        raise RuntimeError("E14 result admission scientific hash mismatch")
    if payload.get("native_runtime") != receipt["native_runtime"]:
        raise RuntimeError("E14 result native runtime mismatch")
    if payload.get("scientific_sha256") != scientific_sha256(payload.get("scientific")):
        raise RuntimeError("E14 result scientific hash mismatch")
    science = payload["scientific"]
    if not science.get("measurement_exact_match") or not science.get("bundle_exact_match"):
        raise RuntimeError("E14 exact replay failed")
    if science.get("measurement") != admission["scientific"]["measurement"]:
        raise RuntimeError("E14 replay measurement differs from admission")
    return payload, sha256_file(RESULT_PATH)


def binding_verdict(measurement: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    failures = measurement_failures(measurement)
    summary = scientific_summary(measurement)
    if failures:
        return "IMPLEMENTATION_INVALID", failures, summary
    if summary["selected_count"] != ACTIVE_COUNT:
        return "FAIL_3B_ROUTER_TRAINING", ["fewer than 1024 direct admissions"], summary
    hlm = summary["arms"][HLM]
    identity = summary["arms"][IDENTITY]
    bars: list[str] = []
    if hlm["train_correct"] != TRAIN_COUNT:
        bars.append("HLM training own-slot accuracy below 256/256")
    if hlm["paraphrase_own"] < 103:
        bars.append("HLM paraphrase own-slot count below 103")
    if hlm["paraphrase_strict"] < 103:
        bars.append("HLM paraphrase strict count below 103")
    if hlm["paraphrase_own"] - identity["paraphrase_own"] < 52:
        bars.append("HLM paraphrase own-slot gain below 52")
    if hlm["paraphrase_strict"] - identity["paraphrase_strict"] < 52:
        bars.append("HLM paraphrase strict gain below 52")
    for name in ("exact_gate", "exact_own", "exact_strict", "exact_nonzero"):
        if hlm[name] != ACTIVE_COUNT:
            bars.append(f"HLM {name} below {ACTIVE_COUNT}")
    if any(hlm[name] != 0 for name in ("locality_gate", "locality_nonzero", "locality_changed")):
        bars.append("HLM locality is not hard zero")
    for arm in ARMS:
        rollback = summary["arms"][arm]["rollback"]
        expected = {
            "exact_count": ACTIVE_COUNT,
            "locality_count": LOCALITY_COUNT,
            "exact_gate_open_count": 0,
            "locality_gate_open_count": 0,
            "exact_bit_identical": True,
            "locality_bit_identical": True,
            "exact_nonzero_delta": 0,
            "locality_nonzero_delta": 0,
        }
        if rollback != expected:
            bars.append(f"{arm} rollback mismatch")
    if not summary["trunk_unchanged"]:
        bars.append("frozen trunk parameter hash changed")
    verdict = "PASS_3B_ROUTER_TRAINING" if not bars else "FAIL_3B_ROUTER_TRAINING"
    return verdict, bars, summary
