"""Fail-closed provenance and scientific contract for HLM5 E10."""

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
from .e8_runtime import direct_summary
from .e10_runtime import (
    CANDIDATE_COUNT,
    CANDIDATE_START,
    CAPACITY_TIERS,
    GATE_BATCH,
    GATE_THRESHOLD,
    MEMORY_SIZE,
    MODEL_FORWARD_BATCH,
    load_capacity_cases,
    population_record,
    summarize_locality,
)


PROTOCOL_COMMIT = "f73b657012f9b28593a92c521670390087c306a3"
PROTOCOL_PATH = "docs/e10-3b-capacity-interference-prereg-2026-08-11.md"
PROTOCOL_SHA256 = "3b3f37902599b76073a6722446b7f17b4c48a9b2542235c97e08fada1c4f65be"

COUNTERFACT_PATH = "scripts/data/counterfact.json"
COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
E8_RESULT_PATH = "results/e8_3b_evidence/result.json"
E8_RESULT_SHA256 = "ec064b4d38c1abb9d88f99bd1c138b01cbfe4f9b15dbdf28edbe415477e678b6"
E9_ADMISSION_PATH = "results/e9_3b_evidence/admission.json"
E9_ADMISSION_SHA256 = "1d67111e3d7d3bba8d59cdd7d04b2197ed380ebce4b3ec00c96982a36a94551e"
E9_RESULT_PATH = "results/e9_3b_evidence/result.json"
E9_RESULT_SHA256 = "4248a926676b53c7eab005bc87f894d8e7a1277e618135872e1e3ff30ddad0b7"
E9_VERDICT_PATH = "results/e9_3b_evidence/verdict.json"
E9_VERDICT_SHA256 = "a025a56b541423610659ea52e40414fd5cad7454bf4687aabf4991a21e9d90b8"

CANDIDATE_CASE_SHA256 = "727ab78b1ae1d50e1fc16edf85e2dfda13d452c846dbcdce1d612b7a9603ea63"
CANDIDATE_CASE_ID_SHA256 = "f1b1fbc4cf575cb2d3232ac1b8f534aeaac52379521cbe20154bbbe7dad7597c"
CANDIDATE_PROMPT_SHA256 = "82cf0827ed57c988fde30b86b7b280663c036cee57d276b701d16583d9dcf005"
CANDIDATE_TARGET_ID_SHA256 = "479c6cbac4d7b5cf88ebf7ccdd6daea2e680ecb264872755f2f4508f26328442"
CANDIDATE_TARGET_SHA256 = "b15f77742627741db8f245214ecbead2a54ca449610a5b358f1dedc3e0238559"
WHITENING_PROMPT_SHA256 = "be7bf125fee1dcf350af6710a6949d77f937c2272de90e8cfd907b08d966fa42"
E9_QUERY_PROMPT_SHA256 = "7f9c199bb1a402fa95bd46ffa879d01a7e36acb3f9921eb2ac2043b3343ff07a"

SEED = 0
PREFLIGHT_SCHEMA = "hlm5-e10-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e10-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e10-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e10-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e10-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e10-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57",
    "hlm5/certify.py": "a13544eb242cb1f827496a46a6d943723ad51f35b827fe47d2da641e26a1c210",
    "hlm5/e7_contract.py": "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d",
    "hlm5/e7_runtime.py": "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc",
    "hlm5/e7b_runtime.py": "bf6e7b0abfe83d0da0f146ae72dc69d52b0469fcc311689c535a7628ea2fdbc2",
    "hlm5/e7b_contract.py": "606d95ca4972d1febadf6348c441449574f1c33eb2f3b7b71b3f79df27c31119",
    "hlm5/e7c_runtime.py": "55b7daf0d1d65303db5a3331ddc9b3db701e512ea45f2823a2dbfece35a58d23",
    "hlm5/e8_contract.py": "633106728f658ce469c22e7f3d51fc013b38b90af24f5a506c6deb721166fd80",
    "hlm5/e8_runtime.py": "dc78a1d7bfea929a66b01d69cc0fdf9b7f7634643d918c6d182c8854fb205be4",
    "hlm5/e9_runtime.py": "7c805950e3eb0618953c4e24d64629c5b427e774a2395d525384a6425643b3f7",
    "hlm5/e9_contract.py": "bf345b9e09ee60672d94a582f2dcea60343a5889bd6410f60af81023cffc5fc7",
    "hlm5/memory.py": "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973",
    "hlm5/public_adapter.py": "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d",
}

REGISTERED_SOURCE_PATHS = (
    "hlm5/e10_contract.py",
    "hlm5/e10_runtime.py",
    "scripts/preflight_e10_3b.py",
    "scripts/register_e10_3b.py",
    "scripts/admit_e10_3b.py",
    "scripts/replay_e10_3b.py",
    "scripts/verify_e10_3b.py",
    "tests/test_e10_3b.py",
)
REGISTERED_COMMANDS = tuple(
    f"python scripts/{name}_e10_3b.py"
    for name in ("preflight", "register", "admit", "replay", "verify")
)

RESULTS_DIR = REPO_ROOT / "results"
STAGING_DIR = RESULTS_DIR / "e10_3b_staging"
PREFLIGHT_PATH = STAGING_DIR / "preflight.json"
EXECUTION_RECEIPT_PATH = STAGING_DIR / "execution_receipt.json"
ATTEMPT_PATH = STAGING_DIR / "attempt.json"
BUNDLE_PATH = STAGING_DIR / "adapter_1024.pt"
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


def _require_staging_path(path: Path) -> Path:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"E10 output must stay under {staging}: {resolved}")
    return resolved


def atomic_create_json(path: Path, value: Any) -> None:
    resolved = _require_staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{resolved.name}.", dir=resolved.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, resolved)
        except FileExistsError as error:
            raise FileExistsError(f"E10 output already exists: {resolved}") from error
    finally:
        temp_path.unlink(missing_ok=True)


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E10 outputs already exist: {present}")


def scientific_sha256(value: dict[str, Any]) -> str:
    return stable_json_sha256(value)


def verify_scientific_hash(payload: dict[str, Any], label: str) -> None:
    scientific = payload.get("scientific")
    if not isinstance(scientific, dict):
        raise RuntimeError(f"{label} has no scientific object")
    if payload.get("scientific_sha256") != scientific_sha256(scientific):
        raise RuntimeError(f"{label} scientific hash mismatch")


def verify_protocol() -> None:
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("registered E10 protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E10 protocol")


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E10 dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def verify_bound_inputs() -> dict[str, Any]:
    values = {
        COUNTERFACT_PATH: COUNTERFACT_SHA256,
        E8_RESULT_PATH: E8_RESULT_SHA256,
        E9_ADMISSION_PATH: E9_ADMISSION_SHA256,
        E9_RESULT_PATH: E9_RESULT_SHA256,
        E9_VERDICT_PATH: E9_VERDICT_SHA256,
    }
    for relative, expected in values.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E10 bound input {relative}")
    e9_verdict = load_json_object(REPO_ROOT / E9_VERDICT_PATH)
    if e9_verdict.get("verdict") != "PASS_3B_WIDE_LOCALITY":
        raise RuntimeError("bound E9 verdict is not PASS_3B_WIDE_LOCALITY")
    return {
        "counterfact_sha256": COUNTERFACT_SHA256,
        "e8_result_sha256": E8_RESULT_SHA256,
        "e9_admission_sha256": E9_ADMISSION_SHA256,
        "e9_result_sha256": E9_RESULT_SHA256,
        "e9_verdict_sha256": E9_VERDICT_SHA256,
        "e9_verdict": "PASS_3B_WIDE_LOCALITY",
    }


def selected_population(tokenizer: Any) -> dict[str, Any]:
    e8 = load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"]["measurement"]
    e9 = selected_e9_population()
    excluded = list(e8["prompt_order"]) + list(e9["prompt_order"])
    cases = load_capacity_cases(REPO_ROOT / COUNTERFACT_PATH, tokenizer, excluded)
    record = population_record(cases)
    expected = {
        "candidate_count": CANDIDATE_COUNT,
        "first_case_id": CANDIDATE_START,
        "last_case_id": 21_318,
        "case_sha256": CANDIDATE_CASE_SHA256,
        "case_id_sha256": CANDIDATE_CASE_ID_SHA256,
        "prompt_sha256": CANDIDATE_PROMPT_SHA256,
        "target_id_sha256": CANDIDATE_TARGET_ID_SHA256,
        "target_sha256": CANDIDATE_TARGET_SHA256,
    }
    if record != expected:
        raise RuntimeError(f"E10 population mismatch: {record}")
    whitening = list(e8["prompt_order"][-18:])
    if stable_json_sha256(whitening) != WHITENING_PROMPT_SHA256:
        raise RuntimeError("E10 whitening prompt hash mismatch")
    if stable_json_sha256(e9["prompt_order"]) != E9_QUERY_PROMPT_SHA256:
        raise RuntimeError("E10 locality prompt hash mismatch")
    if set(row["prompt"] for row in cases) & set(excluded):
        raise RuntimeError("E10 population overlaps prior prompts")
    return {
        "cases": cases,
        "record": record,
        "prompt_order": [row["prompt"] for row in cases],
        "whitening_prompts": whitening,
        "query_rows": e9["query_rows"],
        "query_prompt_order": e9["prompt_order"],
    }


def registered_source_sha256(*, require_clean: bool = False) -> dict[str, str]:
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E10 sources must be committed:\n" + dirty)
    missing = [p for p in REGISTERED_SOURCE_PATHS if not (REPO_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"registered E10 sources missing: {missing}")
    return {p: sha256_file(REPO_ROOT / p) for p in REGISTERED_SOURCE_PATHS}


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("source_sha256") != registered_source_sha256():
        raise RuntimeError("registered E10 source drift")
    if receipt.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E10 dependency drift")


def environment_failures(value: Any, *, expected_node: str | None = None) -> list[str]:
    if not isinstance(value, dict):
        return ["E10 environment missing"]
    failures = [
        f"E10 environment mismatch: {name}"
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if value.get(name) != expected
    ]
    node = value.get("node")
    if not isinstance(node, str) or not node:
        failures.append("E10 physical node missing")
    elif expected_node is not None and node != expected_node:
        failures.append("E10 physical node differs from preflight")
    if set(value) != set(EXPECTED_ENVIRONMENT) | {"node"}:
        failures.append("E10 environment fields are not exhaustive")
    return failures


def _verify_commit(commit: Any) -> None:
    if not isinstance(commit, str) or git_output("rev-parse", commit) != commit:
        raise RuntimeError("registered E10 implementation commit unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E10 preflight")
    payload = load_json_object(PREFLIGHT_PATH)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "candidate_record": {
            "candidate_count": CANDIDATE_COUNT,
            "first_case_id": CANDIDATE_START,
            "last_case_id": 21_318,
            "case_sha256": CANDIDATE_CASE_SHA256,
            "case_id_sha256": CANDIDATE_CASE_ID_SHA256,
            "prompt_sha256": CANDIDATE_PROMPT_SHA256,
            "target_id_sha256": CANDIDATE_TARGET_ID_SHA256,
            "target_sha256": CANDIDATE_TARGET_SHA256,
        },
        "capacity_tiers": list(CAPACITY_TIERS),
        "memory_size": MEMORY_SIZE,
        "model_forward_batch": MODEL_FORWARD_BATCH,
        "gate_batch": GATE_BATCH,
        "commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e10_3b.py -q",
        "tests_returncode": 0,
        "e10_outcomes_computed": False,
        "bound_inputs": verify_bound_inputs(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E10 preflight field mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E10 model invariant mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E10 arithmetic mismatch")
    failures = environment_failures(payload.get("environment"))
    if failures:
        raise RuntimeError("invalid E10 environment: " + "; ".join(failures))
    e8_basis = load_json_object(REPO_ROOT / E8_RESULT_PATH)["scientific"]["basis_and_directions"]
    if payload.get("basis_and_directions") != e8_basis:
        raise RuntimeError("E10 basis differs from E8")
    _verify_commit(payload.get("implementation_commit"))
    _verify_source_receipt(payload)
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
        "bound_inputs": preflight["bound_inputs"],
        "basis_and_directions": preflight["basis_and_directions"],
        "environment": preflight["environment"],
        "native_runtime": preflight["native_runtime"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e10_3b.py -q",
        "tests_returncode": 0,
        "e10_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E10 execution receipt mismatch: {name}")
    _verify_source_receipt(payload)
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
        "candidate_case_sha256": CANDIDATE_CASE_SHA256,
        "pre_admission_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E10 attempt mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def measurement_validity_failures(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return ["E10 measurement missing"]
    failures: list[str] = []
    direct = measurement.get("direct", {})
    rows = direct.get("direct_rows")
    summary = direct.get("direct_summary", {})
    if not isinstance(rows, list) or len(rows) != CANDIDATE_COUNT:
        return ["E10 direct row denominator is not 1200"]
    for index, row in enumerate(rows):
        prefix = f"E10 direct row {index}"
        if row.get("case_index") != index:
            failures.append(prefix + " case index mismatch")
        eligible = row.get("baseline_correct") is False
        if row.get("eligible") is not eligible:
            failures.append(prefix + " eligibility mismatch")
        admitted_expected = bool(eligible and row.get("decision") == "ADMIT_NATIVE")
        if row.get("deployment_admitted") is not admitted_expected:
            failures.append(prefix + " deployment decision mismatch")
        for name in ("baseline_logits_sha256",):
            if not _valid_hash(row.get(name)):
                failures.append(prefix + f" missing {name}")
        if row.get("native_checked") is True:
            if not _valid_hash(row.get("native_logits_sha256")):
                failures.append(prefix + " missing native logits hash")
            prediction = row.get("native_prediction")
            target_id = row.get("target_id")
            margin = row.get("native_margin")
            if row.get("decision") == "ADMIT_NATIVE" and not (
                prediction == target_id
                and isinstance(margin, (int, float))
                and float(margin) > 0.0
            ):
                failures.append(prefix + " native admission is not a strict win")
    admitted = [i for i, row in enumerate(rows) if row.get("deployment_admitted")]
    selected = admitted[: CAPACITY_TIERS[-1]]
    recomputed_direct = direct_summary(rows)
    for name, value in recomputed_direct.items():
        if summary.get(name) != value:
            failures.append(f"E10 direct summary mismatch: {name}")
    if summary.get("selected_indices") != selected:
        failures.append("E10 did not select the first admitted prefix")
    if summary.get("selected_count") != len(selected):
        failures.append("E10 selected count mismatch")
    if summary.get("selected_indices_sha256") != stable_json_sha256(selected):
        failures.append("E10 selected-index hash mismatch")
    selected_case_ids = [int(rows[index]["case_id"]) for index in selected]
    if (
        summary.get("selected_case_ids") != selected_case_ids
        or summary.get("selected_case_ids_sha256")
        != stable_json_sha256(selected_case_ids)
    ):
        failures.append("E10 selected-case identity mismatch")
    key_record = measurement.get("key_whitening", {})
    if (
        key_record.get("exact_key_count") != CANDIDATE_COUNT
        or key_record.get("whitening_prompt_count") != 18
        or key_record.get("floor_fraction") != 0.01
        or not _valid_hash(key_record.get("mean_sha256"))
        or not _valid_hash(key_record.get("transform_sha256"))
    ):
        failures.append("E10 shared key operator receipt mismatch")
    tiers = measurement.get("tiers", {})
    expected_tiers = [tier for tier in CAPACITY_TIERS if len(selected) >= tier]
    if sorted(int(key) for key in tiers) != expected_tiers:
        failures.append("E10 constructible tier set mismatch")
    for tier in expected_tiers:
        item = tiers.get(str(tier), {})
        memory = item.get("memory_receipt", {})
        if (
            memory.get("memory_size") != MEMORY_SIZE
            or memory.get("active_count") != tier
            or memory.get("active_slots") != list(range(tier))
            or memory.get("selected_candidate_indices") != selected[:tier]
            or memory.get("key_mean_sha256") != key_record.get("mean_sha256")
            or memory.get("key_transform_sha256") != key_record.get("transform_sha256")
        ):
            failures.append(f"E10 tier {tier} memory receipt mismatch")
        positive = item.get("positive_scan", {})
        positive_rows = positive.get("rows")
        if not isinstance(positive_rows, list) or len(positive_rows) != tier:
            failures.append(f"E10 tier {tier} positive denominator mismatch")
        else:
            expected_summary = {
                "positive_count": tier,
                "gate_open_count": sum(row.get("gate_open") is True for row in positive_rows),
                "own_slot_count": sum(row.get("selected_slot") == row.get("own_slot") for row in positive_rows),
                "strict_target_success_count": sum(row.get("strict_target_success") is True for row in positive_rows),
                "nonzero_delta_count": sum(row.get("delta_zero") is False for row in positive_rows),
            }
            for name, value in expected_summary.items():
                if positive.get("summary", {}).get(name) != value:
                    failures.append(f"E10 tier {tier} positive summary mismatch: {name}")
            for index, row in enumerate(positive_rows):
                score = row.get("selected_score")
                gate_expected = bool(
                    isinstance(score, (int, float))
                    and math.isfinite(float(score))
                    and float(score) >= GATE_THRESHOLD
                )
                strict_expected = bool(
                    row.get("prediction") == row.get("target_id")
                    and isinstance(row.get("target_margin"), (int, float))
                    and float(row["target_margin"]) > 0.0
                )
                if (
                    row.get("candidate_index") != selected[index]
                    or row.get("own_slot") != index
                    or row.get("gate_open") is not gate_expected
                    or row.get("strict_target_success") is not strict_expected
                ):
                    failures.append(f"E10 tier {tier} positive row {index} mismatch")
                if gate_expected and (
                    row.get("delta_zero") is not False
                    or row.get("native_hidden_sha256")
                    == row.get("adapted_hidden_sha256")
                ):
                    failures.append(f"E10 tier {tier} open positive is vacuous")
                for name in (
                    "prompt_sha256", "native_hidden_sha256", "delta_sha256",
                    "adapted_hidden_sha256", "logits_sha256",
                ):
                    if not _valid_hash(row.get(name)):
                        failures.append(
                            f"E10 tier {tier} positive row {index} missing {name}"
                        )
            values = positive.get("summary", {})
            attentions = [float(row["own_slot_attention"]) for row in positive_rows]
            ordered = sorted(attentions)
            median = (
                ordered[len(ordered) // 2]
                if len(ordered) % 2
                else 0.5 * (
                    ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]
                )
            )
            if values.get("minimum_own_slot_attention") != min(attentions):
                failures.append(f"E10 tier {tier} minimum attention mismatch")
            if values.get("median_own_slot_attention") != median:
                failures.append(f"E10 tier {tier} median attention mismatch")
            if values.get("minimum_target_margin") != min(
                float(row["target_margin"]) for row in positive_rows
            ):
                failures.append(f"E10 tier {tier} minimum margin mismatch")
            nonown = [
                float(row["strongest_nonown_score"])
                for row in positive_rows
                if row.get("strongest_nonown_score") is not None
            ]
            if values.get("maximum_nonown_score") != (
                max(nonown) if nonown else None
            ):
                failures.append(f"E10 tier {tier} maximum non-own score mismatch")
            coherence = values.get("key_coherence")
            crosstalk = values.get("degree_five_max_crosstalk")
            if (
                not isinstance(coherence, (int, float))
                or not math.isfinite(float(coherence))
                or not isinstance(crosstalk, (int, float))
                or not math.isfinite(float(crosstalk))
                or float(crosstalk) != float(coherence) ** 5
            ):
                failures.append(f"E10 tier {tier} coherence/crosstalk mismatch")
        locality = item.get("locality_scan", {})
        locality_rows = locality.get("rows")
        if not isinstance(locality_rows, list) or len(locality_rows) != 12_288:
            failures.append(f"E10 tier {tier} locality denominator mismatch")
        else:
            try:
                expected_locality = summarize_locality(locality_rows)
            except Exception as error:
                failures.append(f"E10 tier {tier} locality summary exception: {error}")
            else:
                if locality.get("summary") != expected_locality:
                    failures.append(f"E10 tier {tier} locality summary mismatch")
            for row in locality_rows:
                score = row.get("selected_score")
                expected_gate = bool(
                    isinstance(score, (int, float))
                    and math.isfinite(float(score))
                    and float(score) >= GATE_THRESHOLD
                )
                if row.get("gate_open") is not expected_gate:
                    failures.append(f"E10 tier {tier} locality gate mismatch")
                    break
                slot = row.get("selected_slot")
                if (
                    not isinstance(slot, int)
                    or not 0 <= slot < tier
                    or row.get("selected_candidate_index") != selected[slot]
                ):
                    failures.append(f"E10 tier {tier} locality slot mapping mismatch")
                    break
                if not expected_gate and not (
                    row.get("delta_zero") is True
                    and row.get("hidden_bit_identical") is True
                    and row.get("native_hidden_sha256") == row.get("adapted_hidden_sha256")
                ):
                    failures.append(f"E10 tier {tier} closed locality gate changed hidden")
                    break
        rollback = item.get("rollback", {})
        roll = rollback.get("summary", {})
        if (
            roll.get("positive_count") != tier
            or roll.get("positive_delta_zero_count") != tier
            or roll.get("positive_hidden_bit_identical_count") != tier
            or roll.get("locality_count") != 12_288
            or roll.get("locality_delta_zero_count") != 12_288
            or roll.get("locality_hidden_bit_identical_count") != 12_288
            or rollback.get("post_active_count") != 0
            or any(label is not None for label in rollback.get("post_labels", []))
        ):
            failures.append(f"E10 tier {tier} rollback mismatch")
    bundle = measurement.get("bundle")
    if len(selected) >= 1024:
        tensor_record = {} if not isinstance(bundle, dict) else bundle.get("tensors", {})
        expected_shapes = {
            "keys": [1024, 2048], "values": [1024, 2048],
            "alphas": [1024], "key_mean": [2048],
            "key_transform": [2048, 2048], "selected_indices": [1024],
            "selected_case_ids": [1024],
        }
        if not isinstance(bundle, dict) or not _valid_hash(bundle.get("file_sha256")):
            failures.append("E10 constructible bundle receipt missing")
        elif set(tensor_record) != set(expected_shapes):
            failures.append("E10 bundle tensor members mismatch")
        else:
            for name, shape in expected_shapes.items():
                value = tensor_record[name]
                if value.get("shape") != shape or not _valid_hash(value.get("sha256")):
                    failures.append(f"E10 bundle tensor receipt mismatch: {name}")
            if bundle.get("active_labels") != memory.get("active_labels"):
                failures.append("E10 bundle labels differ from 1024 memory")
    elif bundle is not None:
        failures.append("E10 emitted a bundle without 1024 admitted keys")
    return failures


def admission_validity_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        failures.append("E10 admission schema or status mismatch")
    try:
        verify_scientific_hash(payload, "E10 admission")
    except Exception as error:
        return failures + [str(error)]
    scientific = payload["scientific"]
    if scientific.get("model_invariants") != expected_model_invariants():
        failures.append("E10 admission model invariants mismatch")
    failures.extend(measurement_validity_failures(scientific.get("measurement")))
    return failures


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
            raise RuntimeError(f"E10 admission provenance mismatch: {name}")
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("invalid E10 admission: " + "; ".join(failures))
    bundle = payload["scientific"]["measurement"].get("bundle")
    if bundle is not None:
        require_file_hash(BUNDLE_PATH, bundle["file_sha256"], "E10 tensor bundle")
    return payload, sha256_file(ADMISSION_PATH)


def result_validity_failures(result: dict[str, Any], admission: dict[str, Any]) -> list[str]:
    try:
        verify_scientific_hash(result, "E10 result")
    except Exception as error:
        return [str(error)]
    scientific = result["scientific"]
    failures: list[str] = []
    if scientific.get("model_invariants") != admission["scientific"].get("model_invariants"):
        failures.append("E10 replay model invariants differ from admission")
    if scientific.get("admission_scientific_sha256") != admission.get("scientific_sha256"):
        failures.append("E10 replay binds a different admission")
    if scientific.get("measurement") != admission["scientific"].get("measurement"):
        failures.append("E10 replay measurement differs from admission")
    if scientific.get("measurement_exact_match") is not True:
        failures.append("E10 exact replay flag failed")
    if scientific.get("bundle_exact_match") is not True:
        failures.append("E10 bundle replay flag failed")
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
            raise RuntimeError(f"E10 result provenance mismatch: {name}")
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E10 result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


def tier_scientific_failures(tier: int, item: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    positive = item["positive_scan"]["summary"]
    locality = item["locality_scan"]["summary"]
    rollback = item["rollback"]["summary"]
    for name in ("gate_open_count", "own_slot_count", "strict_target_success_count", "nonzero_delta_count"):
        if positive.get(name) != tier:
            failures.append(f"tier {tier} positive bar failed: {name}")
    if (
        locality.get("query_count") != 12_288
        or locality.get("gate_open_count") != 0
        or locality.get("delta_zero_count") != 12_288
        or locality.get("hidden_bit_identical_count") != 12_288
    ):
        failures.append(f"tier {tier} locality bar failed")
    for kind in ("exact", "paraphrase", "neighborhood"):
        value = locality.get("by_kind", {}).get(kind, {})
        if value.get("query_count") != 4096 or value.get("gate_open_count") != 0:
            failures.append(f"tier {tier} {kind} locality bar failed")
    if (
        rollback.get("positive_delta_zero_count") != tier
        or rollback.get("positive_hidden_bit_identical_count") != tier
        or rollback.get("locality_delta_zero_count") != 12_288
        or rollback.get("locality_hidden_bit_identical_count") != 12_288
    ):
        failures.append(f"tier {tier} rollback bar failed")
    return failures


def scientific_failures(result: dict[str, Any]) -> list[str]:
    measurement = result["scientific"]["measurement"]
    failures: list[str] = []
    if measurement["direct"]["direct_summary"].get("selected_count", 0) < 1024:
        failures.append("fewer than 1024 candidates were deployment-admitted")
    tiers = measurement.get("tiers", {})
    for tier in CAPACITY_TIERS:
        if str(tier) not in tiers:
            failures.append(f"tier {tier} was not constructible")
        else:
            failures.extend(tier_scientific_failures(tier, tiers[str(tier)]))
    if measurement.get("bundle") is None:
        failures.append("portable 1024-slot bundle is missing")
    return failures


def largest_passing_tier(result: dict[str, Any] | None) -> int | None:
    if result is None:
        return None
    tiers = result.get("scientific", {}).get("measurement", {}).get("tiers", {})
    passing = [
        tier for tier in CAPACITY_TIERS
        if str(tier) in tiers and not tier_scientific_failures(tier, tiers[str(tier)])
    ]
    return max(passing) if passing else None


def verdict_scientific_payload(
    *, verdict: str, result: dict[str, Any] | None,
    validity_failures: list[str], scientific_failures_value: list[str],
) -> dict[str, Any]:
    measurement = None if result is None else result.get("scientific", {}).get("measurement")
    return {
        "verdict": verdict,
        "direct_summary": None if measurement is None else measurement["direct"]["direct_summary"],
        "tier_summaries": None if measurement is None else {
            tier: {
                "positive": value["positive_scan"]["summary"],
                "locality": value["locality_scan"]["summary"],
                "rollback": value["rollback"]["summary"],
            }
            for tier, value in measurement.get("tiers", {}).items()
        },
        "largest_passing_tier": largest_passing_tier(result),
        "bundle": None if measurement is None else measurement.get("bundle"),
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Pinned frozen SmolLM3-3B-Base, exact single-token CounterFact keys, "
            "registered A100/BF16 node/runtime, nested 64/256/1024 active slots, "
            "and the finite spent E9 locality population."
        ),
    }


__all__ = [
    "ADMISSION_PATH", "ADMISSION_SCHEMA", "ATTEMPT_PATH", "ATTEMPT_SCHEMA",
    "BUNDLE_PATH", "CANDIDATE_CASE_SHA256", "DEPENDENCY_SHA256",
    "EXECUTION_RECEIPT_PATH", "EXECUTION_SCHEMA", "NAMED_OUTPUTS",
    "PREFLIGHT_PATH", "PREFLIGHT_SCHEMA", "PROTOCOL_COMMIT", "PROTOCOL_SHA256",
    "REGISTERED_COMMANDS", "RESULT_PATH", "RESULT_SCHEMA", "SEED",
    "VERDICT_PATH", "VERDICT_SCHEMA", "admission_validity_failures",
    "assert_outputs_absent", "atomic_create_json", "environment_failures",
    "git_output", "largest_passing_tier", "load_admission", "load_attempt",
    "load_execution_receipt", "load_json_object", "load_preflight", "load_result",
    "measurement_validity_failures", "registered_source_sha256", "scientific_failures",
    "scientific_sha256", "selected_population", "verdict_scientific_payload",
    "verify_bound_inputs", "verify_dependencies", "verify_protocol",
]
