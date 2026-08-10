"""Fail-closed provenance and verdict contract for the E8 full-path gate."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from .e7_contract import (
    HEAD_IDENTITY_MAX_ABS,
    MODEL_FILE_SHA256,
    MODEL_FORWARD_DTYPE,
    MODEL_HIDDEN_SIZE,
    MODEL_ID,
    MODEL_LAYERS,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    MODEL_VOCAB_SIZE,
    REPO_ROOT,
    RESULTS_DIR,
    git_output,
    registered_samples,
    require_file_hash,
    sha256_file,
    snapshot_path,
    stable_json,
    stable_json_sha256,
    verify_snapshot,
)
from .e7b_contract import (
    BASELINE_ANCHOR_SHA256,
    EXPECTED_NATIVE_RUNTIME,
)
from .e7b_runtime import (
    GEOMETRIC_REACHABLE,
    NATIVE_ADMIT,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    native_decision,
)
from .e8_runtime import (
    CASE_COUNT,
    CASE_START,
    DEGREE,
    GATE_THRESHOLD,
    HEAD_BATCH,
    MEMORY_SIZE,
    TEMPERATURE,
    direct_summary,
    gate_specs,
    load_selected_cases,
    prompt_order,
)


PROTOCOL_COMMIT = "02f5186fefa88e9488f0635a5276ce0a9be5999d"
PROTOCOL_PATH = "docs/e8-fresh-full-path-zca-protocol-2026-08-10.md"
PROTOCOL_SHA256 = "10138337944a3fe709f3dc6f8d1949435fe425129ed9efbe8f820eb14e28e930"
E7C_RESULT_PATH = "results/e7c_3b_evidence/result.json"
E7C_RESULT_SHA256 = "88c9819648ce7a30ba3e6d75aa5da2c317d072db8eae0160631dada212f9cf44"
E7C_VERDICT_PATH = "results/e7c_3b_evidence/verdict.json"
E7C_VERDICT_SHA256 = "32cc956a9a56e0be56e0976c4c17e7862544a2ede07735ac45ea6faffba0f1f6"
E7C_IMPLEMENTATION_COMMIT = "58cedff87f270f1e7047e491a80565b7da05390f"
COUNTERFACT_PATH = "scripts/data/counterfact.json"
COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
CASE_SHA256 = "6d81f2ccf551b76402213593aea5b12ad37b8eb556fe3ae2a5731bd6cf5cb758"
CASE_IDS = [
    *range(20000, 20049),
    20050,
    *range(20052, 20066),
]
PROMPT_COUNT = 337
PROMPT_SHA256 = "7652c5c71736fc3c0b33845664638aead1c09b2ff84d128618c36bcd20d0db00"
NEUTRAL_SHA256 = "c535b5a8eeeae85b0918bee646c82fad78976797f6df6a9bfb939512583e1d69"
WHITEN_SHA256 = "be7bf125fee1dcf350af6710a6949d77f937c2272de90e8cfd907b08d966fa42"
SEED = 0
MIN_ELIGIBLE = 48
MIN_ADMISSION_NUMERATOR = 19
MIN_ADMISSION_DENOMINATOR = 20
MAX_RAW_ADMITTED_LOSS = 1

EXPECTED_BASIS = {
    "mean_sha256": "ae2b4c5d0c3aa902b7ae61dfb320177fea40270c1450f15a9fb5c064bed9e489",
    "eigenvalues_sha256": "7f33aaca070a3539f8c4c65a0e8a56ef82a2813d950ec910335160f971da4971",
    "eigenvectors_sha256": "d5e1b3f6c5fafa4b9fd46bfbcbf1eddd982a2b19ea9a40542ba236a8ebdb3fad",
    "operator_sha256": "ac2536bf5a28014c665177fb89ff907ae2456a276c5c096edc767900b4a8d065",
    "median_positive_eigenvalue": 0.007535659708082676,
    "ridge_fraction": 0.01,
    "ridge": 0.00007535659708082676,
}
EXPECTED_ENVIRONMENT = {
    "cuda_runtime": "12.1",
    "cudnn": 90100,
    "device_capability": [8, 0],
    "device_count": 1,
    "device_name": "NVIDIA A100-SXM-64GB",
    "device_type": "cuda",
    "platform": "Linux-4.18.0-477.27.1.el8_8.x86_64-x86_64-with-glibc2.28",
    "python": "3.11.6",
    "torch": "2.5.1+cu121",
    "transformers": "5.12.1",
}

PREFLIGHT_SCHEMA = "hlm5-e8-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e8-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e8-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e8-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e8-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e8-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57",
    "hlm5/memory.py": "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973",
    "hlm5/model.py": "2f810bde52bbf3e0da9f52640bf36ca924cd665bd6a7036155f0581494b8857c",
    "hlm5/io.py": "5886ce8331ba42ca0ee75dd67746665be58d174e5f3416ad661f81c7d7685439",
    "hlm5/e7_contract.py": "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d",
    "hlm5/e7_runtime.py": "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc",
    "hlm5/e7b_contract.py": "606d95ca4972d1febadf6348c441449574f1c33eb2f3b7b71b3f79df27c31119",
    "hlm5/e7b_runtime.py": "bf6e7b0abfe83d0da0f146ae72dc69d52b0469fcc311689c535a7628ea2fdbc2",
    "hlm5/e7c_runtime.py": "55b7daf0d1d65303db5a3331ddc9b3db701e512ea45f2823a2dbfece35a58d23",
    "hlm5/certify.py": "a13544eb242cb1f827496a46a6d943723ad51f35b827fe47d2da641e26a1c210",
    "hlm5/edit_audit.py": "930d51619bb240e13a15412f6db92b0cf58a87e17a405dc7110bbaaa2e8e8930",
    "hlm5/key_value.py": "a0bacdfcba80397bfacefa7c7762bcb202fb91363da0e6fa902f9c707c70140a",
    "hlm5/public_adapter.py": "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d",
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/e8_contract.py",
    "hlm5/e8_runtime.py",
    "scripts/preflight_e8_3b.py",
    "scripts/register_e8_3b.py",
    "scripts/admit_e8_3b.py",
    "scripts/replay_e8_3b.py",
    "scripts/verify_e8_3b.py",
    "tests/test_e8_3b.py",
)
REGISTERED_COMMANDS = (
    "python scripts/preflight_e8_3b.py",
    "python scripts/register_e8_3b.py",
    "python scripts/admit_e8_3b.py",
    "python scripts/replay_e8_3b.py",
    "python scripts/verify_e8_3b.py",
)

STAGING_DIR = RESULTS_DIR / "e8_3b_staging"
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
        raise ValueError(f"E8 output must stay under {staging}: {resolved}")
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
            raise FileExistsError(f"E8 output already exists: {resolved}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink()


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E8 outputs already exist: {present}")


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
        raise RuntimeError("registered E8 protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E8 protocol")


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E8 dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def verify_bound_inputs() -> dict[str, Any]:
    require_file_hash(REPO_ROOT / E7C_RESULT_PATH, E7C_RESULT_SHA256, "E7c result")
    require_file_hash(REPO_ROOT / E7C_VERDICT_PATH, E7C_VERDICT_SHA256, "E7c verdict")
    require_file_hash(
        REPO_ROOT / COUNTERFACT_PATH, COUNTERFACT_SHA256, "CounterFact data"
    )
    result = load_json_object(REPO_ROOT / E7C_RESULT_PATH)
    verdict = load_json_object(REPO_ROOT / E7C_VERDICT_PATH)
    if verdict.get("verdict") != "PASS_3B_ZCA_GEOMETRY":
        raise RuntimeError("bound E7c verdict is not PASS_3B_ZCA_GEOMETRY")
    if result.get("status") != "COMPLETE":
        raise RuntimeError("bound E7c result is not complete")
    basis = result.get("scientific", {}).get("basis_and_directions", {})
    for name, value in EXPECTED_BASIS.items():
        if basis.get(name) != value:
            raise RuntimeError(f"bound E7c basis mismatch: {name}")
    return {
        "e7c_implementation_commit": E7C_IMPLEMENTATION_COMMIT,
        "e7c_result_sha256": E7C_RESULT_SHA256,
        "e7c_verdict_sha256": E7C_VERDICT_SHA256,
        "e7c_verdict": "PASS_3B_ZCA_GEOMETRY",
        "counterfact_sha256": COUNTERFACT_SHA256,
    }


def selected_population(tokenizer: Any) -> dict[str, Any]:
    samples = registered_samples()
    cases = load_selected_cases(REPO_ROOT / COUNTERFACT_PATH, tokenizer)
    neutral = list(samples["NEUTRAL"])
    whitening = list(samples["WHITEN_TEXT"])
    prompts = prompt_order(cases, neutral, whitening)
    if [case["case_id"] for case in cases] != CASE_IDS:
        raise RuntimeError("E8 selected case ids differ from registration")
    if stable_json_sha256(cases) != CASE_SHA256:
        raise RuntimeError("E8 selected case hash mismatch")
    if stable_json_sha256(neutral) != NEUTRAL_SHA256:
        raise RuntimeError("E8 neutral-list hash mismatch")
    if stable_json_sha256(whitening) != WHITEN_SHA256:
        raise RuntimeError("E8 whitening-list hash mismatch")
    if len(prompts) != PROMPT_COUNT or stable_json_sha256(prompts) != PROMPT_SHA256:
        raise RuntimeError("E8 prompt-order contract mismatch")
    return {
        "cases": cases,
        "neutral_prompts": neutral,
        "whitening_prompts": whitening,
        "prompt_order": prompts,
    }


def expected_model_invariants() -> dict[str, Any]:
    return {
        "class_name": "SmolLM3ForCausalLM",
        "parameter_count": MODEL_PARAMETER_COUNT,
        "hidden_size": MODEL_HIDDEN_SIZE,
        "vocab_size": MODEL_VOCAB_SIZE,
        "layers": MODEL_LAYERS,
        "tied_embeddings": True,
        "head_bias_is_none": True,
        "all_parameters_frozen": True,
        "floating_parameter_dtypes": [MODEL_FORWARD_DTYPE],
    }


def registered_source_sha256(*, require_clean: bool = False) -> dict[str, str]:
    missing = [p for p in REGISTERED_SOURCE_PATHS if not (REPO_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"registered E8 sources missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E8 sources must be committed:\n" + dirty)
    return {p: sha256_file(REPO_ROOT / p) for p in REGISTERED_SOURCE_PATHS}


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("source_sha256") != registered_source_sha256():
        raise RuntimeError("registered E8 source drift")
    if receipt.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E8 dependency receipt mismatch")


def _verify_implementation_ancestry(commit: Any) -> None:
    if not isinstance(commit, str) or git_output("rev-parse", commit) != commit:
        raise RuntimeError("registered E8 implementation commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _verify_basis(record: Any) -> None:
    if not isinstance(record, dict):
        raise RuntimeError("missing E8 basis record")
    for name, value in EXPECTED_BASIS.items():
        if record.get(name) != value:
            raise RuntimeError(f"E8 basis mismatch: {name}")
    if record.get("operator_dtype") != "float64" or record.get("operator_shape") != [
        MODEL_HIDDEN_SIZE,
        MODEL_HIDDEN_SIZE,
    ]:
        raise RuntimeError("E8 operator tensor contract mismatch")
    for name in ("raw_directions_sha256", "zca_directions_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get(name))) is None:
            raise RuntimeError(f"E8 basis record lacks {name}")
    if record.get("direction_dtype") != "float64" or record.get("direction_shape") != [
        CASE_COUNT,
        MODEL_HIDDEN_SIZE,
    ]:
        raise RuntimeError("E8 direction tensor contract mismatch")


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E8 preflight")
    payload = load_json_object(PREFLIGHT_PATH)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "case_start": CASE_START,
        "case_count": CASE_COUNT,
        "case_ids": CASE_IDS,
        "case_sha256": CASE_SHA256,
        "prompt_count": PROMPT_COUNT,
        "prompt_sha256": PROMPT_SHA256,
        "neutral_sha256": NEUTRAL_SHA256,
        "whitening_sha256": WHITEN_SHA256,
        "bound_inputs": verify_bound_inputs(),
        "fresh_prompt_outcomes_computed": False,
        "raw_candidate_equal": False,
        "commands": list(REGISTERED_COMMANDS),
        "preflight_test_command": "python -m pytest tests/test_e8_3b.py -q",
        "preflight_tests_returncode": 0,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E8 preflight field mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E8 preflight model invariants mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E8 preflight native arithmetic mismatch")
    if payload.get("environment") != EXPECTED_ENVIRONMENT:
        raise RuntimeError("E8 preflight environment mismatch")
    identity = payload.get("baseline_only_head_identity", {})
    if (
        identity.get("prompt_sha256") != BASELINE_ANCHOR_SHA256
        or identity.get("argmax_equal") is not True
        or identity.get("native_argmax") != identity.get("independent_argmax")
        or identity.get("max_abs_logit_difference", math.inf) > HEAD_IDENTITY_MAX_ABS
        or identity.get("hidden_dtype") != MODEL_FORWARD_DTYPE
    ):
        raise RuntimeError("E8 preflight head identity failed")
    _verify_basis(payload.get("basis_and_directions"))
    _verify_implementation_ancestry(payload.get("implementation_commit"))
    _verify_source_receipt(payload)
    verify_protocol()
    verify_dependencies()
    verify_snapshot(snapshot_path())
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    if not EXECUTION_RECEIPT_PATH.is_file():
        raise FileNotFoundError("missing E8 execution receipt")
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
        "case_sha256": CASE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "basis_and_directions": preflight["basis_and_directions"],
        "bound_inputs": preflight["bound_inputs"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "fresh_prompt_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
        "test_command": "python -m pytest tests/test_e8_3b.py -q",
        "tests_returncode": 0,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E8 execution receipt field mismatch: {name}")
    if payload.get("environment") != preflight.get("environment"):
        raise RuntimeError("E8 execution environment differs from preflight")
    if payload.get("native_runtime") != preflight.get("native_runtime"):
        raise RuntimeError("E8 native runtime differs from preflight")
    _verify_basis(payload.get("basis_and_directions"))
    _verify_source_receipt(payload)
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    if not ATTEMPT_PATH.is_file():
        raise FileNotFoundError("missing E8 burn marker")
    payload = load_json_object(ATTEMPT_PATH)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": execution_sha,
        "case_sha256": CASE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "pre_admission_outputs_absent": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E8 attempt field mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _direct_failures(
    rows: Any,
    cases: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    label: str,
) -> tuple[list[str], dict[str, Any] | None]:
    failures: list[str] = []
    if not isinstance(rows, list) or len(rows) != CASE_COUNT:
        return [f"{label} direct row count mismatch"], None
    for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
        prefix = f"{label} direct row {index}"
        if (
            row.get("case_index") != index
            or row.get("case_id") != case["case_id"]
            or row.get("target_id") != case["target_id"]
        ):
            failures.append(prefix + " identity mismatch")
        baseline = baseline_rows[index]
        if (
            row.get("baseline_prediction") != baseline.get("prediction")
            or row.get("baseline_correct") != baseline.get("baseline_correct")
            or row.get("baseline_logits_sha256") != baseline.get("logits_sha256")
        ):
            failures.append(prefix + " baseline binding mismatch")
        geometry = row.get("geometry_class")
        lower = row.get("L")
        upper = row.get("U")
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
            failures.append(prefix + " invalid lower bound")
            continue
        if upper is not None and (
            not isinstance(upper, (int, float)) or not math.isfinite(float(upper))
        ):
            failures.append(prefix + " invalid upper bound")
        if geometry == GEOMETRIC_REACHABLE:
            beta = row.get("beta")
            margin = row.get("float64_margin")
            upper_value = math.inf if upper is None else float(upper)
            if (
                not isinstance(beta, (int, float))
                or not float(lower) < float(beta) < upper_value
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or float(margin) <= 0
                or row.get("native_checked") is not True
            ):
                failures.append(prefix + " invalid reachable evidence")
            if row.get("native_logits_dtype") != "bfloat16" or row.get(
                "native_logits_shape"
            ) != [MODEL_VOCAB_SIZE]:
                failures.append(prefix + " invalid native tensor contract")
            if not _valid_hash(row.get("native_logits_sha256")):
                failures.append(prefix + " invalid native logit hash")
            target_logit = row.get("native_target_logit")
            competitor_logit = row.get("native_competitor_logit")
            native_margin = row.get("native_margin")
            prediction = row.get("native_prediction")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in (target_logit, competitor_logit, native_margin)
            ) or not isinstance(prediction, int):
                failures.append(prefix + " invalid native values")
            else:
                expected_margin = float(
                    torch.tensor(float(target_logit), dtype=torch.float32)
                    - torch.tensor(float(competitor_logit), dtype=torch.float32)
                )
                if expected_margin != float(native_margin):
                    failures.append(prefix + " native margin mismatch")
                expected_decision = native_decision(
                    target_id=int(row["target_id"]),
                    prediction=prediction,
                    native_margin=float(native_margin),
                )
                if row.get("decision") != expected_decision:
                    failures.append(prefix + " native decision mismatch")
        elif geometry == REFUSE_HARD:
            if (
                row.get("decision") != REFUSE_HARD
                or row.get("hard_blocker") is not True
            ):
                failures.append(prefix + " hard refusal mismatch")
        elif geometry == REFUSE_INTERVAL:
            upper_value = math.inf if upper is None else float(upper)
            if row.get("decision") != REFUSE_INTERVAL or float(lower) < upper_value:
                failures.append(prefix + " interval refusal mismatch")
        else:
            failures.append(prefix + " unknown geometry class")
        expected_eligible = not bool(row.get("baseline_correct"))
        if row.get("eligible") is not expected_eligible:
            failures.append(prefix + " eligibility mismatch")
        expected_admit = expected_eligible and row.get("decision") == NATIVE_ADMIT
        if row.get("deployment_admitted") is not expected_admit:
            failures.append(prefix + " deployment admission mismatch")
        if not _valid_hash(row.get("baseline_logits_sha256")):
            failures.append(prefix + " baseline hash missing")
    reachable = [
        row for row in rows if row.get("geometry_class") == GEOMETRIC_REACHABLE
    ]
    for position, row in enumerate(reachable):
        start = (position // HEAD_BATCH) * HEAD_BATCH
        size = min(HEAD_BATCH, len(reachable) - start)
        if (
            row.get("native_batch_start") != start
            or row.get("native_batch_size") != size
            or row.get("native_batch_offset") != position - start
        ):
            failures.append(f"{label} reachable row {position} batch mismatch")
    try:
        summary = direct_summary(rows)
    except Exception as error:
        failures.append(f"{label} direct summary exception: {error}")
    else:
        return failures, summary
    return failures, None


def _memory_failures(
    receipt: Any,
    direct_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    label: str,
) -> list[str]:
    if not isinstance(receipt, dict):
        return [f"{label} memory receipt missing"]
    failures: list[str] = []
    admitted_indices = [
        index for index, row in enumerate(direct_rows) if row["deployment_admitted"]
    ]
    expected = {
        "memory_size": MEMORY_SIZE,
        "degree": DEGREE,
        "temperature": TEMPERATURE,
        "active_count": len(admitted_indices),
        "active_slots": list(range(len(admitted_indices))),
        "active_alphas": [1.0] * len(admitted_indices),
        "active_mask_dtype": "bool",
        "active_mask_shape": [MEMORY_SIZE],
        "active_keys_shape": [len(admitted_indices), MODEL_HIDDEN_SIZE],
        "active_values_shape": [len(admitted_indices), MODEL_HIDDEN_SIZE],
        "active_alphas_shape": [len(admitted_indices)],
        "key_dtype": "float32",
        "value_dtype": "float32",
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            failures.append(f"{label} memory field mismatch: {name}")
    expected_slot_map = {
        str(case_index): slot for slot, case_index in enumerate(admitted_indices)
    }
    if receipt.get("slot_by_case") != expected_slot_map:
        failures.append(f"{label} memory slot map mismatch")
    labels = receipt.get("active_labels")
    expected_labels = [cases[index]["target"] for index in admitted_indices]
    if labels != expected_labels:
        failures.append(f"{label} memory labels mismatch")
    for name in (
        "active_mask_sha256",
        "active_keys_sha256",
        "active_values_sha256",
        "active_alphas_sha256",
        "key_mean_sha256",
        "key_transform_sha256",
    ):
        if not _valid_hash(receipt.get(name)):
            failures.append(f"{label} memory hash missing: {name}")
    return failures


def _gate_failures(
    rows: Any,
    summary: Any,
    cases: list[dict[str, Any]],
    neutral: list[str],
    direct_rows: list[dict[str, Any]],
    memory: dict[str, Any],
    label: str,
) -> list[str]:
    specs = gate_specs(cases, neutral)
    if not isinstance(rows, list) or len(rows) != len(specs):
        return [f"{label} gate row count mismatch"]
    failures: list[str] = []
    slot_by_case = memory.get("slot_by_case", {})
    for index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        prefix = f"{label} gate row {index}"
        for field in ("kind", "prompt", "case_index", "case_id", "target_id"):
            if row.get(field) != spec.get(field):
                failures.append(prefix + f" field mismatch: {field}")
        if not isinstance(row.get("actual_score"), (int, float)) or not math.isfinite(
            float(row.get("actual_score", math.nan))
        ):
            failures.append(prefix + " invalid selected score")
            actual_score = math.nan
        else:
            actual_score = float(row["actual_score"])
        active_slots = memory.get("active_slots", [])
        selected_slot = row.get("selected_slot")
        if not active_slots:
            if selected_slot != -1 or actual_score != 0.0:
                failures.append(prefix + " invalid empty-memory selection")
            expected_gate_open = False
        else:
            if selected_slot not in active_slots or actual_score < 0:
                failures.append(prefix + " selected an invalid memory slot")
            expected_gate_open = math.isfinite(actual_score) and (
                actual_score >= GATE_THRESHOLD
            )
        if row.get("gate_open") is not expected_gate_open:
            failures.append(prefix + " gate decision disagrees with selected score")
        if row.get("logits_dtype") != "bfloat16" or row.get("logits_shape") != [
            MODEL_VOCAB_SIZE
        ]:
            failures.append(prefix + " invalid logit tensor contract")
        for name in (
            "delta_sha256",
            "baseline_logits_sha256",
            "adapted_logits_sha256",
        ):
            if not _valid_hash(row.get(name)):
                failures.append(prefix + f" missing {name}")
        if row.get("gate_open") is False and (
            row.get("delta_zero") is not True
            or row.get("logits_bit_identical") is not True
            or row.get("baseline_logits_sha256") != row.get("adapted_logits_sha256")
        ):
            failures.append(prefix + " gate-closed row is not exact identity")
        case_index = row.get("case_index")
        if row.get("kind") == "exact" and isinstance(case_index, int):
            expected_admit = direct_rows[case_index]["deployment_admitted"]
            if row.get("deployment_admitted") is not expected_admit:
                failures.append(prefix + " admission annotation mismatch")
            expected_slot = slot_by_case.get(str(case_index))
            if row.get("own_slot") != expected_slot:
                failures.append(prefix + " own slot mismatch")
            own_weight = row.get("own_slot_weight")
            if expected_slot is None:
                if own_weight is not None:
                    failures.append(
                        prefix + " refused exact row has an own-slot weight"
                    )
            elif not isinstance(own_weight, (int, float)) or not (
                math.isfinite(float(own_weight)) and 0.0 <= float(own_weight) <= 1.0
            ):
                failures.append(prefix + " invalid own-slot weight")
        if isinstance(case_index, int):
            target_id = int(row["target_id"])
            prediction = row.get("adapted_prediction")
            margin = row.get("target_margin")
            expected_success = bool(
                prediction == target_id
                and isinstance(margin, (int, float))
                and float(margin) > 0
            )
            if row.get("target_success_strict") is not expected_success:
                failures.append(prefix + " target-success annotation mismatch")
        start = (index // HEAD_BATCH) * HEAD_BATCH
        size = min(HEAD_BATCH, len(rows) - start)
        if (
            row.get("head_batch_start") != start
            or row.get("head_batch_size") != size
            or row.get("head_batch_offset") != index - start
        ):
            failures.append(prefix + " head batch mismatch")
    admitted_exact = [
        row for row in rows if row["kind"] == "exact" and row["deployment_admitted"]
    ]
    off_support = [
        row
        for row in rows
        if row["kind"] != "exact" or not row.get("deployment_admitted", False)
    ]
    closed = [row for row in rows if not row["gate_open"]]
    expected_summary = {
        "gate_row_count": len(rows),
        "admitted_exact_checked": len(admitted_exact),
        "admitted_exact_gate_open": sum(row["gate_open"] for row in admitted_exact),
        "admitted_exact_own_slot": sum(
            row["selected_slot"] == row["own_slot"] for row in admitted_exact
        ),
        "admitted_exact_target_success": sum(
            row["target_success_strict"] for row in admitted_exact
        ),
        "off_support_checked": len(off_support),
        "false_gate_applications": sum(row["gate_open"] for row in off_support),
        "gate_closed_checked": len(closed),
        "gate_closed_bit_identical": sum(row["logits_bit_identical"] for row in closed),
        "paraphrases_checked": 128,
        "neighborhoods_checked": 128,
        "neutrals_checked": 8,
    }
    if summary != expected_summary:
        failures.append(f"{label} gate summary mismatch")
    return failures


def _rollback_failures(rollback: Any, prompts: list[str]) -> list[str]:
    if not isinstance(rollback, dict):
        return ["rollback object missing"]
    failures: list[str] = []
    post = rollback.get("post_remove_memory", {})
    if (
        post.get("active_count") != 0
        or post.get("removed_alphas_zero") is not True
        or post.get("removed_labels_none") is not True
        or not _valid_hash(post.get("active_mask_sha256"))
    ):
        failures.append("rollback memory did not return to empty")
    rows = rollback.get("rows")
    if not isinstance(rows, list) or len(rows) != PROMPT_COUNT:
        return failures + ["rollback row count mismatch"]
    for index, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
        if row.get("prompt_index") != index or row.get("prompt") != prompt:
            failures.append(f"rollback row {index} identity mismatch")
        if (
            row.get("gate_open") is not False
            or row.get("selected_slot") != -1
            or row.get("delta_zero") is not True
            or row.get("logits_bit_identical") is not True
            or row.get("baseline_logits_sha256") != row.get("adapted_logits_sha256")
        ):
            failures.append(f"rollback row {index} is not exact baseline")
        start = (index // HEAD_BATCH) * HEAD_BATCH
        size = min(HEAD_BATCH, len(rows) - start)
        if (
            row.get("head_batch_start") != start
            or row.get("head_batch_size") != size
            or row.get("head_batch_offset") != index - start
        ):
            failures.append(f"rollback row {index} head batch mismatch")
    expected_summary = {
        "prompt_count": PROMPT_COUNT,
        "gate_open_count": 0,
        "delta_zero_count": PROMPT_COUNT,
        "bit_identical_count": PROMPT_COUNT,
    }
    if rollback.get("summary") != expected_summary:
        failures.append("rollback summary mismatch")
    return failures


def measurement_validity_failures(measurement: Any) -> list[str]:
    if not isinstance(measurement, dict):
        return ["measurement object missing"]
    failures: list[str] = []
    cases = measurement.get("cases")
    prompts = measurement.get("prompt_order")
    samples = registered_samples()
    neutral = list(samples["NEUTRAL"])
    if (
        not isinstance(cases, list)
        or stable_json_sha256(cases) != CASE_SHA256
        or measurement.get("case_sha256") != CASE_SHA256
    ):
        failures.append("measurement case population mismatch")
        return failures
    if (
        not isinstance(prompts, list)
        or len(prompts) != PROMPT_COUNT
        or stable_json_sha256(prompts) != PROMPT_SHA256
        or measurement.get("prompt_order_sha256") != PROMPT_SHA256
    ):
        failures.append("measurement prompt order mismatch")
        return failures
    baseline = measurement.get("baseline_rows")
    if not isinstance(baseline, list) or len(baseline) != CASE_COUNT:
        failures.append("measurement baseline row count mismatch")
    else:
        for index, (row, case) in enumerate(zip(baseline, cases, strict=True)):
            if (
                row.get("case_index") != index
                or row.get("case_id") != case["case_id"]
                or row.get("target_id") != case["target_id"]
                or row.get("baseline_correct")
                != (row.get("prediction") == case["target_id"])
                or row.get("logits_dtype") != "bfloat16"
                or row.get("logits_shape") != [MODEL_VOCAB_SIZE]
                or not _valid_hash(row.get("logits_sha256"))
            ):
                failures.append(f"baseline row {index} mismatch")
    whitening = measurement.get("key_whitening", {})
    if (
        whitening.get("floor_fraction") != 0.01
        or whitening.get("exact_key_count") != CASE_COUNT
        or whitening.get("whitening_prompt_count") != 18
        or whitening.get("mean_dtype") != "float32"
        or whitening.get("mean_shape") != [MODEL_HIDDEN_SIZE]
        or whitening.get("transform_dtype") != "float32"
        or whitening.get("transform_shape") != [MODEL_HIDDEN_SIZE, MODEL_HIDDEN_SIZE]
        or not _valid_hash(whitening.get("mean_sha256"))
        or not _valid_hash(whitening.get("transform_sha256"))
    ):
        failures.append("key-whitening receipt mismatch")
    arms = measurement.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"raw", "candidate"}:
        return failures + ["measurement arms mismatch"]
    for label in ("raw", "candidate"):
        arm = arms[label]
        direct_failures, expected_direct_summary = _direct_failures(
            arm.get("direct_rows"), cases, baseline, label
        )
        failures.extend(direct_failures)
        if arm.get("direct_summary") != expected_direct_summary:
            failures.append(f"{label} direct summary mismatch")
        failures.extend(
            _memory_failures(
                arm.get("memory_receipt"), arm["direct_rows"], cases, label
            )
        )
        failures.extend(
            _gate_failures(
                arm.get("gate_rows"),
                arm.get("gate_summary"),
                cases,
                neutral,
                arm["direct_rows"],
                arm["memory_receipt"],
                label,
            )
        )
    failures.extend(_rollback_failures(measurement.get("rollback"), prompts))
    raw_ids = set(arms["raw"]["direct_summary"]["admitted_case_ids"])
    candidate_ids = set(arms["candidate"]["direct_summary"]["admitted_case_ids"])
    expected_summary = {
        "case_count": CASE_COUNT,
        "eligible_count": arms["candidate"]["direct_summary"]["eligible_count"],
        "raw_admitted": len(raw_ids),
        "candidate_admitted": len(candidate_ids),
        "candidate_gain": len(candidate_ids) - len(raw_ids),
        "candidate_lost_raw_case_ids": sorted(raw_ids - candidate_ids),
        "candidate_lost_raw_count": len(raw_ids - candidate_ids),
        "candidate_gate": arms["candidate"]["gate_summary"],
        "rollback": measurement["rollback"]["summary"],
    }
    if measurement.get("summary") != expected_summary:
        failures.append("measurement aggregate summary mismatch")
    return failures


def admission_validity_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        failures.append("admission schema or status mismatch")
    try:
        verify_scientific_hash(payload, "E8 admission")
    except Exception as error:
        return failures + [str(error)]
    scientific = payload["scientific"]
    if scientific.get("model_invariants") != expected_model_invariants():
        failures.append("admission model invariants mismatch")
    try:
        _verify_basis(scientific.get("basis_and_directions"))
    except Exception as error:
        failures.append(str(error))
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
            raise RuntimeError(f"E8 admission provenance mismatch: {name}")
    if payload.get("scientific", {}).get("basis_and_directions") != receipt.get(
        "basis_and_directions"
    ):
        raise RuntimeError("E8 admission basis differs from registration")


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    if not ADMISSION_PATH.is_file():
        raise FileNotFoundError("missing E8 admission")
    payload = load_json_object(ADMISSION_PATH)
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E8 admission schema or status mismatch")
    verify_admission_provenance(
        payload, receipt=receipt, execution_sha=execution_sha, attempt_sha=attempt_sha
    )
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("invalid E8 admission: " + "; ".join(failures))
    return payload, sha256_file(ADMISSION_PATH)


def result_validity_failures(
    result: dict[str, Any], admission: dict[str, Any]
) -> list[str]:
    try:
        verify_scientific_hash(result, "E8 result")
    except Exception as error:
        return [str(error)]
    scientific = result["scientific"]
    frozen = admission["scientific"]
    failures: list[str] = []
    if scientific.get("admission_scientific_sha256") != admission.get(
        "scientific_sha256"
    ):
        failures.append("result binds a different admission")
    if scientific.get("model_invariants") != frozen.get("model_invariants"):
        failures.append("result model invariants differ from admission")
    if scientific.get("basis_and_directions") != frozen.get("basis_and_directions"):
        failures.append("result basis differs from admission")
    if scientific.get("measurement") != frozen.get("measurement"):
        failures.append("new-process replay measurement differs from admission")
    for name in (
        "measurement_exact_match",
        "raw_full_logit_hashes_match",
        "candidate_full_logit_hashes_match",
        "rollback_full_logit_hashes_match",
    ):
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
            raise RuntimeError(f"E8 result provenance mismatch: {name}")


def load_result() -> tuple[dict[str, Any], str]:
    receipt, _execution_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    if not RESULT_PATH.is_file():
        raise FileNotFoundError("missing E8 replay result")
    payload = load_json_object(RESULT_PATH)
    if payload.get("schema") != RESULT_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E8 result schema or status mismatch")
    verify_result_provenance(
        payload,
        receipt=receipt,
        admission=admission,
        admission_sha=admission_sha,
        attempt_sha=attempt_sha,
    )
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E8 result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


def scientific_failures(result: dict[str, Any]) -> list[str]:
    measurement = result.get("scientific", {}).get("measurement", {})
    summary = measurement.get("summary", {})
    failures: list[str] = []
    eligible = summary.get("eligible_count")
    raw_admitted = summary.get("raw_admitted")
    candidate_admitted = summary.get("candidate_admitted")
    if not isinstance(eligible, int) or eligible < MIN_ELIGIBLE:
        failures.append("fewer than 48 cases are edit-eligible")
    if isinstance(eligible, int) and isinstance(candidate_admitted, int):
        if (
            MIN_ADMISSION_DENOMINATOR * candidate_admitted
            < MIN_ADMISSION_NUMERATOR * eligible
        ):
            failures.append("candidate deployment admission is below 95% of eligible")
    else:
        failures.append("candidate admission counts are missing")
    if (
        not isinstance(raw_admitted, int)
        or not isinstance(candidate_admitted, int)
        or candidate_admitted < raw_admitted
    ):
        failures.append("candidate admits fewer facts than raw control")
    if summary.get("candidate_lost_raw_count", math.inf) > MAX_RAW_ADMITTED_LOSS:
        failures.append("candidate loses more than one raw-admitted fact")
    gate = summary.get("candidate_gate", {})
    if isinstance(candidate_admitted, int):
        for name in (
            "admitted_exact_checked",
            "admitted_exact_gate_open",
            "admitted_exact_own_slot",
            "admitted_exact_target_success",
        ):
            if gate.get(name) != candidate_admitted:
                failures.append(f"candidate full-path bar failed: {name}")
    if gate.get("false_gate_applications") != 0:
        failures.append("a primary-arm off-support prompt opened the gate")
    rollback = summary.get("rollback", {})
    if rollback != {
        "prompt_count": PROMPT_COUNT,
        "gate_open_count": 0,
        "delta_zero_count": PROMPT_COUNT,
        "bit_identical_count": PROMPT_COUNT,
    }:
        failures.append("complete memory removal did not restore exact baseline")
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
            else result.get("scientific", {}).get("measurement", {}).get("summary")
        ),
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Pinned frozen SmolLM3-3B-Base, registered A100/BF16 runtime, "
            "64 preregistered exact-key CounterFact cases, fixed head-only ZCA "
            "values, released HLM5 multi-slot gate, and exact complete rollback."
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
    "stable_json",
]
