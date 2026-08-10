"""Fail-closed provenance and verdict contract for E7c confirmation."""

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
    require_file_hash,
    sha256_file,
    snapshot_path,
    stable_json,
    stable_json_sha256,
    verify_snapshot,
)
from .e7b_contract import (
    BASELINE_ANCHOR_SHA256,
    ELIGIBLE_TOKEN_COUNT,
    EXPECTED_NATIVE_RUNTIME,
    PROMPT,
    PROMPT_SHA256,
    eligible_token_ids,
)
from .e7b_runtime import (
    GEOMETRIC_REACHABLE,
    HEAD_BATCH,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    native_decision,
    summarize_rows,
)


PROTOCOL_COMMIT = "18df17ad9ffc9001b8c0c2e75ce5291690012253"
PROTOCOL_PATH = "docs/e7c-fresh-zca-geometry-confirmation-prereg-2026-08-10.md"
PROTOCOL_SHA256 = "59f7b4570951378041be3f800bb8ce1e7438e449b9d95d057a98cbaa2fc6039a"
E7B_RESULT_PATH = "results/e7b_3b_evidence/result.json"
E7B_RESULT_SHA256 = "9a5232b33f8828195794cb137ee2ec83e10ec8af524418abe3608cf0ac2d4d11"
E7B_VERDICT_PATH = "results/e7b_3b_evidence/verdict.json"
E7B_VERDICT_SHA256 = "5f5e35f71d432d2df0287c7dec1350e8b53c3dae7bea0f025634536a3a6fd016"
SPENT_SCREEN_PATH = "results/e7c_spent_geometry_screen.json"
SPENT_SCREEN_SHA256 = "80e529870bfd9d727e85ae4115c8c399b3f23ad94143d1a01cc713e060790104"
SPENT_SCREEN_SCIENTIFIC_SHA256 = (
    "83e17daed19cfcf5b13e1d6672714d9f7537dde5ed0466a4cbc64631568c6bf2"
)
SCREEN_IMPLEMENTATION_COMMIT = "ec182f9f6511a513e2a1ba923d2fe50e13e5b00d"
SELECTED_CANDIDATE = "zca_half_raw_r1e2"

E7_POOL_SHA256 = "5660e3ae657779c4a2444cc4123848b86a4e2669aa7fea5dd5972532fc126b52"
E7B_POOL_SHA256 = "8fb09e2ceda0ad576dcb4771cbdbf757810cfd90969e3e8b4283d9582495136f"
TARGET_POOL_OFFSET = 2400
TARGET_POOL_COUNT = 1200
TARGET_POOL_FIRST = 7863
TARGET_POOL_LAST = 10844
TARGET_POOL_SHA256 = "fdde53009a17b075fadacafeb15053df08f7f9e97d12f5a9c950d0a5b58b2ff4"
SEED = 0
MIN_CANDIDATE_GEOMETRIC = 1188
MIN_CANDIDATE_NATIVE = 1188
MIN_REACHABILITY_GAIN = 180
MAX_RAW_REACHABLE_LOST = 12

EXPECTED_BASIS = {
    "mean_sha256": ("ae2b4c5d0c3aa902b7ae61dfb320177fea40270c1450f15a9fb5c064bed9e489"),
    "eigenvalues_sha256": (
        "7f33aaca070a3539f8c4c65a0e8a56ef82a2813d950ec910335160f971da4971"
    ),
    "eigenvectors_sha256": (
        "d5e1b3f6c5fafa4b9fd46bfbcbf1eddd982a2b19ea9a40542ba236a8ebdb3fad"
    ),
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

PREFLIGHT_SCHEMA = "hlm5-e7c-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e7c-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e7c-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e7c-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e7c-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e7c-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57",
    "hlm5/memory.py": "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973",
    "hlm5/model.py": "2f810bde52bbf3e0da9f52640bf36ca924cd665bd6a7036155f0581494b8857c",
    "hlm5/io.py": "5886ce8331ba42ca0ee75dd67746665be58d174e5f3416ad661f81c7d7685439",
    "hlm5/e7_contract.py": "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d",
    "hlm5/e7_runtime.py": "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc",
    "hlm5/e7b_contract.py": "606d95ca4972d1febadf6348c441449574f1c33eb2f3b7b71b3f79df27c31119",
    "hlm5/e7b_runtime.py": "bf6e7b0abfe83d0da0f146ae72dc69d52b0469fcc311689c535a7628ea2fdbc2",
    "hlm5/certify.py": "a13544eb242cb1f827496a46a6d943723ad51f35b827fe47d2da641e26a1c210",
    "hlm5/edit_audit.py": "930d51619bb240e13a15412f6db92b0cf58a87e17a405dc7110bbaaa2e8e8930",
    "hlm5/key_value.py": "a0bacdfcba80397bfacefa7c7762bcb202fb91363da0e6fa902f9c707c70140a",
    "hlm5/public_adapter.py": "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d",
    "scripts/run_e7_3b.py": "48ba22bfc1e6a0824a7f659bd945e028e8d1eb5999b07c21d018a74e6a97d10e",
    "scripts/analyze_e7c_spent_geometry.py": "2d24c8367f3c6e9f1734d16107b96b17f224811124a1d6300e2d5599619aa109",
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/e7c_contract.py",
    "hlm5/e7c_runtime.py",
    "scripts/preflight_e7c_3b.py",
    "scripts/register_e7c_3b.py",
    "scripts/admit_e7c_3b.py",
    "scripts/replay_e7c_3b.py",
    "scripts/verify_e7c_3b.py",
    "tests/test_e7c_3b.py",
)
REGISTERED_COMMANDS = (
    "python scripts/preflight_e7c_3b.py",
    "python scripts/register_e7c_3b.py",
    "python scripts/admit_e7c_3b.py",
    "python scripts/replay_e7c_3b.py",
    "python scripts/verify_e7c_3b.py",
)

STAGING_DIR = RESULTS_DIR / "e7c_3b_staging"
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
        raise ValueError(f"E7c output must stay under {staging}: {resolved}")
    return resolved


def atomic_create_json(path: Path, value: Any) -> None:
    """Atomically publish a complete JSON file without overwrite semantics."""
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
            raise FileExistsError(f"E7c output already exists: {resolved}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink()


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E7c outputs already exist: {present}")


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
        raise RuntimeError("registered E7c protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    require_file_hash(REPO_ROOT / PROTOCOL_PATH, PROTOCOL_SHA256, "E7c protocol")


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E7c dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def verify_development_evidence() -> dict[str, Any]:
    pins = {
        "e7b_result": (E7B_RESULT_PATH, E7B_RESULT_SHA256),
        "e7b_verdict": (E7B_VERDICT_PATH, E7B_VERDICT_SHA256),
        "spent_screen": (SPENT_SCREEN_PATH, SPENT_SCREEN_SHA256),
    }
    for label, (relative, digest) in pins.items():
        require_file_hash(REPO_ROOT / relative, digest, label)
    e7b_verdict = load_json_object(REPO_ROOT / E7B_VERDICT_PATH)
    screen = load_json_object(REPO_ROOT / SPENT_SCREEN_PATH)
    if e7b_verdict.get("verdict") != "FAIL_3B_NATIVE_ADMISSION":
        raise RuntimeError("bound E7b verdict is not the valid scientific failure")
    if screen.get("scientific_sha256") != SPENT_SCREEN_SCIENTIFIC_SHA256:
        raise RuntimeError("spent E7c scientific hash mismatch")
    if screen.get("implementation_commit") != SCREEN_IMPLEMENTATION_COMMIT:
        raise RuntimeError("spent E7c implementation commit mismatch")
    scientific = screen.get("scientific", {})
    if (
        scientific.get("selected_candidate") != SELECTED_CANDIDATE
        or scientific.get("ready_for_fresh_pool_preregistration") is not True
    ):
        raise RuntimeError("spent E7c evidence did not select the registered candidate")
    if screen.get("scientific_sha256") != stable_json_sha256(scientific):
        raise RuntimeError("spent E7c scientific object is not canonical")
    return {
        "e7b_result_sha256": E7B_RESULT_SHA256,
        "e7b_verdict_sha256": E7B_VERDICT_SHA256,
        "spent_screen_sha256": SPENT_SCREEN_SHA256,
        "spent_screen_scientific_sha256": SPENT_SCREEN_SCIENTIFIC_SHA256,
        "selected_candidate": SELECTED_CANDIDATE,
    }


def target_pools(tokenizer: Any) -> tuple[list[int], list[int], list[int]]:
    eligible = eligible_token_ids(tokenizer)
    if len(eligible) != ELIGIBLE_TOKEN_COUNT:
        raise RuntimeError(f"E7c eligible-token count mismatch: {len(eligible)}")
    e7 = eligible[:1200]
    e7b = eligible[1200:2400]
    fresh = eligible[TARGET_POOL_OFFSET : TARGET_POOL_OFFSET + TARGET_POOL_COUNT]
    if stable_json_sha256(e7) != E7_POOL_SHA256:
        raise RuntimeError("E7 target-pool hash mismatch")
    if stable_json_sha256(e7b) != E7B_POOL_SHA256:
        raise RuntimeError("E7b target-pool hash mismatch")
    if (
        len(fresh) != TARGET_POOL_COUNT
        or fresh[0] != TARGET_POOL_FIRST
        or fresh[-1] != TARGET_POOL_LAST
        or stable_json_sha256(fresh) != TARGET_POOL_SHA256
        or set(fresh) & (set(e7) | set(e7b))
    ):
        raise RuntimeError("registered disjoint E7c target pool mismatch")
    return e7, e7b, fresh


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
        raise FileNotFoundError(f"registered E7c sources missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E7c sources must be committed:\n" + dirty)
    return {p: sha256_file(REPO_ROOT / p) for p in REGISTERED_SOURCE_PATHS}


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("source_sha256") != registered_source_sha256():
        raise RuntimeError("registered E7c source drift")
    if receipt.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E7c dependency receipt mismatch")


def _verify_basis(record: Any) -> None:
    if not isinstance(record, dict):
        raise RuntimeError("missing E7c basis record")
    for name, value in EXPECTED_BASIS.items():
        if record.get(name) != value:
            raise RuntimeError(f"E7c basis mismatch: {name}")
    if record.get("operator_dtype") != "float64" or record.get("operator_shape") != [
        MODEL_HIDDEN_SIZE,
        MODEL_HIDDEN_SIZE,
    ]:
        raise RuntimeError("E7c operator tensor contract mismatch")
    for name in (
        "operator_sha256",
        "raw_directions_sha256",
        "zca_directions_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get(name))) is None:
            raise RuntimeError(f"E7c basis record lacks {name}")
    if record.get("direction_dtype") != "float64" or record.get("direction_shape") != [
        TARGET_POOL_COUNT,
        MODEL_HIDDEN_SIZE,
    ]:
        raise RuntimeError("E7c direction tensor contract mismatch")


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E7c preflight")
    payload = load_json_object(PREFLIGHT_PATH)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "eligible_token_count": ELIGIBLE_TOKEN_COUNT,
        "target_pool_count": TARGET_POOL_COUNT,
        "target_pool_first": TARGET_POOL_FIRST,
        "target_pool_last": TARGET_POOL_LAST,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "registered_target_pool_count": TARGET_POOL_COUNT,
        "registered_target_pool_first": TARGET_POOL_FIRST,
        "registered_target_pool_last": TARGET_POOL_LAST,
        "registered_target_pool_sha256": TARGET_POOL_SHA256,
        "target_pool_overlap_with_e7": 0,
        "target_pool_overlap_with_e7b": 0,
        "raw_candidate_equal": False,
        "preflight_test_command": "python -m pytest tests/test_e7c_3b.py -q",
        "preflight_tests_returncode": 0,
        "fresh_prompt_outcomes_computed": False,
        "commands": list(REGISTERED_COMMANDS),
        "development_evidence": verify_development_evidence(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7c preflight field mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E7c preflight model invariants mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7c preflight native arithmetic mismatch")
    if payload.get("environment") != EXPECTED_ENVIRONMENT:
        raise RuntimeError("E7c preflight environment mismatch")
    if payload.get("implementation_commit") != git_output("rev-parse", "HEAD"):
        raise RuntimeError("E7c preflight implementation commit is not live HEAD")
    identity = payload.get("baseline_only_head_identity", {})
    if (
        identity.get("prompt_sha256") != BASELINE_ANCHOR_SHA256
        or identity.get("argmax_equal") is not True
        or identity.get("native_argmax") != identity.get("independent_argmax")
        or identity.get("max_abs_logit_difference", math.inf) > HEAD_IDENTITY_MAX_ABS
        or identity.get("hidden_dtype") != MODEL_FORWARD_DTYPE
    ):
        raise RuntimeError("E7c preflight head identity failed")
    _verify_basis(payload.get("basis_and_directions"))
    _verify_source_receipt(payload)
    verify_protocol()
    verify_dependencies()
    verify_snapshot(snapshot_path())
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    if not EXECUTION_RECEIPT_PATH.is_file():
        raise FileNotFoundError("missing E7c execution receipt")
    payload = load_json_object(EXECUTION_RECEIPT_PATH)
    expected = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "preflight_sha256": preflight_sha,
        "model_file_sha256": MODEL_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "basis_and_directions": preflight["basis_and_directions"],
        "development_evidence": preflight["development_evidence"],
        "registered_commands": list(REGISTERED_COMMANDS),
        "fresh_prompt_outcomes_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
        "preflight_file_sha256_recomputed": preflight_sha,
        "test_command": "python -m pytest tests/test_e7c_3b.py -q",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7c execution receipt field mismatch: {name}")
    if payload.get("tests_returncode") != 0:
        raise RuntimeError("registered E7c tests did not pass")
    if payload.get("environment") != preflight.get("environment"):
        raise RuntimeError("E7c execution environment differs from preflight")
    if payload.get("native_runtime") != preflight.get("native_runtime"):
        raise RuntimeError("E7c native runtime differs from preflight")
    _verify_basis(payload.get("basis_and_directions"))
    _verify_source_receipt(payload)
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    if not ATTEMPT_PATH.is_file():
        raise FileNotFoundError("missing E7c burn marker")
    payload = load_json_object(ATTEMPT_PATH)
    expected = {
        "schema": ATTEMPT_SCHEMA,
        "status": "POOL_SPENT",
        "execution_receipt_sha256": execution_sha,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "pre_admission_outputs_absent": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": receipt["implementation_commit"],
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7c attempt field mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def _expected_batch_fields(position: int, total: int) -> tuple[int, int, int]:
    start = (position // HEAD_BATCH) * HEAD_BATCH
    return start, min(HEAD_BATCH, total - start), position - start


def _arm_validity_failures(arm: Any, label: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(arm, dict):
        return [f"{label} arm is not an object"]
    rows = arm.get("rows")
    if not isinstance(rows, list) or len(rows) != TARGET_POOL_COUNT:
        return [f"{label} row count mismatch"]
    if stable_json_sha256([row.get("target_id") for row in rows]) != TARGET_POOL_SHA256:
        failures.append(f"{label} row target ids mismatch")
    reachable: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prefix = f"{label} row {index}"
        if row.get("pool_index") != index:
            failures.append(prefix + " pool index mismatch")
        geometry_class = row.get("geometry_class")
        decision = row.get("decision")
        lower = row.get("L")
        upper = row.get("U")
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
            failures.append(prefix + " invalid lower bound")
            continue
        if upper is not None and (
            not isinstance(upper, (int, float)) or not math.isfinite(float(upper))
        ):
            failures.append(prefix + " invalid upper bound")
        if geometry_class == REFUSE_HARD:
            if decision != REFUSE_HARD or row.get("hard_blocker") is not True:
                failures.append(prefix + " hard-refusal mismatch")
            if row.get("beta") is not None or row.get("native_checked") is not None:
                failures.append(prefix + " hard refusal was evaluated")
        elif geometry_class == REFUSE_INTERVAL:
            upper_value = math.inf if upper is None else float(upper)
            if (
                decision != REFUSE_INTERVAL
                or row.get("hard_blocker") is not False
                or float(lower) < upper_value
            ):
                failures.append(prefix + " interval-refusal mismatch")
            if row.get("beta") is not None or row.get("native_checked") is not None:
                failures.append(prefix + " interval refusal was evaluated")
        elif geometry_class == GEOMETRIC_REACHABLE:
            reachable.append(row)
            upper_value = math.inf if upper is None else float(upper)
            beta = row.get("beta")
            margin = row.get("float64_margin")
            if (
                row.get("hard_blocker") is not False
                or not isinstance(beta, (int, float))
                or not float(lower) < float(beta) < upper_value
                or not isinstance(margin, (int, float))
                or not math.isfinite(float(margin))
                or float(margin) <= 0
            ):
                failures.append(prefix + " invalid geometric reachability")
            if row.get("native_checked") is not True:
                failures.append(prefix + " lacks native evidence")
            prediction = row.get("native_prediction")
            native_margin = row.get("native_margin")
            target_logit = row.get("native_target_logit")
            competitor_logit = row.get("native_competitor_logit")
            if (
                not isinstance(prediction, int)
                or not 0 <= prediction < MODEL_VOCAB_SIZE
                or not isinstance(target_logit, (int, float))
                or not math.isfinite(float(target_logit))
                or not isinstance(competitor_logit, (int, float))
                or not math.isfinite(float(competitor_logit))
                or not isinstance(native_margin, (int, float))
                or not math.isfinite(float(native_margin))
            ):
                failures.append(prefix + " invalid native scalars")
            else:
                recomputed = float(
                    torch.tensor(float(target_logit), dtype=torch.float32)
                    - torch.tensor(float(competitor_logit), dtype=torch.float32)
                )
                if recomputed != float(native_margin):
                    failures.append(prefix + " native margin mismatch")
                if decision != native_decision(
                    target_id=int(row["target_id"]),
                    prediction=int(prediction),
                    native_margin=float(native_margin),
                ):
                    failures.append(prefix + " native decision mismatch")
            if row.get("native_logits_dtype") != "bfloat16" or row.get(
                "native_logits_shape"
            ) != [MODEL_VOCAB_SIZE]:
                failures.append(prefix + " native tensor mismatch")
            if (
                re.fullmatch(r"[0-9a-f]{64}", str(row.get("native_logits_sha256")))
                is None
            ):
                failures.append(prefix + " missing native hash")
            competitor_id = row.get("native_competitor_id")
            if (
                not isinstance(competitor_id, int)
                or competitor_id == row.get("target_id")
                or not 0 <= competitor_id < MODEL_VOCAB_SIZE
            ):
                failures.append(prefix + " competitor mismatch")
        else:
            failures.append(prefix + " unknown geometry class")
    for position, row in enumerate(reachable):
        start, size, offset = _expected_batch_fields(position, len(reachable))
        if (
            row.get("native_batch_start") != start
            or row.get("native_batch_size") != size
            or row.get("native_batch_offset") != offset
        ):
            failures.append(f"{label} reachable row {position} batch mismatch")
    try:
        expected_summary = summarize_rows(rows)
    except Exception as error:
        failures.append(f"{label} summary exception: {error}")
    else:
        if arm.get("summary") != expected_summary:
            failures.append(f"{label} summary mismatch")
    return failures


def dual_summary(
    raw_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = summarize_rows(raw_rows)
    candidate = summarize_rows(candidate_rows)
    raw_reachable = {
        int(row["target_id"])
        for row in raw_rows
        if row.get("geometry_class") == GEOMETRIC_REACHABLE
    }
    candidate_reachable = {
        int(row["target_id"])
        for row in candidate_rows
        if row.get("geometry_class") == GEOMETRIC_REACHABLE
    }
    return {
        "target_count": TARGET_POOL_COUNT,
        "raw": raw,
        "candidate": candidate,
        "reachability_gain": (
            candidate["geometrically_reachable"] - raw["geometrically_reachable"]
        ),
        "gained_reachable_target_ids": sorted(candidate_reachable - raw_reachable),
        "lost_raw_reachable_target_ids": sorted(raw_reachable - candidate_reachable),
        "lost_raw_reachable_count": len(raw_reachable - candidate_reachable),
    }


def admission_validity_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        failures.append("admission schema or status mismatch")
    try:
        verify_scientific_hash(payload, "E7c admission")
    except Exception as error:
        failures.append(str(error))
        return failures
    scientific = payload["scientific"]
    if (
        scientific.get("prompt") != PROMPT
        or scientific.get("prompt_sha256") != PROMPT_SHA256
    ):
        failures.append("admission prompt mismatch")
    if (
        scientific.get("target_pool_sha256") != TARGET_POOL_SHA256
        or scientific.get("target_pool_count") != TARGET_POOL_COUNT
    ):
        failures.append("admission target pool mismatch")
    if scientific.get("model_invariants") != expected_model_invariants():
        failures.append("admission model invariants mismatch")
    try:
        _verify_basis(scientific.get("basis_and_directions"))
    except Exception as error:
        failures.append(str(error))
    if scientific.get("certificate_boundary_dtypes") != {
        "head": "float64",
        "hidden": "float64",
        "raw_directions": "float64",
        "candidate_directions": "float64",
    }:
        failures.append("admission certificate boundary mismatch")
    arms = scientific.get("arms")
    if not isinstance(arms, dict):
        return failures + ["admission arms are missing"]
    failures.extend(_arm_validity_failures(arms.get("raw"), "raw"))
    failures.extend(_arm_validity_failures(arms.get("candidate"), "candidate"))
    if isinstance(arms.get("raw"), dict) and isinstance(arms.get("candidate"), dict):
        raw_rows = arms["raw"].get("rows")
        candidate_rows = arms["candidate"].get("rows")
        if isinstance(raw_rows, list) and isinstance(candidate_rows, list):
            try:
                expected_summary = dual_summary(raw_rows, candidate_rows)
            except Exception as error:
                failures.append(f"dual summary exception: {error}")
            else:
                if scientific.get("summary") != expected_summary:
                    failures.append("admission dual summary mismatch")
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
            raise RuntimeError(f"E7c admission provenance mismatch: {name}")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7c admission native arithmetic is not registered")
    if payload.get("scientific", {}).get("basis_and_directions") != receipt.get(
        "basis_and_directions"
    ):
        raise RuntimeError("E7c admission basis differs from registration")


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    if not ADMISSION_PATH.is_file():
        raise FileNotFoundError("missing E7c admission")
    payload = load_json_object(ADMISSION_PATH)
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E7c admission schema or status mismatch")
    verify_admission_provenance(
        payload, receipt=receipt, execution_sha=execution_sha, attempt_sha=attempt_sha
    )
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("invalid E7c admission: " + "; ".join(failures))
    return payload, sha256_file(ADMISSION_PATH)


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
            raise RuntimeError(f"E7c result provenance mismatch: {name}")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7c result native arithmetic is not registered")


def result_validity_failures(
    result: dict[str, Any], admission: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    try:
        verify_scientific_hash(result, "E7c result")
    except Exception as error:
        return [str(error)]
    scientific = result["scientific"]
    frozen = admission["scientific"]
    if scientific.get("prompt_sha256") != PROMPT_SHA256:
        failures.append("result prompt mismatch")
    if (
        scientific.get("target_pool_sha256") != TARGET_POOL_SHA256
        or scientific.get("target_pool_count") != TARGET_POOL_COUNT
    ):
        failures.append("result target pool mismatch")
    if scientific.get("model_invariants") != frozen.get("model_invariants"):
        failures.append("result model invariants differ from admission")
    if scientific.get("basis_and_directions") != frozen.get("basis_and_directions"):
        failures.append("result basis or direction hashes differ from admission")
    if scientific.get("admission_scientific_sha256") != admission.get(
        "scientific_sha256"
    ):
        failures.append("result binds a different admission")
    if scientific.get("arms") != frozen.get("arms"):
        failures.append("new-process replay arms differ from admission")
    if scientific.get("summary") != frozen.get("summary"):
        failures.append("new-process replay summary differs from admission")
    for name in (
        "raw_all_rows_match",
        "candidate_all_rows_match",
        "raw_all_native_hashes_match",
        "candidate_all_native_hashes_match",
    ):
        if scientific.get(name) is not True:
            failures.append(f"result replay bar failed: {name}")
    return failures


def load_result() -> tuple[dict[str, Any], str]:
    receipt, _execution_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    if not RESULT_PATH.is_file():
        raise FileNotFoundError("missing E7c replay result")
    payload = load_json_object(RESULT_PATH)
    if payload.get("schema") != RESULT_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E7c result schema or status mismatch")
    verify_result_provenance(
        payload,
        receipt=receipt,
        admission=admission,
        admission_sha=admission_sha,
        attempt_sha=attempt_sha,
    )
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E7c result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


def scientific_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    summary = result.get("scientific", {}).get("summary", {})
    raw = summary.get("raw", {})
    candidate = summary.get("candidate", {})
    geometric = candidate.get("geometrically_reachable")
    admitted = candidate.get("native_admitted")
    raw_geometric = raw.get("geometrically_reachable")
    if not isinstance(geometric, int) or geometric < MIN_CANDIDATE_GEOMETRIC:
        failures.append("candidate geometrically reachable count is below 1188")
    if not isinstance(admitted, int) or admitted < MIN_CANDIDATE_NATIVE:
        failures.append("candidate native-admitted count is below 1188")
    if isinstance(geometric, int) and isinstance(admitted, int):
        if 100 * admitted < 99 * geometric:
            failures.append("candidate native retention is below 99%")
    if (
        not isinstance(raw_geometric, int)
        or not isinstance(geometric, int)
        or (geometric - raw_geometric < MIN_REACHABILITY_GAIN)
    ):
        failures.append("candidate reachability gain is below 180 targets")
    lost = summary.get("lost_raw_reachable_count")
    if not isinstance(lost, int) or lost > MAX_RAW_REACHABLE_LOST:
        failures.append("candidate lost more than 12 raw-reachable targets")
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
        "summary": None
        if result is None
        else result.get("scientific", {}).get("summary"),
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Pinned SmolLM3-3B-Base revision, prompt, registered A100/BF16 "
            "runtime, and third disjoint single-token pool; exact affine "
            "reachability and strict ordinary-head wins for the frozen raw-row "
            "ZCA half-whitening direction only."
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
