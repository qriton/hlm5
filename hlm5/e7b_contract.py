"""Fail-closed provenance and verdict contract for the E7b native admission."""

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
from .e7b_runtime import (
    GEOMETRIC_REACHABLE,
    HEAD_BATCH,
    NATIVE_ADMIT,
    REFUSE_HARD,
    REFUSE_INTERVAL,
    REFUSE_TIE,
    native_decision,
    summarize_rows,
)


PROTOCOL_COMMIT = "b034dcf86e7d4463f845484ee7ecacd4848167ca"
PROTOCOL_PATH = "docs/e7b-native-admission-protocol-2026-08-10.md"
PROTOCOL_SHA256 = (
    "087979cb44514f244bb58ca5c7e59837e897802ca504b95202a5b02b7a91a18f"
)
E7_RESULT_PATH = "results/e7_3b_result.json"
E7_RESULT_SHA256 = (
    "42174f0e8d3658e39a8a82b730844fd0c4e77161d4ebc81456243b645141c05c"
)
E7_VERDICT_PATH = "results/e7_3b_verdict.json"
E7_VERDICT_SHA256 = (
    "83c139aac4f25f98ac58dee5fd468f7a2bae920732a8f7587953033370dcf0ed"
)
E7_SENTINEL = {
    "target_id": 922,
    "beta": 217.078019333642,
    "float64_margin": 2.2585672901487284,
    "native_margin": 0.0,
    "native_prediction": 220,
    "native_logits_sha256": (
        "9f39d5f1f7b6c14fac9cab289915c1a8af6ef60bd8d6dbe029f45740b21929a8"
    ),
}
PROMPT = "The capital of Vorenia is"
PROMPT_SHA256 = (
    "0a30c4661b062d65055580bd1dad19d345ab3a9f6f726a28afb7c68548272e1c"
)
BASELINE_ANCHOR_SHA256 = (
    "32b3c71b5eea9980b3b927757d3c5c7854416b3f703170d886dd779630784d33"
)
E7_POOL_COUNT = 1200
E7_POOL_SHA256 = (
    "5660e3ae657779c4a2444cc4123848b86a4e2669aa7fea5dd5972532fc126b52"
)
TARGET_POOL_OFFSET = 1200
TARGET_POOL_COUNT = 1200
TARGET_POOL_FIRST = 4671
TARGET_POOL_LAST = 7859
TARGET_POOL_SHA256 = (
    "8fb09e2ceda0ad576dcb4771cbdbf757810cfd90969e3e8b4283d9582495136f"
)
ELIGIBLE_TOKEN_COUNT = 41518
MIN_GEOMETRIC_COUNT = 960
SEED = 0
EXPECTED_NATIVE_RUNTIME = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_bf16_reduced_precision_reduction": True,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
    "autocast_enabled": False,
}

PREFLIGHT_SCHEMA = "hlm5-e7b-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e7b-3b-execution-receipt-v1"
ATTEMPT_SCHEMA = "hlm5-e7b-3b-attempt-v1"
ADMISSION_SCHEMA = "hlm5-e7b-3b-admission-v1"
RESULT_SCHEMA = "hlm5-e7b-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e7b-3b-verdict-v1"

DEPENDENCY_SHA256 = {
    "hlm5/__init__.py": (
        "6122cf699b74d696a62fc95ee46b3bcde76ae9fd6e4bccda8a17fc55609d0e57"
    ),
    "hlm5/memory.py": (
        "91cb50810a99079e77056291a775a878483ca6c5ddb05f2c671623c8887db973"
    ),
    "hlm5/model.py": (
        "2f810bde52bbf3e0da9f52640bf36ca924cd665bd6a7036155f0581494b8857c"
    ),
    "hlm5/io.py": (
        "5886ce8331ba42ca0ee75dd67746665be58d174e5f3416ad661f81c7d7685439"
    ),
    "hlm5/e7_contract.py": (
        "5b7717f5da089db8b1d9d1d611251a2d01ef8fcbf5a719836692abde4a0fc74d"
    ),
    "hlm5/e7_runtime.py": (
        "c56bda88db7173d11e81c7705f31ef5439b3d9b3b7946117e56578d9059917cc"
    ),
    "hlm5/certify.py": (
        "a13544eb242cb1f827496a46a6d943723ad51f35b827fe47d2da641e26a1c210"
    ),
    "hlm5/edit_audit.py": (
        "930d51619bb240e13a15412f6db92b0cf58a87e17a405dc7110bbaaa2e8e8930"
    ),
    "hlm5/key_value.py": (
        "a0bacdfcba80397bfacefa7c7762bcb202fb91363da0e6fa902f9c707c70140a"
    ),
    "hlm5/public_adapter.py": (
        "9a7f77976ef604d318f42821e36491a13a156ae7e00af49a255ca2fec90f650d"
    ),
    "scripts/run_e7_3b.py": (
        "48ba22bfc1e6a0824a7f659bd945e028e8d1eb5999b07c21d018a74e6a97d10e"
    ),
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/e7b_contract.py",
    "hlm5/e7b_runtime.py",
    "scripts/preflight_e7b_3b.py",
    "scripts/register_e7b_3b.py",
    "scripts/admit_e7b_3b.py",
    "scripts/replay_e7b_3b.py",
    "scripts/verify_e7b_3b.py",
    "tests/test_e7b_3b.py",
)
REGISTERED_COMMANDS = (
    "python scripts/preflight_e7b_3b.py",
    "python scripts/register_e7b_3b.py",
    "python scripts/admit_e7b_3b.py",
    "python scripts/replay_e7b_3b.py",
    "python scripts/verify_e7b_3b.py",
)

STAGING_DIR = RESULTS_DIR / "e7b_3b_staging"
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
        raise ValueError(f"E7b output must stay under {staging}: {resolved}")
    return resolved


def atomic_create_json(path: Path, value: Any) -> None:
    """Atomically publish a complete JSON file without overwrite semantics."""
    resolved = _require_staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
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
            raise FileExistsError(f"E7b output already exists: {resolved}") from error
    finally:
        if temp_path.exists():
            temp_path.unlink()


def assert_outputs_absent(paths: Iterable[Path]) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise FileExistsError(f"registered E7b outputs already exist: {present}")


def scientific_sha256(scientific: dict[str, Any]) -> str:
    return stable_json_sha256(scientific)


def verify_scientific_hash(payload: dict[str, Any], label: str) -> None:
    scientific = payload.get("scientific")
    if not isinstance(scientific, dict):
        raise RuntimeError(f"{label} has no scientific object")
    expected = scientific_sha256(scientific)
    if payload.get("scientific_sha256") != expected:
        raise RuntimeError(f"{label} scientific hash mismatch")


def verify_protocol() -> None:
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("registered E7b protocol commit is unavailable")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    require_file_hash(
        REPO_ROOT / PROTOCOL_PATH,
        PROTOCOL_SHA256,
        "registered E7b protocol",
    )


def verify_dependencies() -> dict[str, str]:
    for relative, expected in DEPENDENCY_SHA256.items():
        require_file_hash(REPO_ROOT / relative, expected, f"E7b dependency {relative}")
    return dict(DEPENDENCY_SHA256)


def verify_e7_inputs() -> dict[str, Any]:
    result_path = REPO_ROOT / E7_RESULT_PATH
    verdict_path = REPO_ROOT / E7_VERDICT_PATH
    require_file_hash(result_path, E7_RESULT_SHA256, "bound E7 result")
    require_file_hash(verdict_path, E7_VERDICT_SHA256, "bound E7 verdict")
    result = load_json_object(result_path)
    verdict = load_json_object(verdict_path)
    if verdict.get("verdict") != "FAIL_3B_PORTABILITY" or verdict.get(
        "validity_failures"
    ) != []:
        raise RuntimeError("bound E7 verdict is not the implementation-valid failure")
    rows = result.get("direct_certified_injection", {}).get("rows", [])
    matches = [row for row in rows if row.get("target_id") == E7_SENTINEL["target_id"]]
    if len(matches) != 1:
        raise RuntimeError("bound E7 sentinel is absent or duplicated")
    row = matches[0]
    observed = {
        "target_id": row.get("target_id"),
        "beta": row.get("beta"),
        "float64_margin": row.get("float64_margin"),
        "native_margin": row.get("ordinary_bfloat16_margin"),
        "native_prediction": row.get("ordinary_head_prediction"),
        "native_logits_sha256": row.get("ordinary_logits_sha256"),
    }
    if observed != E7_SENTINEL:
        raise RuntimeError(f"bound E7 sentinel mismatch: {observed}")
    rule_decision = native_decision(
        target_id=observed["target_id"],
        prediction=observed["native_prediction"],
        native_margin=observed["native_margin"],
    )
    if rule_decision != REFUSE_TIE:
        raise RuntimeError("E7b rule did not refuse the spent E7 tie")
    return {
        "result_sha256": E7_RESULT_SHA256,
        "verdict_sha256": E7_VERDICT_SHA256,
        "verdict": verdict["verdict"],
        "validity_failures": [],
        "sentinel": {**observed, "e7b_rule_decision": rule_decision},
    }


def eligible_token_ids(tokenizer: Any) -> list[int]:
    eligible: list[int] = []
    for token, token_id in tokenizer.get_vocab().items():
        text = token.replace("Ġ", " ")
        if text.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", text.strip()):
            eligible.append(int(token_id))
    eligible.sort()
    return eligible


def target_pools(tokenizer: Any) -> tuple[list[int], list[int]]:
    eligible = eligible_token_ids(tokenizer)
    if len(eligible) != ELIGIBLE_TOKEN_COUNT:
        raise RuntimeError(f"E7b eligible-token count mismatch: {len(eligible)}")
    e7_pool = eligible[:E7_POOL_COUNT]
    fresh = eligible[TARGET_POOL_OFFSET : TARGET_POOL_OFFSET + TARGET_POOL_COUNT]
    if stable_json_sha256(e7_pool) != E7_POOL_SHA256:
        raise RuntimeError("E7 source target-pool hash mismatch")
    if (
        len(fresh) != TARGET_POOL_COUNT
        or fresh[0] != TARGET_POOL_FIRST
        or fresh[-1] != TARGET_POOL_LAST
        or stable_json_sha256(fresh) != TARGET_POOL_SHA256
        or set(e7_pool) & set(fresh)
    ):
        raise RuntimeError("registered disjoint E7b target pool mismatch")
    return e7_pool, fresh


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
    missing = [
        relative
        for relative in REGISTERED_SOURCE_PATHS
        if not (REPO_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"registered E7b sources missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError("registered E7b sources must be committed:\n" + dirty)
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in REGISTERED_SOURCE_PATHS
    }


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    current = registered_source_sha256()
    if receipt.get("source_sha256") != current:
        raise RuntimeError("registered E7b source drift")
    if receipt.get("dependency_sha256") != DEPENDENCY_SHA256:
        raise RuntimeError("registered E7b dependency receipt mismatch")


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E7b preflight")
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
        "e7_target_pool_count": E7_POOL_COUNT,
        "e7_target_pool_sha256": E7_POOL_SHA256,
        "target_pool_count": TARGET_POOL_COUNT,
        "target_pool_first": TARGET_POOL_FIRST,
        "target_pool_last": TARGET_POOL_LAST,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "registered_target_pool_count": TARGET_POOL_COUNT,
        "registered_target_pool_first": TARGET_POOL_FIRST,
        "registered_target_pool_last": TARGET_POOL_LAST,
        "registered_target_pool_sha256": TARGET_POOL_SHA256,
        "registered_e7_pool_sha256": E7_POOL_SHA256,
        "target_pool_overlap_with_e7": 0,
        "candidate_outputs_computed": False,
        "commands": list(REGISTERED_COMMANDS),
        "e7_inputs": verify_e7_inputs(),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7b preflight field mismatch: {name}")
    if payload.get("model_invariants") != expected_model_invariants():
        raise RuntimeError("E7b preflight model invariants mismatch")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7b preflight native arithmetic mismatch")
    identity = payload.get("baseline_only_head_identity", {})
    if (
        identity.get("prompt_sha256") != BASELINE_ANCHOR_SHA256
        or identity.get("argmax_equal") is not True
        or identity.get("native_argmax") != identity.get("independent_argmax")
        or identity.get("max_abs_logit_difference", math.inf)
        > HEAD_IDENTITY_MAX_ABS
        or identity.get("hidden_dtype") != MODEL_FORWARD_DTYPE
    ):
        raise RuntimeError("E7b preflight head identity failed")
    _verify_source_receipt(payload)
    verify_protocol()
    verify_dependencies()
    verify_snapshot(snapshot_path())
    return payload, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha = load_preflight()
    if not EXECUTION_RECEIPT_PATH.is_file():
        raise FileNotFoundError("missing E7b execution receipt")
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
        "registered_commands": list(REGISTERED_COMMANDS),
        "candidate_outputs_computed_before_receipt": False,
        "pre_attempt_outputs_absent": True,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7b execution receipt field mismatch: {name}")
    if payload.get("tests_returncode") != 0:
        raise RuntimeError("registered E7b tests did not pass")
    if payload.get("environment") != preflight.get("environment"):
        raise RuntimeError("E7b execution environment differs from preflight")
    if payload.get("native_runtime") != preflight.get("native_runtime"):
        raise RuntimeError("E7b native runtime differs from preflight")
    _verify_source_receipt(payload)
    return payload, sha256_file(EXECUTION_RECEIPT_PATH)


def load_attempt() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    if not ATTEMPT_PATH.is_file():
        raise FileNotFoundError("missing E7b burn marker")
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
            raise RuntimeError(f"E7b attempt field mismatch: {name}")
    return payload, sha256_file(ATTEMPT_PATH)


def load_admission() -> tuple[dict[str, Any], str]:
    receipt, execution_sha = load_execution_receipt()
    _attempt, attempt_sha = load_attempt()
    if not ADMISSION_PATH.is_file():
        raise FileNotFoundError("missing E7b admission")
    payload = load_json_object(ADMISSION_PATH)
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("E7b admission schema or status mismatch")
    verify_admission_provenance(
        payload,
        receipt=receipt,
        execution_sha=execution_sha,
        attempt_sha=attempt_sha,
    )
    verify_scientific_hash(payload, "E7b admission")
    failures = admission_validity_failures(payload)
    if failures:
        raise RuntimeError("invalid E7b admission: " + "; ".join(failures))
    return payload, sha256_file(ADMISSION_PATH)


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
            raise RuntimeError(f"E7b admission provenance mismatch: {name}")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7b admission native arithmetic is not registered")


def _expected_batch_fields(position: int, total: int) -> tuple[int, int, int]:
    start = (position // HEAD_BATCH) * HEAD_BATCH
    return start, min(HEAD_BATCH, total - start), position - start


def admission_validity_failures(payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != ADMISSION_SCHEMA or payload.get("status") != "COMPLETE":
        failures.append("admission schema or status mismatch")
    try:
        verify_scientific_hash(payload, "E7b admission")
    except Exception as error:
        failures.append(str(error))
        return failures
    scientific = payload["scientific"]
    if scientific.get("prompt") != PROMPT or scientific.get(
        "prompt_sha256"
    ) != PROMPT_SHA256:
        failures.append("admission prompt mismatch")
    if scientific.get("target_pool_sha256") != TARGET_POOL_SHA256:
        failures.append("admission target-pool hash mismatch")
    if scientific.get("target_pool_count") != TARGET_POOL_COUNT:
        failures.append("admission target-pool count mismatch")
    if scientific.get("model_invariants") != expected_model_invariants():
        failures.append("admission model invariants mismatch")
    if scientific.get("e7_sentinel") != verify_e7_inputs()["sentinel"]:
        failures.append("admission E7 sentinel regression mismatch")
    boundary = scientific.get("certificate_boundary_dtypes")
    if boundary != {
        "head": "float64",
        "hidden": "float64",
        "directions": "float64",
        "slopes": "float64",
    }:
        failures.append("admission certificate boundary mismatch")

    rows = scientific.get("rows")
    if not isinstance(rows, list) or len(rows) != TARGET_POOL_COUNT:
        failures.append("admission row count mismatch")
        return failures
    target_ids = [row.get("target_id") for row in rows]
    if stable_json_sha256(target_ids) != TARGET_POOL_SHA256:
        failures.append("admission row target ids mismatch")

    reachable: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("pool_index") != index:
            failures.append(f"admission row {index} pool index mismatch")
        geometry_class = row.get("geometry_class")
        decision = row.get("decision")
        lower = row.get("L")
        upper = row.get("U")
        if not isinstance(lower, (int, float)) or not math.isfinite(float(lower)):
            failures.append(f"admission row {index} invalid lower bound")
            continue
        if upper is not None and (
            not isinstance(upper, (int, float)) or not math.isfinite(float(upper))
        ):
            failures.append(f"admission row {index} invalid upper bound")
        if geometry_class == REFUSE_HARD:
            if decision != REFUSE_HARD or row.get("hard_blocker") is not True:
                failures.append(f"admission row {index} hard-refusal mismatch")
            if row.get("beta") is not None or row.get("native_checked") is not None:
                failures.append(f"admission row {index} hard refusal was evaluated")
        elif geometry_class == REFUSE_INTERVAL:
            upper_value = math.inf if upper is None else float(upper)
            if (
                decision != REFUSE_INTERVAL
                or row.get("hard_blocker") is not False
                or float(lower) < upper_value
            ):
                failures.append(f"admission row {index} interval-refusal mismatch")
            if row.get("beta") is not None or row.get("native_checked") is not None:
                failures.append(f"admission row {index} interval refusal was evaluated")
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
                failures.append(f"admission row {index} invalid geometric reachability")
            if row.get("native_checked") is not True:
                failures.append(f"admission row {index} lacks native evidence")
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
                failures.append(f"admission row {index} invalid native scalars")
            else:
                recomputed_margin = float(
                    torch.tensor(float(target_logit), dtype=torch.float32)
                    - torch.tensor(float(competitor_logit), dtype=torch.float32)
                )
                if recomputed_margin != float(native_margin):
                    failures.append(f"admission row {index} native margin mismatch")
            try:
                expected_decision = native_decision(
                    target_id=row["target_id"],
                    prediction=int(prediction),
                    native_margin=float(native_margin),
                )
            except Exception:
                expected_decision = None
            if decision != expected_decision:
                failures.append(f"admission row {index} native decision mismatch")
            if row.get("native_logits_dtype") != "bfloat16" or row.get(
                "native_logits_shape"
            ) != [MODEL_VOCAB_SIZE]:
                failures.append(f"admission row {index} native tensor mismatch")
            native_hash = row.get("native_logits_sha256")
            if not isinstance(native_hash, str) or re.fullmatch(
                r"[0-9a-f]{64}", native_hash
            ) is None:
                failures.append(f"admission row {index} missing native hash")
            competitor_id = row.get("native_competitor_id")
            if (
                not isinstance(competitor_id, int)
                or competitor_id == row.get("target_id")
                or not 0 <= competitor_id < MODEL_VOCAB_SIZE
            ):
                failures.append(f"admission row {index} competitor mismatch")
        else:
            failures.append(f"admission row {index} unknown geometry class")

    total_reachable = len(reachable)
    for position, row in enumerate(reachable):
        start, size, offset = _expected_batch_fields(position, total_reachable)
        if (
            row.get("native_batch_start") != start
            or row.get("native_batch_size") != size
            or row.get("native_batch_offset") != offset
        ):
            failures.append(f"admission reachable row {position} batch mismatch")
    try:
        expected_summary = summarize_rows(rows)
    except Exception as error:
        failures.append(f"admission summary exception: {error}")
    else:
        if scientific.get("summary") != expected_summary:
            failures.append("admission summary mismatch")
    return failures


def load_result() -> tuple[dict[str, Any], str]:
    receipt, _execution_sha = load_execution_receipt()
    admission, admission_sha = load_admission()
    _attempt, attempt_sha = load_attempt()
    if not RESULT_PATH.is_file():
        raise FileNotFoundError("missing E7b replay result")
    payload = load_json_object(RESULT_PATH)
    expected = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RuntimeError(f"E7b result field mismatch: {name}")
    verify_result_provenance(
        payload,
        receipt=receipt,
        admission=admission,
        admission_sha=admission_sha,
        attempt_sha=attempt_sha,
    )
    verify_scientific_hash(payload, "E7b result")
    failures = result_validity_failures(payload, admission)
    if failures:
        raise RuntimeError("invalid E7b result: " + "; ".join(failures))
    return payload, sha256_file(RESULT_PATH)


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
            raise RuntimeError(f"E7b result provenance mismatch: {name}")
    if payload.get("native_runtime") != EXPECTED_NATIVE_RUNTIME:
        raise RuntimeError("E7b result native arithmetic is not registered")


def result_validity_failures(
    result: dict[str, Any],
    admission: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    try:
        verify_scientific_hash(result, "E7b result")
    except Exception as error:
        failures.append(str(error))
        return failures
    scientific = result["scientific"]
    admission_scientific = admission["scientific"]
    if scientific.get("prompt_sha256") != PROMPT_SHA256:
        failures.append("result prompt mismatch")
    if scientific.get("target_pool_sha256") != TARGET_POOL_SHA256:
        failures.append("result target-pool mismatch")
    if scientific.get("target_pool_count") != TARGET_POOL_COUNT:
        failures.append("result target-pool count mismatch")
    if scientific.get("model_invariants") != admission_scientific.get(
        "model_invariants"
    ):
        failures.append("result model invariants differ from admission")
    if scientific.get("admission_scientific_sha256") != admission.get(
        "scientific_sha256"
    ):
        failures.append("result scientific object binds a different admission")
    required_true = (
        "geometry_all_rows_match",
        "native_all_decisions_match",
        "admitted_native_values_match",
        "admitted_native_hashes_match",
    )
    for name in required_true:
        if scientific.get(name) is not True:
            failures.append(f"result replay bar failed: {name}")
    replay_rows = scientific.get("rows")
    admission_rows = admission_scientific.get("rows")
    if replay_rows != admission_rows:
        failures.append("new-process replay rows differ from admission")
    if scientific.get("summary") != admission_scientific.get("summary"):
        failures.append("new-process replay summary differs from admission")
    return failures


def scientific_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    summary = result.get("scientific", {}).get("summary", {})
    geometric = summary.get("geometrically_reachable")
    admitted = summary.get("native_admitted")
    if not isinstance(geometric, int) or geometric < MIN_GEOMETRIC_COUNT:
        failures.append(
            f"geometrically reachable count is below {MIN_GEOMETRIC_COUNT}"
        )
    if not isinstance(admitted, int) or admitted < 1:
        failures.append("native admission is vacuous")
    if isinstance(geometric, int) and isinstance(admitted, int):
        if 100 * admitted < 99 * geometric:
            failures.append("native admission retained less than 99% of geometry")
    rows = result.get("scientific", {}).get("rows", [])
    admitted_rows = [row for row in rows if row.get("decision") == NATIVE_ADMIT]
    if any(
        row.get("native_prediction") != row.get("target_id")
        or not isinstance(row.get("native_margin"), (int, float))
        or row["native_margin"] <= 0
        for row in admitted_rows
    ):
        failures.append("an admitted row lacks a strict native target win")
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
        "summary": None if result is None else result.get("scientific", {}).get("summary"),
        "validity_failures": validity_failures,
        "scientific_failures": scientific_failures_value,
        "claim_boundary": (
            "Pinned SmolLM3-3B-Base revision and registered CUDA/BF16 runtime; "
            "exact-sign geometry plus observed strict native-head admission on the "
            "registered disjoint single-token pool."
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
