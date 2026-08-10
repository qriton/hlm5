"""Fail-closed provenance and sample helpers for the frozen-3B E7 study."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from .io import REPO_ROOT, RESULTS_DIR


PROTOCOL_COMMIT = "6a73bab5a37622b14fb30800fd92f298c479d781"
PROTOCOL_PATH = "docs/e7-3b-portability-protocol-2026-08-10.md"
PROTOCOL_SHA256 = (
    "979a9b9abcba1420c7ed9253ad7f6d2353b0bccc499a486b45319f29cbd43be0"
)
MODEL_ID = "HuggingFaceTB/SmolLM3-3B-Base"
MODEL_REVISION = "d78a42f79198603e614095753484a04c10c2b940"
MODEL_PARAMETER_COUNT = 3_075_098_624
MODEL_HIDDEN_SIZE = 2048
MODEL_VOCAB_SIZE = 128256
MODEL_LAYERS = 36
MODEL_FORWARD_DTYPE = "bfloat16"
CERTIFICATE_DTYPE = "float64"
GATE_DTYPE = "float32"
TARGET_POOL_COUNT = 1200
TARGET_POOL_SHA256 = (
    "5660e3ae657779c4a2444cc4123848b86a4e2669aa7fea5dd5972532fc126b52"
)
HEAD_IDENTITY_MAX_ABS = 0.125
GATE_THRESHOLD = 0.95
TEMPERATURE = 0.10
DEGREE = 5
SEED = 0
CONTRACT_SCHEMA = "hlm5-e7-3b-contract-v1"
PREFLIGHT_SCHEMA = "hlm5-e7-3b-preflight-v1"
EXECUTION_SCHEMA = "hlm5-e7-3b-execution-receipt-v1"
RESULT_SCHEMA = "hlm5-e7-3b-result-v1"
VERDICT_SCHEMA = "hlm5-e7-3b-verdict-v1"

MODEL_FILE_SHA256 = {
    "config.json": "e8336e843aecb733691a043ad4209168e1730158b5e03c2672591cc946036a47",
    "generation_config.json": (
        "7d830c1a274c4484a50eb34ed68cdbd3611a4981fd4d539c7ec1fed67e01cec4"
    ),
    "model.safetensors.index.json": (
        "e3a7254c086c4a78f95ad5864f435d165ffed1ed8998326ce5107b4423c15b76"
    ),
    "special_tokens_map.json": (
        "9faaa15579bbf433176493a621142ff580ae2d13d81bf4702cb72f4d6902dc62"
    ),
    "tokenizer_config.json": (
        "f398d9edb0184aac1c9fd1b1bdcd3a5a1527ae560fd847719635a66e2b54e9cd"
    ),
    "tokenizer.json": (
        "ab4da6b2aa68247e9c0fa9b97fc7fcc796505038d01f7e144522a65ce0dbd2e5"
    ),
    "model-00001-of-00002.safetensors": (
        "7e270ac568ee1880ddbadad66ccdcd9906d52415e8904e2f300c75250b9c7d49"
    ),
    "model-00002-of-00002.safetensors": (
        "c6a6e7690a66dcc386a6a9b456686e7c308d45cba884d116c928dafa7fa987ae"
    ),
}
MODEL_WEIGHT_BYTES = {
    "model-00001-of-00002.safetensors": 4_966_315_264,
    "model-00002-of-00002.safetensors": 1_183_919_744,
}
SOURCE_INPUT_SHA256 = {
    "scripts/run_1b_faithful_certdosed.py": (
        "7ba4b2d8386df6e51f508f5af55caa68d1ab5fb3689356139d3d30db585a523b"
    ),
    "scripts/run_1b_envelope_multikey.py": (
        "733ccd0ea1241ede9198e3e9c9c274984bff1ea4f74f6985c639ce99c6126f40"
    ),
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/__init__.py",
    "hlm5/certify.py",
    "hlm5/e7_contract.py",
    "hlm5/e7_runtime.py",
    "hlm5/memory.py",
    "hlm5/public_adapter.py",
    "scripts/preflight_e7_3b.py",
    "scripts/run_e7_3b.py",
    "scripts/verify_e7_3b.py",
    "slurm/e7_3b_leonardo.sbatch",
    "tests/test_e7_3b.py",
)
REGISTERED_COMMANDS = (
    "python scripts/preflight_e7_3b.py",
    "python scripts/run_e7_3b.py",
    "python scripts/verify_e7_3b.py",
)

STAGING_DIR = RESULTS_DIR / "e7_3b_staging"
PREFLIGHT_PATH = STAGING_DIR / "preflight.json"
EXECUTION_RECEIPT_PATH = STAGING_DIR / "execution_receipt.json"
RESULT_PATH = STAGING_DIR / "result.json"
VERDICT_PATH = STAGING_DIR / "verdict.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _require_staging_path(path: Path) -> Path:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"E7 output must stay under {staging}: {resolved}")
    return resolved


def atomic_write_json(path: Path, value: Any) -> None:
    resolved = _require_staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        dir=resolved.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, resolved)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: got {actual}, expected {expected}"
        )
    return actual


def snapshot_path() -> Path:
    override = os.environ.get("HLM5_E7_SNAPSHOT")
    if override:
        return Path(override).expanduser().resolve()
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    return (
        hf_home
        / "hub"
        / "models--HuggingFaceTB--SmolLM3-3B-Base"
        / "snapshots"
        / MODEL_REVISION
    ).resolve()


def verify_snapshot(path: Path | None = None) -> dict[str, dict[str, Any]]:
    root = snapshot_path() if path is None else path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pinned E7 snapshot is not cached: {root}")
    records: dict[str, dict[str, Any]] = {}
    for name, expected in MODEL_FILE_SHA256.items():
        file_path = root / name
        actual = require_file_hash(file_path, expected, f"E7 model file {name}")
        size = file_path.stat().st_size
        expected_size = MODEL_WEIGHT_BYTES.get(name)
        if expected_size is not None and size != expected_size:
            raise RuntimeError(
                f"E7 model file size mismatch for {name}: {size} != {expected_size}"
            )
        records[name] = {
            "sha256": actual,
            "bytes": size,
        }
    return records


def registered_source_sha256(*, require_clean: bool = False) -> dict[str, str]:
    missing = [
        relative
        for relative in REGISTERED_SOURCE_PATHS
        if not (REPO_ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"registered E7 sources missing: {missing}")
    if require_clean:
        dirty = git_output("status", "--porcelain", "--", *REGISTERED_SOURCE_PATHS)
        if dirty:
            raise RuntimeError(
                "registered E7 implementation must be committed:\n" + dirty
            )
    return {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in REGISTERED_SOURCE_PATHS
    }


def verify_protocol() -> None:
    if git_output("rev-parse", PROTOCOL_COMMIT) != PROTOCOL_COMMIT:
        raise RuntimeError("registered E7 protocol commit is unavailable")
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
        "registered E7 protocol",
    )


def _assignment_values(
    relative: str,
    names: tuple[str, ...],
) -> dict[str, Any]:
    path = REPO_ROOT / relative
    require_file_hash(path, SOURCE_INPUT_SHA256[relative], relative)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    statements: list[ast.stmt] = list(tree.body)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            statements.extend(node.body)
    allowed = set(names)
    environment: dict[str, Any] = {}
    for statement in statements:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in allowed:
            continue
        expression = ast.Expression(statement.value)
        value = eval(  # noqa: S307 - pinned local literals, no builtins
            compile(expression, str(path), "eval"),
            {"__builtins__": {}},
            dict(environment),
        )
        environment[target.id] = value
    missing = [name for name in names if name not in environment]
    if missing:
        raise RuntimeError(f"could not extract registered constants {missing} from {path}")
    return {name: environment[name] for name in names}


def registered_samples() -> dict[str, Any]:
    faithful = _assignment_values(
        "scripts/run_1b_faithful_certdosed.py",
        ("FACTS", "NEUTRAL", "WHITEN_TEXT"),
    )
    envelope = _assignment_values(
        "scripts/run_1b_envelope_multikey.py",
        ("ORIG_FACT", "NOVEL", "REAL", "GENERIC"),
    )
    return {**faithful, **envelope}


def target_pool(tokenizer: Any, *, require_registered: bool = True) -> list[int]:
    pool: list[int] = []
    for token, token_id in tokenizer.get_vocab().items():
        text = token.replace("Ġ", " ")
        if text.startswith(" ") and re.fullmatch(r"[A-Za-z]{3,}", text.strip()):
            pool.append(int(token_id))
    pool = sorted(pool)[:TARGET_POOL_COUNT]
    digest = stable_json_sha256(pool)
    if require_registered and (
        len(pool) != TARGET_POOL_COUNT or digest != TARGET_POOL_SHA256
    ):
        raise RuntimeError(
            "registered target pool mismatch: "
            f"count={len(pool)}, sha256={digest}"
        )
    return pool


def float64_boundary_record(**tensors: torch.Tensor) -> dict[str, str]:
    if not tensors:
        raise ValueError("at least one certificate tensor is required")
    record = {
        name: str(tensor.dtype).removeprefix("torch.")
        for name, tensor in tensors.items()
    }
    wrong = {name: dtype for name, dtype in record.items() if dtype != "float64"}
    if wrong:
        raise RuntimeError(f"E7 certificate boundary is not float64: {wrong}")
    return record


def _verify_source_receipt(receipt: dict[str, Any]) -> None:
    recorded = receipt.get("source_sha256")
    current = registered_source_sha256()
    if recorded != current:
        raise RuntimeError(
            f"registered E7 source drift: recorded={recorded}, current={current}"
        )


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError("missing E7 preflight receipt")
    preflight = load_json_object(PREFLIGHT_PATH)
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError("E7 preflight schema mismatch")
    if preflight.get("status") != "READY":
        raise RuntimeError(f"E7 preflight is not READY: {preflight.get('status')}")
    expected_identity = {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_FILE_SHA256,
        "target_pool_count": TARGET_POOL_COUNT,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "registered_target_pool_count": TARGET_POOL_COUNT,
        "registered_target_pool_sha256": TARGET_POOL_SHA256,
        "sample_sha256": stable_json_sha256(registered_samples()),
        "candidate_outputs_computed": False,
        "commands": list(REGISTERED_COMMANDS),
    }
    for name, expected in expected_identity.items():
        if preflight.get(name) != expected:
            raise RuntimeError(f"E7 preflight field mismatch: {name}")
    invariants = preflight.get("model_invariants", {})
    if invariants != {
        "class_name": "SmolLM3ForCausalLM",
        "parameter_count": MODEL_PARAMETER_COUNT,
        "hidden_size": MODEL_HIDDEN_SIZE,
        "vocab_size": MODEL_VOCAB_SIZE,
        "layers": MODEL_LAYERS,
        "tied_embeddings": True,
        "head_bias_is_none": True,
        "all_parameters_frozen": True,
        "floating_parameter_dtypes": [MODEL_FORWARD_DTYPE],
    }:
        raise RuntimeError("E7 preflight model invariants do not match")
    head_identity = preflight.get("baseline_only_head_identity", {})
    if (
        head_identity.get("argmax_equal") is not True
        or head_identity.get("native_argmax")
        != head_identity.get("independent_argmax")
        or head_identity.get("max_abs_logit_difference", math.inf)
        > HEAD_IDENTITY_MAX_ABS
        or head_identity.get("hidden_dtype") != MODEL_FORWARD_DTYPE
    ):
        raise RuntimeError("E7 preflight baseline head identity failed")
    _verify_source_receipt(preflight)
    verify_protocol()
    verify_snapshot()
    return preflight, sha256_file(PREFLIGHT_PATH)


def load_execution_receipt() -> tuple[dict[str, Any], str]:
    preflight, preflight_sha256 = load_preflight()
    if not EXECUTION_RECEIPT_PATH.is_file():
        raise FileNotFoundError("missing E7 execution receipt")
    receipt = load_json_object(EXECUTION_RECEIPT_PATH)
    if receipt.get("schema") != EXECUTION_SCHEMA:
        raise RuntimeError("E7 execution receipt schema mismatch")
    if receipt.get("status") != "READY":
        raise RuntimeError("E7 execution receipt is not READY")
    if receipt.get("preflight_sha256") != preflight_sha256:
        raise RuntimeError("E7 execution receipt binds a different preflight")
    if receipt.get("implementation_commit") != preflight.get(
        "implementation_commit"
    ):
        raise RuntimeError("E7 receipt implementation commit mismatch")
    if receipt.get("protocol_commit") != PROTOCOL_COMMIT or receipt.get(
        "protocol_sha256"
    ) != PROTOCOL_SHA256:
        raise RuntimeError("E7 execution receipt protocol mismatch")
    if receipt.get("model_file_sha256") != MODEL_FILE_SHA256:
        raise RuntimeError("E7 execution receipt model identity mismatch")
    if receipt.get("sample_sha256") != preflight.get("sample_sha256"):
        raise RuntimeError("E7 execution receipt sample mismatch")
    if receipt.get("target_pool_sha256") != TARGET_POOL_SHA256:
        raise RuntimeError("E7 execution receipt target-pool mismatch")
    if receipt.get("registered_commands") != list(REGISTERED_COMMANDS):
        raise RuntimeError("E7 execution receipt command mismatch")
    if receipt.get("tests_returncode") != 0:
        raise RuntimeError("E7 registered tests did not pass")
    if receipt.get("candidate_outputs_computed_before_receipt") is not False:
        raise RuntimeError("E7 execution receipt was not pre-outcome")
    if receipt.get("environment") != preflight.get("environment"):
        raise RuntimeError("E7 receipt environment differs from preflight")
    _verify_source_receipt(receipt)
    return receipt, sha256_file(EXECUTION_RECEIPT_PATH)


def result_contract(producer: Path | str) -> dict[str, Any]:
    preflight, preflight_sha256 = load_preflight()
    receipt, execution_sha256 = load_execution_receipt()
    producer_path = Path(producer).resolve()
    try:
        producer_relative = producer_path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"producer is outside repository: {producer_path}") from error
    expected_producer_sha = preflight["source_sha256"].get(producer_relative)
    actual_producer_sha = sha256_file(producer_path)
    if expected_producer_sha != actual_producer_sha:
        raise RuntimeError("E7 producer changed after preflight")
    return {
        "schema": CONTRACT_SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": preflight["implementation_commit"],
        "producer": producer_relative,
        "producer_sha256": actual_producer_sha,
        "preflight_sha256": preflight_sha256,
        "execution_receipt_sha256": execution_sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "model_forward_dtype": MODEL_FORWARD_DTYPE,
        "certificate_arithmetic_dtype": CERTIFICATE_DTYPE,
        "gate_arithmetic_dtype": GATE_DTYPE,
        "target_pool_sha256": TARGET_POOL_SHA256,
        "seed": SEED,
        "torch_version": preflight["environment"]["torch"],
        "transformers_version": preflight["environment"]["transformers"],
        "upstream_sha256": {
            "preflight": preflight_sha256,
            "execution_receipt": execution_sha256,
        },
        "commands": list(REGISTERED_COMMANDS),
        "receipt_status": receipt["status"],
    }


def result_validity_failures(result: dict[str, Any]) -> list[str]:
    """Return binding E7 implementation-contract failures."""
    failures: list[str] = []
    if result.get("schema") != RESULT_SCHEMA:
        failures.append("result schema mismatch")
    if result.get("status") != "COMPLETE":
        failures.append("result is not COMPLETE")
    if result.get("sample_sha256") != stable_json_sha256(registered_samples()):
        failures.append("result sample hash mismatch")
    expected_model = {
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
    if result.get("model_invariants") != expected_model:
        failures.append("result model invariants mismatch")
    envelope = result.get("envelope", {})
    if envelope.get("n_keys") != 60:
        failures.append("envelope did not evaluate all 60 keys")
    if envelope.get("pool") != TARGET_POOL_COUNT:
        failures.append("envelope target pool is incomplete")
    if envelope.get("target_pool_sha256") != TARGET_POOL_SHA256:
        failures.append("envelope target pool hash mismatch")
    key_rows = envelope.get("keys", [])
    if len(key_rows) != 60:
        failures.append("envelope key-row count mismatch")
    if any(
        sum(row.get("risk_counts", {}).values()) != TARGET_POOL_COUNT
        for row in key_rows
    ):
        failures.append("an envelope risk partition does not cover the pool")
    scalar = envelope.get("original_scalar_vector_check", {})
    if scalar.get("targets_checked") != TARGET_POOL_COUNT or not scalar.get(
        "all_match"
    ):
        failures.append("original-key scalar/vector envelope mismatch")

    validity = result.get("validity", {})
    required_true = (
        "model_invariants_match",
        "all_envelope_candidate_doses_inside",
        "all_envelope_candidate_margins_positive",
        "original_scalar_vector_match",
        "all_risk_counts_sum_to_pool",
        "all_direct_doses_inside_with_positive_float64_margin",
        "all_direct_rows_checked_by_ordinary_bfloat16_head",
        "all_faithful_doses_inside_with_positive_float64_margin",
        "all_gate_rows_have_required_evidence",
    )
    for name in required_true:
        if validity.get(name) is not True:
            failures.append(f"validity bar failed: {name}")
    if validity.get("keys_checked") != 60:
        failures.append("validity key count mismatch")
    if validity.get("target_pool_count") != TARGET_POOL_COUNT:
        failures.append("validity target-pool count mismatch")

    direct = result.get("direct_certified_injection", {})
    direct_rows = direct.get("rows", [])
    original_rows = [row for row in key_rows if row.get("is_original_key")]
    if len(original_rows) != 1:
        failures.append("expected exactly one original envelope row")
    elif direct.get("accepted_count") != original_rows[0].get("n_reachable"):
        failures.append("direct accepted set does not match original reachability")
    if len(direct_rows) != direct.get("accepted_count"):
        failures.append("direct row count mismatch")
    if direct.get("ordinary_head_success_count") != sum(
        bool(row.get("ordinary_head_success")) for row in direct_rows
    ):
        failures.append("direct ordinary-head success summary mismatch")
    for row in direct_rows:
        upper = row.get("U")
        if not row.get("ordinary_head_checked"):
            failures.append("direct row lacks ordinary-head evidence")
            break
        if not (
            row.get("L") < row.get("beta")
            and (upper is None or row.get("beta") < upper)
            and row.get("float64_margin", 0) > 0
        ):
            failures.append("direct row has an invalid certified dose")
            break

    dtypes = result.get("certificate_boundary_dtypes", {})
    if not dtypes or any(dtype != "float64" for dtype in dtypes.values()):
        failures.append("certificate boundary lacks float64 runtime evidence")

    faithful = result.get("faithful", {})
    facts = faithful.get("facts", [])
    if faithful.get("registered_fact_count") != 18 or len(facts) != 18:
        failures.append("faithful fact count mismatch")
    accepted_facts = [row for row in facts if row.get("admission") == "ADMIT"]
    refused_facts = [row for row in facts if row.get("admission") == "REFUSE"]
    eligible_facts = [row for row in facts if not row.get("baseline_correct")]
    if faithful.get("eligible_fact_count") != len(eligible_facts):
        failures.append("faithful eligible count mismatch")
    if faithful.get("baseline_correct_count") != len(facts) - len(eligible_facts):
        failures.append("faithful baseline-correct count mismatch")
    if len(accepted_facts) + len(refused_facts) != len(eligible_facts):
        failures.append("an eligible faithful fact lacks an admission decision")
    if faithful.get("accepted_count") != len(accepted_facts):
        failures.append("faithful accepted count mismatch")
    if faithful.get("refused_count") != len(refused_facts):
        failures.append("faithful refused count mismatch")
    for row in accepted_facts:
        certificate = row.get("certificate", {})
        upper = certificate.get("U")
        if not (
            certificate.get("L") < row.get("beta")
            and (upper is None or row.get("beta") < upper)
            and row.get("float64_margin", 0) > 0
        ):
            failures.append("faithful accepted row has an invalid dose")
            break

    gate_rows = faithful.get("gate_rows", [])
    if len(gate_rows) != 80 or validity.get("gate_rows_checked") != 80:
        failures.append("gate/locality row count mismatch")
    evidence = {
        "actual_score",
        "selected_slot",
        "gate_open",
        "baseline_logits_sha256",
        "adapted_logits_sha256",
    }
    if any(not evidence.issubset(row) for row in gate_rows):
        failures.append("gate/locality row lacks required evidence")
    for row in gate_rows:
        baseline_hash = row.get("baseline_logits_sha256", "")
        adapted_hash = row.get("adapted_logits_sha256", "")
        if not (
            isinstance(row.get("actual_score"), (int, float))
            and math.isfinite(row["actual_score"])
            and isinstance(row.get("selected_slot"), int)
            and isinstance(row.get("gate_open"), bool)
            and isinstance(baseline_hash, str)
            and len(baseline_hash) == 64
            and isinstance(adapted_hash, str)
            and len(adapted_hash) == 64
        ):
            failures.append("gate/locality row has malformed runtime evidence")
            break
        hashes_equal = baseline_hash == adapted_hash
        if bool(row.get("logits_bit_identical")) != hashes_equal:
            failures.append("gate/locality bit-identity evidence is inconsistent")
            break
    exact_rows = [row for row in gate_rows if row.get("kind") == "exact"]
    paraphrase_rows = [row for row in gate_rows if row.get("kind") == "paraphrase"]
    neutral_rows = [row for row in gate_rows if row.get("kind") == "neutral"]
    if (len(exact_rows), len(paraphrase_rows), len(neutral_rows)) != (18, 54, 8):
        failures.append("gate row kind partition mismatch")
    summary = faithful.get("summary", {})
    accepted_indices = {row["fact_index"] for row in accepted_facts}
    refused_indices = {row["fact_index"] for row in refused_facts}
    exact_accepted = [
        row for row in exact_rows if row.get("fact_index") in accepted_indices
    ]
    exact_refused = [
        row for row in exact_rows if row.get("fact_index") in refused_indices
    ]
    recomputed_summary = {
        "accepted_exact_checked": len(exact_accepted),
        "accepted_exact_gate_successes": sum(
            bool(row.get("gate_open")) for row in exact_accepted
        ),
        "accepted_exact_full_successes": sum(
            bool(row.get("target_success")) for row in exact_accepted
        ),
        "refused_exact_checked": len(exact_refused),
        "paraphrases_checked": len(paraphrase_rows),
        "neutral_checked": len(neutral_rows),
        "false_gate_applications": sum(
            bool(row.get("gate_open"))
            for row in exact_refused + paraphrase_rows + neutral_rows
        ),
        "neutral_bit_identical": sum(
            bool(row.get("logits_bit_identical")) for row in neutral_rows
        ),
    }
    for name, expected in recomputed_summary.items():
        if summary.get(name) != expected:
            failures.append(f"faithful gate summary mismatch: {name}")
    return failures


def scientific_failures(result: dict[str, Any]) -> list[str]:
    """Return failures of the frozen E7 scientific PASS bars."""
    failures: list[str] = []
    direct = result["direct_certified_injection"]
    direct_rows = direct.get("rows", [])
    if len(direct_rows) != direct.get("accepted_count") or not all(
        row.get("ordinary_head_success") is True for row in direct_rows
    ):
        failures.append("not every directly certified target succeeded")
    faithful = result["faithful"]
    facts = faithful.get("facts", [])
    accepted_facts = [row for row in facts if row.get("admission") == "ADMIT"]
    refused_facts = [row for row in facts if row.get("admission") == "REFUSE"]
    accepted = len(accepted_facts)
    if accepted < 1:
        failures.append("faithful pass would be vacuous: no eligible fact accepted")
    gate_rows = faithful.get("gate_rows", [])
    accepted_indices = {row["fact_index"] for row in accepted_facts}
    refused_indices = {row["fact_index"] for row in refused_facts}
    accepted_exact = [
        row
        for row in gate_rows
        if row.get("kind") == "exact" and row.get("fact_index") in accepted_indices
    ]
    refused_exact = [
        row
        for row in gate_rows
        if row.get("kind") == "exact" and row.get("fact_index") in refused_indices
    ]
    paraphrases = [row for row in gate_rows if row.get("kind") == "paraphrase"]
    neutrals = [row for row in gate_rows if row.get("kind") == "neutral"]
    if len(accepted_exact) != accepted:
        failures.append("not every accepted faithful exact key was checked")
    if not all(row.get("gate_open") is True for row in accepted_exact):
        failures.append("not every accepted faithful exact key opened the gate")
    if not all(row.get("target_success") is True for row in accepted_exact):
        failures.append("not every accepted faithful edit succeeded end-to-end")
    if len(refused_exact) != len(refused_facts):
        failures.append("not every refused faithful exact key was checked")
    if len(paraphrases) != 54:
        failures.append("not all registered paraphrases were checked")
    if len(neutrals) != 8:
        failures.append("not all registered neutral prompts were checked")
    if any(
        row.get("gate_open") is True
        for row in refused_exact + paraphrases + neutrals
    ):
        failures.append("an off-support query opened the gate")
    if not all(
        row.get("logits_bit_identical") is True
        and row.get("baseline_logits_sha256") == row.get("adapted_logits_sha256")
        for row in neutrals
    ):
        failures.append("a gated-off neutral output was not bit-identical")
    return failures
