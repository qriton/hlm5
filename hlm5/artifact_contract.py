"""Fail-closed provenance helpers for the exact-sign artifact refresh."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from .io import MODELS_DIR, REPO_ROOT, RESULTS_DIR


CONTRACT_SCHEMA = "hlm5-certificate-contract-v1"
PREFLIGHT_SCHEMA = "hlm5-certificate-refresh-preflight-v1"
PROTOCOL_COMMIT = "c8935f8c6cdb3dff2331758c86fcaea711232824"
PROTOCOL_PATH = "docs/certificate-exact-sign-refresh-protocol-2026-08-10.md"
PROTOCOL_SHA256 = (
    "7b1f139e2f003b30e595dffea14feb1de9537c8a1808f5518fb1640b676b0200"
)
CERTIFICATE_EPS = 0.0
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
EXPECTED_CHECKPOINT_SHA256 = (
    "3e1c94d28125c2f86f3eeca030db3610f2fa679512c29c6b6608b24fc363e0f1"
)
EXPECTED_TOKENIZER_SHA256 = (
    "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588"
)
EXPECTED_COUNTERFACT_SHA256 = (
    "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
)
GPT2_XL_MODEL_ID = "gpt2-xl"
GPT2_XL_REVISION = "15ea56dee5df4983c59b2538573817e1667135e2"
TOKENIZER_PATH = MODELS_DIR / "tokenizers" / "fineweb-65536-compat.json"
CHECKPOINT_PATH = MODELS_DIR / "hlm5_lm_baseline_fineweb_g3_final.pt"
COUNTERFACT_PATH = REPO_ROOT / "scripts" / "data" / "counterfact.json"
STAGING_DIR = RESULTS_DIR / "certificate_refresh_staging"
PREFLIGHT_PATH = STAGING_DIR / "preflight.json"
ONE_KEY_SAMPLE_CONTRACT = {
    "fact": "The capital of Vorenia is",
    "target_pool": "sorted first 1200 leading-space ASCII words length >=3",
    "synthesis_cases": "first 40 unreachable targets",
    "synthesis_steps": 250,
}
BETA_STAR_SAMPLE_CONTRACT = {
    "fact": "The capital of Vorenia is",
    "target_pool": "sorted first 1200 leading-space ASCII words length >=3",
    "dose_grid_points": 120,
    "dose_cap": "L + max(50, 5*L), additionally capped by U",
}
SYNTHESIS_FLIP_SAMPLE_CONTRACT = {
    "fact": "The capital of Vorenia is",
    "target_pool": "sorted first 1200 leading-space ASCII words length >=3",
    "synthesis_cases": "first 40 unreachable targets",
    "synthesis_steps": 300,
    "beta_grid": [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
        50000,
        100000,
    ],
}
MULTI_KEY_SAMPLE_CONTRACT = {
    "target_pool": "sorted first 1200 leading-space ASCII words length >=3",
    "key_count": 60,
    "key_categories": {"novel": 20, "real": 20, "generic": 20},
    "original_key": "The capital of Vorenia is",
}
FAITHFUL_SAMPLE_CONTRACT = {
    "facts": 17,
    "neutral_prompts": 8,
    "paraphrases_per_fact": 3,
    "gate_thresh": 0.95,
    "degree": 5,
    "temperature": 0.10,
    "synthesis_steps": 300,
    "bootstrap_resamples": 2000,
}
COUNTERFACT_SAMPLE_CONTRACT = {
    "records": 1000,
    "selection": "first 1000 records with nonempty new/true target encodings",
    "target_token": "first token of leading-space target string",
    "dose_grid_points": 120,
}
REGISTERED_SOURCE_PATHS = (
    "hlm5/artifact_contract.py",
    "hlm5/certify.py",
    "hlm5/io.py",
    "hlm5/memory.py",
    "hlm5/model.py",
    "scripts/audit_certificate_artifacts.py",
    "scripts/compare_certificate_refresh.py",
    "scripts/preflight_certificate_refresh.py",
    "scripts/run_1b_betastar.py",
    "scripts/run_1b_certificate.py",
    "scripts/run_1b_envelope_multikey.py",
    "scripts/run_1b_faithful_certdosed.py",
    "scripts/run_1b_synth_verify.py",
    "scripts/run_cert_counterfact_gpt2xl.py",
    "scripts/verify_artifacts.py",
    "tests/test_certificate_artifact_audit.py",
    "tests/test_certificate_refresh_contract.py",
    "tests/test_certify.py",
    "tests/test_smoke.py",
)


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


def reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def repo_relative(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository: {resolved}") from exc


def require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: got {actual}, expected {expected}"
        )
    return actual


def _require_staging_path(path: Path) -> Path:
    resolved = path.resolve()
    staging = STAGING_DIR.resolve()
    if resolved == staging or staging not in resolved.parents:
        raise ValueError(f"refresh output must stay under {staging}: {resolved}")
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
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
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


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = _require_staging_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{stable_json(row)}\n" for row in rows)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
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


def load_contract_jsonl(
    path: Path,
    expected_contract_sha256: str,
    *,
    require_prefix: bool = False,
) -> list[dict[str, Any]]:
    """Load contract-bound rows and reject malformed or ambiguous resumes."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    case_ids: set[Any] = set()
    usable_indices: set[Any] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            raise RuntimeError(f"blank JSONL row {line_number}; refusing resume")
        try:
            row = json.loads(raw_line, parse_constant=reject_nonfinite_json)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"malformed staged JSONL row {line_number}; refusing resume"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(
                f"staged JSONL row {line_number} is not an object; refusing resume"
            )
        if row.get("certificate_contract_sha256") != expected_contract_sha256:
            raise RuntimeError(
                f"staged row {line_number} has a different certificate contract"
            )
        case_id = row.get("case_id")
        usable_index = row.get("usable_index")
        valid_case_id = isinstance(case_id, (str, int)) and not isinstance(
            case_id,
            bool,
        )
        valid_usable_index = (
            isinstance(usable_index, int)
            and not isinstance(usable_index, bool)
            and usable_index >= 0
        )
        if not valid_case_id or not valid_usable_index:
            raise RuntimeError(
                f"staged row {line_number} lacks a valid case_id/usable_index"
            )
        if case_id in case_ids or usable_index in usable_indices:
            raise RuntimeError(
                f"duplicate case_id/usable_index in staged row {line_number}"
            )
        case_ids.add(case_id)
        usable_indices.add(usable_index)
        rows.append(row)
    rows.sort(key=lambda row: row["usable_index"])
    if require_prefix and [row["usable_index"] for row in rows] != list(
        range(len(rows))
    ):
        raise RuntimeError("staged usable indices are not an ordered prefix from zero")
    return rows


def float64_boundary_record(**tensors: torch.Tensor) -> dict[str, str]:
    """Return runtime dtype evidence and fail if any certificate tensor is not fp64."""
    if not tensors:
        raise ValueError("at least one certificate tensor is required")
    record = {
        name: str(tensor.dtype).removeprefix("torch.")
        for name, tensor in tensors.items()
    }
    wrong = {name: dtype for name, dtype in record.items() if dtype != "float64"}
    if wrong:
        raise RuntimeError(f"certificate boundary is not float64: {wrong}")
    return record


def load_preflight() -> tuple[dict[str, Any], str]:
    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"missing refresh preflight: {PREFLIGHT_PATH}; run "
            "`python scripts/preflight_certificate_refresh.py` first"
        )
    value = load_json_object(PREFLIGHT_PATH)
    if value.get("schema") != PREFLIGHT_SCHEMA:
        raise RuntimeError("preflight schema mismatch")
    if value.get("protocol_commit") != PROTOCOL_COMMIT:
        raise RuntimeError("preflight protocol commit mismatch")
    if value.get("protocol_sha256") != PROTOCOL_SHA256:
        raise RuntimeError("preflight protocol SHA-256 mismatch")
    require_file_hash(
        REPO_ROOT / PROTOCOL_PATH,
        PROTOCOL_SHA256,
        "registered refresh protocol",
    )
    return value, sha256_file(PREFLIGHT_PATH)


def verify_registered_sources(preflight: dict[str, Any]) -> None:
    """Ensure no registered implementation/helper/test changed after preflight."""
    recorded = preflight.get("source_sha256")
    if not isinstance(recorded, dict):
        raise RuntimeError("preflight has no source SHA-256 registry")
    expected_keys = set(REGISTERED_SOURCE_PATHS)
    if set(recorded) != expected_keys:
        missing = sorted(expected_keys - set(recorded))
        extra = sorted(set(recorded) - expected_keys)
        raise RuntimeError(
            f"preflight source registry mismatch: missing={missing}, extra={extra}"
        )
    changed = {}
    for relative in REGISTERED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != recorded[relative]:
            changed[relative] = {
                "actual": actual,
                "preflight": recorded[relative],
            }
    if changed:
        raise RuntimeError(f"registered source changed after preflight: {changed}")


def require_staged_artifact(
    path: Path,
    expected_producer: str,
    expected_sample_contract: dict[str, Any] | None = None,
    expected_upstream_sha256: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    resolved = _require_staging_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"required staged artifact is missing: {resolved}")
    payload = load_json_object(resolved)
    contract = payload.get("certificate_contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"staged artifact has no certificate contract: {resolved}")
    preflight, preflight_sha256 = load_preflight()
    verify_registered_sources(preflight)
    expected = {
        "schema": CONTRACT_SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "eps": CERTIFICATE_EPS,
        "arithmetic_dtype": CERTIFICATE_ARITHMETIC_DTYPE,
        "producer": expected_producer,
        "producer_sha256": preflight["source_sha256"].get(expected_producer),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": "float32",
        "torch_version": preflight.get("environment", {}).get("torch"),
        "python_version": preflight.get("environment", {}).get("python"),
        "seed": 0,
    }
    if expected_sample_contract is not None:
        expected["sample_contract"] = expected_sample_contract
    if expected_upstream_sha256 is not None:
        expected["upstream_sha256"] = dict(
            sorted(expected_upstream_sha256.items())
        )
    if expected_producer == "scripts/run_cert_counterfact_gpt2xl.py":
        require_file_hash(
            COUNTERFACT_PATH,
            EXPECTED_COUNTERFACT_SHA256,
            "CounterFact data",
        )
        expected.update(
            {
                "model_id": GPT2_XL_MODEL_ID,
                "model_revision": GPT2_XL_REVISION,
                "data": repo_relative(COUNTERFACT_PATH),
                "data_sha256": EXPECTED_COUNTERFACT_SHA256,
            }
        )
    else:
        require_file_hash(
            CHECKPOINT_PATH,
            EXPECTED_CHECKPOINT_SHA256,
            "HLM5 checkpoint",
        )
        require_file_hash(
            TOKENIZER_PATH,
            EXPECTED_TOKENIZER_SHA256,
            "HLM5 tokenizer",
        )
        expected.update(
            {
                "checkpoint": repo_relative(CHECKPOINT_PATH),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "tokenizer": repo_relative(TOKENIZER_PATH),
                "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            }
        )
    mismatches = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"staged artifact contract mismatch for {resolved}: {mismatches}"
        )
    return payload, sha256_file(resolved)


def require_preflight(producer: Path | str, lane: str) -> tuple[dict[str, Any], str]:
    if lane not in {"hlm5", "counterfact"}:
        raise ValueError(f"unknown refresh lane: {lane}")
    value, receipt_sha256 = load_preflight()
    verify_registered_sources(value)
    producer_path = repo_relative(producer)
    expected_source = value.get("source_sha256", {}).get(producer_path)
    actual_source = sha256_file(Path(producer))
    if expected_source != actual_source:
        raise RuntimeError(
            f"producer changed after preflight: {producer_path}; "
            f"got {actual_source}, receipt has {expected_source}"
        )
    readiness_key = f"{lane}_ready"
    if value.get(readiness_key) is not True:
        raise RuntimeError(f"preflight does not authorize {lane}: {value.get(readiness_key)}")
    return value, receipt_sha256


def _base_contract(
    producer: Path | str,
    preflight_sha256: str,
    model_forward_dtype: str,
    sample_contract: dict[str, Any],
    upstream: dict[str, str] | None,
) -> dict[str, Any]:
    producer_path = Path(producer)
    return {
        "schema": CONTRACT_SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "eps": CERTIFICATE_EPS,
        "arithmetic_dtype": CERTIFICATE_ARITHMETIC_DTYPE,
        "producer": repo_relative(producer_path),
        "producer_sha256": sha256_file(producer_path),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": model_forward_dtype,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "seed": 0,
        "sample_contract": sample_contract,
        "upstream_sha256": dict(sorted((upstream or {}).items())),
    }


def hlm5_contract(
    producer: Path | str,
    preflight_sha256: str,
    model_forward_dtype: str,
    sample_contract: dict[str, Any],
    upstream: dict[str, str] | None = None,
) -> dict[str, Any]:
    require_file_hash(
        CHECKPOINT_PATH,
        EXPECTED_CHECKPOINT_SHA256,
        "HLM5 checkpoint",
    )
    require_file_hash(TOKENIZER_PATH, EXPECTED_TOKENIZER_SHA256, "HLM5 tokenizer")
    value = _base_contract(
        producer,
        preflight_sha256,
        model_forward_dtype,
        sample_contract,
        upstream,
    )
    value.update(
        {
            "checkpoint": repo_relative(CHECKPOINT_PATH),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "tokenizer": repo_relative(TOKENIZER_PATH),
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        }
    )
    return value


def counterfact_contract(
    producer: Path | str,
    preflight_sha256: str,
    sample_contract: dict[str, Any],
) -> dict[str, Any]:
    require_file_hash(
        COUNTERFACT_PATH,
        EXPECTED_COUNTERFACT_SHA256,
        "CounterFact data",
    )
    value = _base_contract(
        producer,
        preflight_sha256,
        "float32",
        sample_contract,
        upstream=None,
    )
    value.update(
        {
            "model_id": GPT2_XL_MODEL_ID,
            "model_revision": GPT2_XL_REVISION,
            "data": repo_relative(COUNTERFACT_PATH),
            "data_sha256": EXPECTED_COUNTERFACT_SHA256,
        }
    )
    return value
