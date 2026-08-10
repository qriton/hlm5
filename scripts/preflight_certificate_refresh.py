"""Freeze exact inputs and implementation before certificate refresh outcomes."""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.artifact_contract import (
    CHECKPOINT_PATH,
    COUNTERFACT_PATH,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COUNTERFACT_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    GPT2_XL_REVISION,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    REGISTERED_SOURCE_PATHS,
    REPO_ROOT,
    TOKENIZER_PATH,
    atomic_write_json,
    repo_relative,
    sha256_file,
)


COMMANDS = (
    "python scripts/run_1b_certificate.py",
    "python scripts/run_1b_betastar.py",
    "python scripts/run_1b_synth_verify.py",
    "python scripts/run_1b_envelope_multikey.py",
    "python scripts/run_1b_faithful_certdosed.py",
    "python scripts/run_cert_counterfact_gpt2xl.py",
    "python scripts/compare_certificate_refresh.py",
)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_record(path: Path, expected: str) -> dict[str, Any]:
    exists = path.is_file()
    actual = sha256_file(path) if exists else None
    return {
        "path": repo_relative(path),
        "exists": exists,
        "sha256": actual,
        "expected_sha256": expected,
        "matches": actual == expected,
    }


def gpt2_snapshot_record() -> dict[str, Any]:
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    candidates = (
        hf_home / "hub" / "models--gpt2-xl" / "snapshots" / GPT2_XL_REVISION,
        hf_home
        / "hub"
        / "models--openai-community--gpt2-xl"
        / "snapshots"
        / GPT2_XL_REVISION,
    )
    present = next((path for path in candidates if path.is_dir()), None)
    return {
        "revision": GPT2_XL_REVISION,
        "cached": present is not None,
        "cache_path": str(present) if present is not None else None,
    }


def build_report() -> dict[str, Any]:
    protocol_file = REPO_ROOT / PROTOCOL_PATH
    if sha256_file(protocol_file) != PROTOCOL_SHA256:
        raise RuntimeError("registered protocol bytes changed")
    missing_sources = [
        relative
        for relative in REGISTERED_SOURCE_PATHS
        if not (REPO_ROOT / relative).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(f"registered source files missing: {missing_sources}")
    dirty = git_output(
        "status",
        "--porcelain",
        "--",
        *REGISTERED_SOURCE_PATHS,
    )
    if dirty:
        raise RuntimeError(
            "registered implementation must be committed before preflight:\n" + dirty
        )

    checkpoint = file_record(CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256)
    tokenizer = file_record(TOKENIZER_PATH, EXPECTED_TOKENIZER_SHA256)
    counterfact = file_record(COUNTERFACT_PATH, EXPECTED_COUNTERFACT_SHA256)
    gpt2_snapshot = gpt2_snapshot_record()
    transformers_version = package_version("transformers")
    tokenizers_version = package_version("tokenizers")
    hlm5_ready = (
        checkpoint["matches"]
        and tokenizer["matches"]
        and tokenizers_version is not None
    )
    counterfact_ready = (
        counterfact["matches"]
        and gpt2_snapshot["cached"]
        and transformers_version is not None
    )
    source_sha256 = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in REGISTERED_SOURCE_PATHS
    }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_path": PROTOCOL_PATH,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": git_output("rev-parse", "HEAD"),
        "source_sha256": source_sha256,
        "inputs": {
            "hlm5_checkpoint": checkpoint,
            "hlm5_tokenizer": tokenizer,
            "counterfact": counterfact,
            "gpt2_xl_snapshot": gpt2_snapshot,
        },
        "commands": list(COMMANDS),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "tokenizers": tokenizers_version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        },
        "registered_dtypes": {
            "hlm5_model_forward": "float32",
            "gpt2_xl_model_forward": "float32",
            "certificate_arithmetic": "float64",
        },
        "hlm5_ready": hlm5_ready,
        "counterfact_ready": counterfact_ready,
        "status": (
            "READY"
            if hlm5_ready and counterfact_ready
            else "HLM5_READY_COUNTERFACT_INCOMPLETE"
            if hlm5_ready
            else "INCOMPLETE"
        ),
        "claim_scope": (
            "Input/source freeze only; no model forward or refreshed certificate "
            "outcome was computed."
        ),
    }


def main() -> None:
    report = build_report()
    atomic_write_json(PREFLIGHT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote={repo_relative(PREFLIGHT_PATH)}")
    if not report["hlm5_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
