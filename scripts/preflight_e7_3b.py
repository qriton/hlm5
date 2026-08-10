"""Fail-closed baseline-only preflight for the frozen-3B HLM5 E7 study."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch
import transformers

from hlm5.e7_contract import (
    EXECUTION_RECEIPT_PATH,
    EXECUTION_SCHEMA,
    HEAD_IDENTITY_MAX_ABS,
    MODEL_FILE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    SEED,
    SOURCE_INPUT_SHA256,
    TARGET_POOL_COUNT,
    TARGET_POOL_SHA256,
    atomic_write_json,
    git_output,
    registered_samples,
    registered_source_sha256,
    sha256_file,
    snapshot_path,
    stable_json_sha256,
    target_pool,
    tensor_sha256,
    verify_protocol,
    verify_snapshot,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    encode_batch,
    load_pinned_model,
    load_pinned_tokenizer,
    native_head_logits,
)


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e7_3b.py", "-q"]


def environment_record(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


@torch.inference_mode()
def baseline_head_identity(
    model: object,
    tokenizer: object,
    prompt: str,
    device: torch.device,
) -> dict[str, object]:
    encoded = encode_batch(tokenizer, [prompt], device)
    outputs = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    native_logits = outputs.logits[0, -1]
    hidden = outputs.hidden_states[-1][0, -1]
    independent_logits = native_head_logits(model, hidden)
    difference = (native_logits.float() - independent_logits.float()).abs()
    max_abs = float(difference.max())
    native_argmax = int(native_logits.argmax())
    independent_argmax = int(independent_logits.argmax())
    if native_argmax != independent_argmax or max_abs > HEAD_IDENTITY_MAX_ABS:
        raise RuntimeError(
            "baseline head identity failed: "
            f"argmax {native_argmax} != {independent_argmax} or "
            f"max_abs {max_abs} > {HEAD_IDENTITY_MAX_ABS}"
        )
    return {
        "prompt_sha256": stable_json_sha256(prompt),
        "native_argmax": native_argmax,
        "independent_argmax": independent_argmax,
        "argmax_equal": True,
        "max_abs_logit_difference": max_abs,
        "maximum_allowed": HEAD_IDENTITY_MAX_ABS,
        "native_logits_sha256": tensor_sha256(native_logits),
        "independent_logits_sha256": tensor_sha256(independent_logits),
        "hidden_sha256": tensor_sha256(hidden),
        "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
        "logits_dtype": str(native_logits.dtype).removeprefix("torch."),
    }


def main() -> None:
    torch.manual_seed(SEED)
    verify_protocol()
    source_sha256 = registered_source_sha256(require_clean=True)
    implementation_commit = git_output("rev-parse", "HEAD")
    tracked = set(git_output("ls-files", *source_sha256).splitlines())
    if tracked != set(source_sha256):
        raise RuntimeError("every registered E7 source must be tracked by git")

    snapshot = snapshot_path()
    model_files = verify_snapshot(snapshot)
    samples = registered_samples()

    if not torch.cuda.is_available():
        raise RuntimeError("E7 preflight requires one CUDA device")
    device = torch.device("cuda", 0)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device is visible")
    tokenizer = load_pinned_tokenizer(snapshot)
    pool = target_pool(tokenizer)

    tests = subprocess.run(
        TEST_COMMAND,
        cwd=REPO_ROOT_BOOTSTRAP,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(
            "registered E7 CPU tests failed:\n" + tests.stdout + tests.stderr
        )

    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    anchor_prompt = samples["FACTS"][0][0]
    head_identity = baseline_head_identity(model, tokenizer, anchor_prompt, device)
    environment = environment_record(device)

    preflight = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "branch": git_output("branch", "--show-current"),
        "source_sha256": source_sha256,
        "source_input_sha256": dict(SOURCE_INPUT_SHA256),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "model_files": model_files,
        "snapshot_selector": (
            "HLM5_E7_SNAPSHOT" if os.environ.get("HLM5_E7_SNAPSHOT") else "HF_HOME"
        ),
        "target_pool_count": len(pool),
        "target_pool_sha256": stable_json_sha256(pool),
        "registered_target_pool_count": TARGET_POOL_COUNT,
        "registered_target_pool_sha256": TARGET_POOL_SHA256,
        "sample_sha256": stable_json_sha256(samples),
        "model_invariants": invariants,
        "baseline_only_head_identity": head_identity,
        "environment": environment,
        "candidate_outputs_computed": False,
        "commands": list(REGISTERED_COMMANDS),
    }
    atomic_write_json(PREFLIGHT_PATH, preflight)
    preflight_sha256 = sha256_file(PREFLIGHT_PATH)

    execution = {
        "schema": EXECUTION_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "preflight_sha256": preflight_sha256,
        "source_sha256": source_sha256,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "sample_sha256": preflight["sample_sha256"],
        "target_pool_sha256": TARGET_POOL_SHA256,
        "environment": environment,
        "registered_commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e7_3b.py -q",
        "tests_returncode": tests.returncode,
        "tests_stdout": tests.stdout.strip(),
        "tests_stderr": tests.stderr.strip(),
        "candidate_outputs_computed_before_receipt": False,
    }
    atomic_write_json(EXECUTION_RECEIPT_PATH, execution)
    print(f"READY preflight: {PREFLIGHT_PATH}")
    print(f"READY execution receipt: {EXECUTION_RECEIPT_PATH}")


if __name__ == "__main__":
    main()
