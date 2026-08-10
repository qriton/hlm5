"""Outcome-blind preflight for the registered E8 full-path gate."""

# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.e7_contract import (
    HEAD_IDENTITY_MAX_ABS,
    MODEL_FILE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    registered_samples,
    snapshot_path,
    stable_json_sha256,
    tensor_sha256,
    verify_snapshot,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    encode_batch,
    load_pinned_model,
    load_pinned_tokenizer,
    native_head_logits,
)
from hlm5.e7b_runtime import configure_native_runtime, environment_record
from hlm5.e7c_runtime import raw_and_zca_directions
from hlm5.e8_contract import (
    CASE_COUNT,
    CASE_IDS,
    CASE_SHA256,
    CASE_START,
    DEPENDENCY_SHA256,
    EXPECTED_BASIS,
    EXPECTED_ENVIRONMENT,
    NAMED_OUTPUTS,
    NEUTRAL_SHA256,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROMPT_COUNT,
    PROMPT_SHA256,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    SEED,
    WHITEN_SHA256,
    assert_outputs_absent,
    atomic_create_json,
    git_output,
    registered_source_sha256,
    selected_population,
    verify_bound_inputs,
    verify_dependencies,
    verify_protocol,
)


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e8_3b.py", "-q"]


@torch.inference_mode()
def baseline_head_identity(
    model: object, tokenizer: object, prompt: str, device: torch.device
) -> dict[str, object]:
    encoded = encode_batch(tokenizer, [prompt], device)
    with torch.autocast(device_type="cuda", enabled=False):
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    native_logits = outputs.logits[0, -1].detach().contiguous()
    hidden = outputs.hidden_states[-1][0, -1].detach().contiguous()
    with torch.autocast(device_type="cuda", enabled=False):
        independent_logits = native_head_logits(model, hidden)
    difference = (native_logits.float() - independent_logits.float()).abs()
    max_abs = float(difference.max())
    native_argmax = int(native_logits.argmax())
    independent_argmax = int(independent_logits.argmax())
    if native_argmax != independent_argmax or max_abs > HEAD_IDENTITY_MAX_ABS:
        raise RuntimeError("E8 baseline native-head identity failed")
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
    assert_outputs_absent(NAMED_OUTPUTS)
    torch.manual_seed(SEED)
    native_runtime = configure_native_runtime()
    verify_protocol()
    verify_dependencies()
    bound_inputs = verify_bound_inputs()
    source_sha256 = registered_source_sha256(require_clean=True)
    implementation_commit = git_output("rev-parse", "HEAD")
    tracked = set(git_output("ls-files", *source_sha256).splitlines())
    if tracked != set(source_sha256):
        raise RuntimeError("every registered E8 source must be tracked by git")

    snapshot = snapshot_path()
    model_files = verify_snapshot(snapshot)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E8 preflight requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)

    tests = subprocess.run(
        TEST_COMMAND,
        cwd=REPO_ROOT_BOOTSTRAP,
        capture_output=True,
        text=True,
        check=False,
    )
    if tests.returncode != 0:
        raise RuntimeError(
            "registered E8 tests failed before preflight:\n"
            + tests.stdout
            + tests.stderr
        )

    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    head_native = model.get_output_embeddings().weight.detach()
    target_ids = torch.tensor(
        [case["target_id"] for case in population["cases"]],
        device=device,
        dtype=torch.long,
    )
    raw, candidate, basis = raw_and_zca_directions(head_native, target_ids)
    for name, value in EXPECTED_BASIS.items():
        if basis.get(name) != value:
            raise RuntimeError(f"E8 registered basis mismatch: {name}")
    if torch.equal(raw, candidate):
        raise RuntimeError("E8 raw and ZCA directions collapsed to one arm")
    anchor = registered_samples()["FACTS"][0][0]
    head_identity = baseline_head_identity(model, tokenizer, anchor, device)
    environment = environment_record(device)
    if environment != EXPECTED_ENVIRONMENT:
        raise RuntimeError(f"registered E8 environment mismatch: {environment}")

    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "branch": git_output("branch", "--show-current"),
        "source_sha256": source_sha256,
        "dependency_sha256": dict(DEPENDENCY_SHA256),
        "bound_inputs": bound_inputs,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "model_files": model_files,
        "snapshot_selector": (
            "HLM5_E7_SNAPSHOT" if os.environ.get("HLM5_E7_SNAPSHOT") else "HF_HOME"
        ),
        "case_start": CASE_START,
        "case_count": CASE_COUNT,
        "case_ids": CASE_IDS,
        "case_sha256": CASE_SHA256,
        "prompt_count": PROMPT_COUNT,
        "prompt_sha256": PROMPT_SHA256,
        "neutral_sha256": NEUTRAL_SHA256,
        "whitening_sha256": WHITEN_SHA256,
        "basis_and_directions": basis,
        "raw_candidate_equal": False,
        "model_invariants": invariants,
        "baseline_only_head_identity": head_identity,
        "environment": environment,
        "native_runtime": native_runtime,
        "preflight_test_command": "python -m pytest tests/test_e8_3b.py -q",
        "preflight_tests_returncode": tests.returncode,
        "preflight_tests_stdout": tests.stdout.strip(),
        "preflight_tests_stderr": tests.stderr.strip(),
        "fresh_prompt_outcomes_computed": False,
        "commands": list(REGISTERED_COMMANDS),
    }
    atomic_create_json(PREFLIGHT_PATH, payload)
    print(f"READY preflight: {PREFLIGHT_PATH}")


if __name__ == "__main__":
    main()
