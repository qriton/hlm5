"""Outcome-blind preflight for HLM5 E11 cross-node portability."""

# ruff: noqa: E402
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hlm5.e7_contract import (
    MODEL_FILE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    snapshot_path,
    verify_snapshot,
)
from hlm5.e7_runtime import (
    assert_model_invariants,
    load_pinned_model,
    load_pinned_tokenizer,
)
from hlm5.e7b_runtime import configure_native_runtime
from hlm5.e11_contract import (
    DEPENDENCY_SHA256,
    E10_EVIDENCE_COMMIT,
    NAMED_OUTPUTS,
    PREFLIGHT_PATH,
    PREFLIGHT_SCHEMA,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REGISTERED_COMMANDS,
    assert_outputs_absent,
    atomic_create_json,
    environment_failures,
    frozen_e10,
    git_output,
    registered_source_sha256,
    selected_population,
    verify_dependencies,
    verify_protocol,
)
from hlm5.e11_runtime import e11_environment_record, load_bundle


TEST_COMMAND = [sys.executable, "-m", "pytest", "tests/test_e11_3b.py", "-q"]


def main() -> None:
    assert_outputs_absent(NAMED_OUTPUTS)
    torch.manual_seed(0)
    native_runtime = configure_native_runtime()
    verify_protocol()
    verify_dependencies()
    evidence = frozen_e10()
    source_sha256 = registered_source_sha256(require_clean=True)
    implementation_commit = git_output("rev-parse", "HEAD")
    if set(git_output("ls-files", *source_sha256).splitlines()) != set(source_sha256):
        raise RuntimeError("every registered E11 source must be tracked")
    tests = subprocess.run(TEST_COMMAND, cwd=ROOT, capture_output=True, text=True)
    if tests.returncode:
        raise RuntimeError(
            "registered E11 tests failed:\n" + tests.stdout + tests.stderr
        )
    snapshot = snapshot_path()
    model_files = verify_snapshot(snapshot)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("registered E11 preflight requires exactly one GPU")
    device = torch.device("cuda", 0)
    environment = e11_environment_record(device)
    failures = environment_failures(environment)
    if failures:
        raise RuntimeError(
            "registered E11 environment mismatch: " + "; ".join(failures)
        )
    tokenizer = load_pinned_tokenizer(snapshot)
    population = selected_population(tokenizer)
    model = load_pinned_model(snapshot, device)
    invariants = assert_model_invariants(model)
    loaded = load_bundle(
        ROOT / "results/e10_3b_evidence/adapter_1024.pt", evidence["bundle"]
    )
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_commit": implementation_commit,
        "branch": git_output("branch", "--show-current"),
        "source_sha256": source_sha256,
        "dependency_sha256": dict(DEPENDENCY_SHA256),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": dict(MODEL_FILE_SHA256),
        "model_files": model_files,
        "model_invariants": invariants,
        "environment": environment,
        "native_runtime": native_runtime,
        "e10_evidence_commit": E10_EVIDENCE_COMMIT,
        "e10_raw_sha256": evidence["raw_sha256"],
        "e10_bundle_manifest": evidence["bundle"],
        "bundle_members": sorted(loaded),
        "candidate_count": len(population["cases"]),
        "locality_count": len(population["query_prompt_order"]),
        "commands": list(REGISTERED_COMMANDS),
        "test_command": "python -m pytest tests/test_e11_3b.py -q",
        "tests_returncode": tests.returncode,
        "tests_stdout": tests.stdout.strip(),
        "tests_stderr": tests.stderr.strip(),
        "e11_outcomes_computed": False,
    }
    atomic_create_json(PREFLIGHT_PATH, payload)
    print(f"READY preflight: {PREFLIGHT_PATH}")


if __name__ == "__main__":
    main()
