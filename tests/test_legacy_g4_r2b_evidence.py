"""Integrity checks for the completed legacy G4 R2b bridge evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "legacy_g4_hybrid_520k_r2b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_verdict_hash_chain_matches_archived_bytes() -> None:
    verdict = load("verdict.json")
    expected = {
        "attempt_sha256": "attempt.json",
        "result_sha256": "result.json",
        "train_log_sha256": "train.log",
        "target_marker_sha256": "ckpt_step522000.sharded",
        "stdout_sha256": "leonardo-51813699.out",
        "stderr_sha256": "leonardo-51813699.err",
    }
    for field, filename in expected.items():
        assert verdict["evidence"][field] == sha256(EVIDENCE / filename)


def test_result_and_marker_prove_state_complete_bridge_pass() -> None:
    attempt = load("attempt.json")
    result = load("result.json")
    marker = load("ckpt_step522000.sharded")
    verdict = load("verdict.json")

    assert result["attempt_sha256"] == sha256(EVIDENCE / "attempt.json")
    assert attempt["launcher_sha256"] == (
        "d9f0d10c983fab5146933ecc201c07062245c2a71fff33bdffba9b4813292f9a")
    assert result["status"] == verdict["formal_verdict"] == "PASS_BRIDGE"
    assert result["printed_ppl"] == verdict["printed_ppl"] == 15.58
    assert result["printed_ppl"] <= result["printed_ppl_ceiling"] == 15.77
    assert result["target_shards"] == verdict["target_shards"] == 96
    assert result["target_bytes"] == verdict["target_bytes"] == 43294668998
    assert result["target_aggregate_sha256"] == verdict["target_aggregate_sha256"]
    assert result["stateful_reload_verified"] is True
    assert marker["step"] == 521999
    assert marker["world_size"] == 96
    assert marker["checkpoint_schema"] == "hlm5-g4-state-complete-v1"
    assert marker["optimizer_state_saved"] is True
    assert marker["runtime_state_saved"] is True
    assert marker["val_ppl"] == verdict["precise_marker_ppl"]
