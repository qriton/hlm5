"""Integrity and scope checks for the completed R2c invalid result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "legacy_g4_hybrid_522k_r2c"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_verdict_hash_chain_matches_archived_bytes() -> None:
    verdict = load("verdict.json")
    expected = {
        "attempt_sha256": "attempt.json",
        "result_sha256": "result.json",
        "environment_sha256": "environment.json",
        "continuous_log_sha256": "continuous.log",
        "split_pre_restart_log_sha256": "split-pre-restart.log",
        "midpoint_report_sha256": "midpoint-comparison/aggregate.json",
        "stdout_sha256": "leonardo-51821287.out",
        "stderr_sha256": "leonardo-51821287.err",
        "evidence_archive_sha256": "hlm5-g4-hybrid-r2c-51821287-evidence.tar.gz",
    }
    for field, filename in expected.items():
        assert verdict["evidence"][field] == sha256(EVIDENCE / filename)


def test_invalid_result_stops_before_restart_and_covers_all_ranks() -> None:
    verdict = load("verdict.json")
    result = load("result.json")
    aggregate = load("midpoint-comparison/aggregate.json")

    assert verdict["formal_verdict"] == result["status"] == "INVALID_REPRODUCIBILITY"
    assert verdict["restart_segment_executed"] is False
    assert result["midpoint_exact"] is False
    assert result["evidence"]["final_report_sha256"] is None
    assert result["evidence"]["split_post_restart_log_sha256"] is None
    assert aggregate["status"] == "FAIL_MISMATCH"
    assert aggregate["rank_report_count"] == 96
    assert aggregate["mismatching_ranks"] == list(range(96))
    for rank in range(96):
        report = load(f"midpoint-comparison/rank{rank:03d}.json")
        assert report["exact"] is False
        assert report["expected_step"] == 522499


def test_ppl_difference_is_small_but_not_relabelled_as_exact() -> None:
    verdict = load("verdict.json")
    assert verdict["continuous_midpoint_ppl"] == 15.520899957093135
    assert verdict["split_midpoint_ppl"] == 15.514838289543917
    assert abs(verdict["split_minus_continuous_ppl"]) < 0.01
    assert verdict["formal_verdict"] != "PASS_CONTINUITY"
    assert "no completion ladder" in verdict["next_gate"]
