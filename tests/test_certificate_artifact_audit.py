"""Tests for the deterministic certificate-artifact provenance audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_certificate_artifacts.py"
SPEC = importlib.util.spec_from_file_location("certificate_artifact_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_current_release_is_correctly_blocked_from_refresh_claim() -> None:
    report = audit.build_report()
    assert report["status"] == "REFRESH_REQUIRED"
    assert report["audit_implementation"]["sha256"] == audit.sha256_file(
        MODULE_PATH
    )
    assert len(report["artifacts"]) == 6
    assert all(
        item["status"] == "REFRESH_REQUIRED" for item in report["artifacts"]
    )
    assert not report["verify_artifacts_certificate_payload_coverage"]["complete"]


def test_missing_verifier_coverage_cannot_produce_ready_status() -> None:
    records = [{"status": "REFRESHED_AND_BOUND"} for _ in range(6)]
    assert audit.overall_status(records, {"complete": False}) == (
        "REFRESH_REQUIRED"
    )
    assert audit.overall_status(records, {"complete": True}) == (
        "READY_TO_PUBLISH"
    )


def test_all_producers_have_exact_signs_but_no_float64_contract() -> None:
    report = audit.build_report()
    for item in report["artifacts"]:
        producer = item["producer"]
        assert producer["exact_sign_source"]
        assert producer["certificate_arithmetic_dtype_marker"] is None
        assert producer["float32_certificate_evidence"]


def test_counterfact_resume_is_not_contract_bound() -> None:
    report = audit.build_report()
    counterfact = next(
        item
        for item in report["artifacts"]
        if item["name"] == "counterfact_certificate_features"
    )
    assert counterfact["producer"]["resumable_output"]
    assert not counterfact["producer"]["resume_binds_certificate_contract"]
    assert not counterfact["resume_fail_closed"]
    assert not counterfact["contract_matches"]["jsonl_sha256"]


def test_historical_lineage_distinguishes_untouched_and_partial_refreshes() -> None:
    report = audit.build_report()
    by_name = {item["name"]: item for item in report["artifacts"]}
    assert by_name["one_key_envelope_and_synthesis"]["lineage"] == (
        "pre_hardening_bytes"
    )
    assert by_name["synthesis_flip"]["lineage"] == "pre_hardening_bytes"
    assert by_name["multi_key_envelope"]["lineage"] == (
        "touched_in_hardening_commit_but_unbound"
    )
    assert by_name["beta_star"]["lineage"] == (
        "touched_in_hardening_commit_but_unbound"
    )
    assert all(
        payload["bytes_match_registered_lineage"]
        for item in report["artifacts"]
        for payload in item["payloads"]
    )


def test_lineage_is_hash_based_and_fails_closed_for_changed_bytes() -> None:
    assert audit.lineage_label(
        [{"bytes_match_registered_lineage": False, "registered_lineage": None}]
    ) == "unregistered_or_refreshed_bytes"
