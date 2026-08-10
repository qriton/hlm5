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


def test_current_release_is_ready_to_publish() -> None:
    report = audit.build_report()
    assert report["status"] == "READY_TO_PUBLISH"
    assert report["audit_implementation"]["sha256"] == audit.sha256_file(
        MODULE_PATH
    )
    assert len(report["artifacts"]) == 6
    assert all(item["status"] == "REFRESHED_AND_BOUND" for item in report["artifacts"])
    assert report["verify_artifacts_certificate_payload_coverage"]["complete"]
    assert report["promotion_manifest"]["status"] == "BOUND"


def test_missing_verifier_coverage_cannot_produce_ready_status() -> None:
    records = [{"status": "REFRESHED_AND_BOUND"} for _ in range(6)]
    manifest = {"status": "BOUND"}
    assert audit.overall_status(records, {"complete": False}, manifest) == (
        "REFRESH_REQUIRED"
    )
    assert audit.overall_status(records, {"complete": True}, manifest) == (
        "READY_TO_PUBLISH"
    )
    assert audit.overall_status(
        records, {"complete": True}, {"status": "INVALID"}
    ) == "REFRESH_REQUIRED"


def test_producers_and_release_payloads_are_bound() -> None:
    report = audit.build_report()
    for item in report["artifacts"]:
        producer = item["producer"]
        assert producer["exact_sign_source"]
        assert producer["certificate_arithmetic_dtype_marker"] == "float64"
        assert not producer["float32_certificate_evidence"]
        assert item["status"] == "REFRESHED_AND_BOUND"
        assert item["certificate_contract"]["present"]
        assert all(item["contract_matches"].values())


def test_counterfact_resume_and_release_rows_are_bound() -> None:
    report = audit.build_report()
    counterfact = next(
        item
        for item in report["artifacts"]
        if item["name"] == "counterfact_certificate_features"
    )
    assert counterfact["producer"]["resumable_output"]
    assert counterfact["producer"]["resume_binds_certificate_contract"]
    assert counterfact["resume_fail_closed"]
    assert counterfact["contract_matches"]["jsonl_sha256"]


def test_promoted_bytes_no_longer_match_historical_unbound_lineage() -> None:
    report = audit.build_report()
    by_name = {item["name"]: item for item in report["artifacts"]}
    assert all(
        item["lineage"] == "unregistered_or_refreshed_bytes"
        for item in by_name.values()
    )
    assert all(
        not payload["bytes_match_registered_lineage"]
        for item in report["artifacts"]
        for payload in item["payloads"]
    )


def test_promotion_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    value = audit.load_json(audit.PROMOTION_MANIFEST)
    value["artifact_sha256"]["results/betastar.json"] = "0" * 64
    tampered = tmp_path / "manifest.json"
    tampered.write_text(audit.stable_json(value), encoding="utf-8")
    record = audit.promotion_manifest_record(tampered)
    assert record["status"] == "INVALID"
    assert "artifact_sha256" in record["mismatches"]


def test_lineage_is_hash_based_and_fails_closed_for_changed_bytes() -> None:
    assert audit.lineage_label(
        [{"bytes_match_registered_lineage": False, "registered_lineage": None}]
    ) == "unregistered_or_refreshed_bytes"
