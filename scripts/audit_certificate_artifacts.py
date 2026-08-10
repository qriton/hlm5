"""Audit exact-sign certificate artifact provenance without rerunning models.

This command is deliberately read-only unless ``--write`` is supplied.  A
``REFRESH_REQUIRED`` status means the audit succeeded and found that published
quantitative artifacts are not yet bound to the current certificate contract.
It is not a model-performance failure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "certificate_artifact_refresh_audit.json"
HARDENING_COMMIT = "a485c31fe4fd079363701c0ea21fe8878f01e878"
INITIAL_RELEASE_COMMIT = "e2451b2213bb00e59903b4e4e89a8e6c59bd8584"
EXPECTED_CHECKPOINT_SHA256 = (
    "3e1c94d28125c2f86f3eeca030db3610f2fa679512c29c6b6608b24fc363e0f1"
)
EXPECTED_TOKENIZER_SHA256 = (
    "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588"
)
REGISTERED_ARTIFACT_LINEAGE = {
    "results/cert_envelope_synth.json": (
        "6944e29505f49fc4fd991705f9334c4fb35f2e3e4edd75185dd167fd66f6cd57",
        INITIAL_RELEASE_COMMIT,
        "pre_hardening_bytes",
    ),
    "results/cert_envelope_multikey.json": (
        "b0a31b3a0fede31a197b23d9441e48df5e0fc29ec2fdba3369fbdb6d356a8f69",
        HARDENING_COMMIT,
        "touched_in_hardening_commit_but_unbound",
    ),
    "results/betastar.json": (
        "a68d98b1687c99380a0c07a53907e9a462b497e2db70048f223b98654dbefb08",
        HARDENING_COMMIT,
        "touched_in_hardening_commit_but_unbound",
    ),
    "results/synth_verify.json": (
        "e5c3c9b4f54d857f06c5f6ae8b2cc64038357d4baee7cdc32e4646e63a849454",
        INITIAL_RELEASE_COMMIT,
        "pre_hardening_bytes",
    ),
    "results/hlm5_1b_faithful_certdosed.json": (
        "9d04423442a7564346e706b5ae44bb5a56e4cb524615cb21dab6aeecf414caa8",
        HARDENING_COMMIT,
        "touched_in_hardening_commit_but_unbound",
    ),
    "results/cert_counterfact_gpt2xl.jsonl": (
        "05acfdb47ffedbca1986f2def735b5d10d0fa95bbd3c487feb73929b3625e4bd",
        INITIAL_RELEASE_COMMIT,
        "pre_hardening_bytes",
    ),
    "results/cert_counterfact_gpt2xl_summary.json": (
        "009007821ead41dbfc2b066f73607408c000247cfa67263ba4bd7eda82a9a22e",
        INITIAL_RELEASE_COMMIT,
        "pre_hardening_bytes",
    ),
}
EPS_PATTERN = re.compile(r"^\s*EPS\s*=\s*([0-9.eE+-]+)", re.MULTILINE)
REQUIRED_CONTRACT_FIELDS = (
    "eps",
    "arithmetic_dtype",
    "producer",
    "producer_sha256",
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    producer: str
    payloads: tuple[str, ...]
    contract_artifact: str
    requirement: str
    resumable_jsonl: bool = False


SPECS = (
    ArtifactSpec(
        "one_key_envelope_and_synthesis",
        "scripts/run_1b_certificate.py",
        ("results/cert_envelope_synth.json",),
        "results/cert_envelope_synth.json",
        "frozen_hlm5_1b",
    ),
    ArtifactSpec(
        "multi_key_envelope",
        "scripts/run_1b_envelope_multikey.py",
        ("results/cert_envelope_multikey.json",),
        "results/cert_envelope_multikey.json",
        "frozen_hlm5_1b",
    ),
    ArtifactSpec(
        "beta_star",
        "scripts/run_1b_betastar.py",
        ("results/betastar.json",),
        "results/betastar.json",
        "frozen_hlm5_1b",
    ),
    ArtifactSpec(
        "synthesis_flip",
        "scripts/run_1b_synth_verify.py",
        ("results/synth_verify.json",),
        "results/synth_verify.json",
        "frozen_hlm5_1b",
    ),
    ArtifactSpec(
        "faithful_certificate_dosing",
        "scripts/run_1b_faithful_certdosed.py",
        ("results/hlm5_1b_faithful_certdosed.json",),
        "results/hlm5_1b_faithful_certdosed.json",
        "frozen_hlm5_1b",
    ),
    ArtifactSpec(
        "counterfact_certificate_features",
        "scripts/run_cert_counterfact_gpt2xl.py",
        (
            "results/cert_counterfact_gpt2xl.jsonl",
            "results/cert_counterfact_gpt2xl_summary.json",
        ),
        "results/cert_counterfact_gpt2xl_summary.json",
        "gpt2_xl_and_counterfact",
        resumable_jsonl=True,
    ),
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def source_contract(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    eps_values = [float(value) for value in EPS_PATTERN.findall(source)]
    dtype_marker = re.search(
        r"^\s*CERTIFICATE_ARITHMETIC_DTYPE\s*=\s*[\"']([^\"']+)[\"']",
        source,
        re.MULTILINE,
    )
    float32_lines = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(source.splitlines(), start=1)
        if ".float()" in line and ("W =" in line or "h =" in line)
    ]
    resumable = "if OUT.exists()" in source or "load_contract_jsonl(" in source
    resume_binds_contract = resumable and (
        "certificate_contract" in source or "load_contract_jsonl(" in source
    )
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "eps_assignments": eps_values,
        "exact_sign_source": bool(eps_values)
        and all(value == 0.0 for value in eps_values),
        "certificate_arithmetic_dtype_marker": (
            dtype_marker.group(1) if dtype_marker else None
        ),
        "float32_certificate_evidence": float32_lines,
        "resumable_output": resumable,
        "resume_binds_certificate_contract": resume_binds_contract,
    }


def artifact_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    digest = sha256_file(path) if path.is_file() else None
    registered = REGISTERED_ARTIFACT_LINEAGE.get(relative)
    matches_registered = registered is not None and digest == registered[0]
    return {
        "path": relative,
        "exists": path.is_file(),
        "sha256": digest,
        "registered_sha256": registered[0] if registered else None,
        "registered_commit": registered[1] if registered else None,
        "registered_lineage": registered[2] if registered else None,
        "bytes_match_registered_lineage": matches_registered,
    }


def contract_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "missing_fields": list(REQUIRED_CONTRACT_FIELDS)}
    value = load_json(path).get("certificate_contract")
    if not isinstance(value, dict):
        return {"present": False, "missing_fields": list(REQUIRED_CONTRACT_FIELDS)}
    missing = [field for field in REQUIRED_CONTRACT_FIELDS if field not in value]
    return {
        "present": True,
        "value": value,
        "missing_fields": missing,
    }


def lineage_label(payloads: list[dict[str, Any]]) -> str:
    if not payloads or not all(
        item["bytes_match_registered_lineage"] for item in payloads
    ):
        return "unregistered_or_refreshed_bytes"
    labels = {item["registered_lineage"] for item in payloads}
    return labels.pop() if len(labels) == 1 else "mixed_registered_lineage"


def evaluate_spec(spec: ArtifactSpec) -> dict[str, Any]:
    producer_path = ROOT / spec.producer
    source = source_contract(producer_path)
    payloads = [artifact_record(relative) for relative in spec.payloads]
    contract = contract_record(ROOT / spec.contract_artifact)
    value = contract.get("value", {})
    contract_matches = {
        "eps_zero": value.get("eps") == 0.0,
        "arithmetic_float64": value.get("arithmetic_dtype") == "float64",
        "producer_path": value.get("producer") == spec.producer,
        "producer_sha256": value.get("producer_sha256") == source["sha256"],
    }
    if spec.requirement == "frozen_hlm5_1b":
        contract_matches.update(
            {
                "checkpoint_sha256": value.get("checkpoint_sha256")
                == EXPECTED_CHECKPOINT_SHA256,
                "tokenizer_sha256": value.get("tokenizer_sha256")
                == EXPECTED_TOKENIZER_SHA256,
            }
        )
    else:
        row_payload = ROOT / spec.payloads[0]
        contract_matches.update(
            {
                "model_revision_present": bool(value.get("model_revision")),
                "data_sha256_present": bool(value.get("data_sha256")),
                "jsonl_sha256": row_payload.is_file()
                and value.get("jsonl_sha256") == sha256_file(row_payload),
            }
        )
    source_ready = (
        source["exact_sign_source"]
        and source["certificate_arithmetic_dtype_marker"] == "float64"
        and not source["float32_certificate_evidence"]
    )
    resume_ready = not spec.resumable_jsonl or source[
        "resume_binds_certificate_contract"
    ]
    ready = (
        source_ready
        and resume_ready
        and all(item["exists"] for item in payloads)
        and contract["present"]
        and not contract["missing_fields"]
        and all(contract_matches.values())
    )
    return {
        "name": spec.name,
        "requirement": spec.requirement,
        "producer": source,
        "payloads": payloads,
        "lineage": lineage_label(payloads),
        "certificate_contract": contract,
        "contract_matches": contract_matches,
        "resume_fail_closed": resume_ready,
        "status": "REFRESHED_AND_BOUND" if ready else "REFRESH_REQUIRED",
    }


def verifier_payload_coverage(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    call_string_arguments = {
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for value in (*node.args, *(keyword.value for keyword in node.keywords))
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    required = sorted(relative for spec in SPECS for relative in spec.payloads)
    covered = [
        relative
        for relative in required
        if Path(relative).name in call_string_arguments
    ]
    return {
        "covered": covered,
        "required": required,
        "complete": covered == required,
    }


def overall_status(
    records: list[dict[str, Any]], verifier_coverage: dict[str, Any]
) -> str:
    artifacts_bound = all(
        item["status"] == "REFRESHED_AND_BOUND" for item in records
    )
    return (
        "READY_TO_PUBLISH"
        if artifacts_bound and verifier_coverage["complete"]
        else "REFRESH_REQUIRED"
    )


def build_report() -> dict[str, Any]:
    audit_path = Path(__file__).resolve()
    verifier_path = ROOT / "scripts" / "verify_artifacts.py"
    records = [evaluate_spec(spec) for spec in SPECS]
    verifier_coverage = verifier_payload_coverage(verifier_path)
    downstream = [
        artifact_record("results/cert_vs_editors_analysis.json"),
        artifact_record("scripts/make_figures.py"),
        artifact_record("paper/hlm5-paper.tex"),
    ]
    return {
        "schema": "hlm5-certificate-artifact-refresh-audit-v1",
        "as_of": "2026-08-10",
        "status": overall_status(records, verifier_coverage),
        "audit_implementation": {
            "path": audit_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(audit_path),
        },
        "existing_verifier": {
            "path": verifier_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(verifier_path),
        },
        "hardening_commit": HARDENING_COMMIT,
        "initial_release_commit": INITIAL_RELEASE_COMMIT,
        "required_contract": {
            "eps": 0.0,
            "arithmetic_dtype": "float64",
            "producer_path_and_sha256": True,
            "frozen_hlm5_1b_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "frozen_hlm5_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "counterfact_model_data_and_jsonl_hashes": True,
            "verifier_covers_all_primary_payloads": True,
        },
        "artifacts": records,
        "verify_artifacts_certificate_payload_coverage": verifier_coverage,
        "downstream_refresh_after_producers": downstream,
        "claim_scope": (
            "Static provenance/completeness audit only; no model forward pass, "
            "metric refresh, performance claim, training, or external compute."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the deterministic report to {OUTPUT.relative_to(ROOT)}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    if args.write:
        OUTPUT.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(stable_json(report))
    print(f"status={report['status']}")
    if args.write:
        print(f"wrote={OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
