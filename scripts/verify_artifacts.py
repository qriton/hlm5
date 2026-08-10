"""Self-containment verification for hlm5-release/results/.

Loads each externally-sourced JSON artifact copied into results/ and asserts
the load-bearing numbers the paper cites against them. Prints one PASS/FAIL
line per artifact and exits nonzero if any artifact fails.

Run from repo root: python scripts/verify_artifacts.py
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PROTOCOL_COMMIT = "c8935f8c6cdb3dff2331758c86fcaea711232824"
PROTOCOL_SHA256 = "7b1f139e2f003b30e595dffea14feb1de9537c8a1808f5518fb1640b676b0200"
CHECKPOINT_SHA256 = "3e1c94d28125c2f86f3eeca030db3610f2fa679512c29c6b6608b24fc363e0f1"
TOKENIZER_SHA256 = "15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588"
COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
GPT2_XL_REVISION = "15ea56dee5df4983c59b2538573817e1667135e2"
PREFLIGHT_SHA256 = "0a14300d9012ea5acf3be41398d654a07de0dd8650bb227ab2b7cbad0427f4f4"
COMPARISON_SHA256 = "8c044e7e91d31dee57c3c95c23588bd703f60c35826c93c565f29b7f106862d3"
PROMOTED_ARTIFACT_SHA256 = {
    "results/cert_envelope_synth.json": (
        "4c60f0288dd026a822ceb87cb5836fa1c77726f68270877d5d82251e373e54aa"
    ),
    "results/betastar.json": (
        "273cf07354a9b21550421ce9975540d75612f2bd80aa37f84187517f8711db4d"
    ),
    "results/synth_verify.json": (
        "01a74443e9b42845dd99db440c848c9af77f4684060d756e76962cb0bcf4ae2a"
    ),
    "results/cert_envelope_multikey.json": (
        "9289c42c6a12c29ed76f5b04371fd2211a3becaaa8f1b3895bdfcc0268cc0df0"
    ),
    "results/hlm5_1b_faithful_certdosed.json": (
        "b3dc615514c0daf063963894b41ebe617a6eac6f02e6b4ea549211985c9ee732"
    ),
    "results/cert_counterfact_gpt2xl.jsonl": (
        "1bf182bfb30e36729fa457b51d8886c661f4c0f9ac6626a55c7031cc79986ac3"
    ),
    "results/cert_counterfact_gpt2xl_summary.json": (
        "92ee35f7ce7043b0d55ca0e09d1f544339bf6c810c2d96c96fc6e2e654b1a90c"
    ),
}
E7_ARTIFACT_SHA256 = {
    "results/e7_3b_execution_receipt.json": (
        "c89928493758258c83b12b0f067531f11f40f92549f4d7207acc16d72f8cebc5"
    ),
    "results/e7_3b_preflight.json": (
        "93e95bfb21b2efc5c04b69c09b6ede17c07b1fca739cdfd37d6e391d672fbd19"
    ),
    "results/e7_3b_result.json": (
        "42174f0e8d3658e39a8a82b730844fd0c4e77161d4ebc81456243b645141c05c"
    ),
    "results/e7_3b_verdict.json": (
        "83c139aac4f25f98ac58dee5fd468f7a2bae920732a8f7587953033370dcf0ed"
    ),
}

FAILURES: list[str] = []


def load(name: str):
    path = os.path.join(RESULTS, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(name: str) -> list[dict]:
    path = os.path.join(RESULTS, name)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_sha256(name: str) -> str:
    return sha256_path(os.path.join(RESULTS, name))


def stable_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if not condition:
        FAILURES.append(label)


def verify_capacity_search_report() -> None:
    d = load("hlm5_capacity_search_report.json")
    val = d.get("max_passing_pairs")
    check(
        "hlm5_capacity_search_report.json: max_passing_pairs == 105",
        val == 105,
        f"asserted max_passing_pairs == 105, found {val!r}",
    )


def verify_scale_suite_report() -> None:
    d = load("hlm5_scale_suite_report.json")
    capacity = d.get("capacity", [])

    def find(memory_size: int, dim: int):
        for entry in capacity:
            if entry.get("memory_size") == memory_size and entry.get("dim") == dim:
                return entry
        return None

    # The 95/128 point: memory_size=128, dim=64 saturates below the memory's
    # nominal slot count (max_passing_pairs=95 of 128 slots).
    e_95 = find(128, 64)
    ok_95 = e_95 is not None and e_95.get("max_passing_pairs") == 95
    check(
        "hlm5_scale_suite_report.json: capacity point M=128,D=64 max_passing_pairs == 95 (the 95/128 point)",
        ok_95,
        f"asserted capacity[memory_size=128,dim=64].max_passing_pairs == 95, found {e_95.get('max_passing_pairs') if e_95 else 'ENTRY NOT FOUND'!r}",
    )

    # Saturation points: M=128,D=128 -> 128 and M=256,D=256 -> 256.
    e_128 = find(128, 128)
    ok_128 = e_128 is not None and e_128.get("max_passing_pairs") == 128
    check(
        "hlm5_scale_suite_report.json: saturation point M=128,D=128 max_passing_pairs == 128",
        ok_128,
        f"asserted capacity[memory_size=128,dim=128].max_passing_pairs == 128, found {e_128.get('max_passing_pairs') if e_128 else 'ENTRY NOT FOUND'!r}",
    )

    e_256 = find(256, 256)
    ok_256 = e_256 is not None and e_256.get("max_passing_pairs") == 256
    check(
        "hlm5_scale_suite_report.json: saturation point M=256,D=256 max_passing_pairs == 256",
        ok_256,
        f"asserted capacity[memory_size=256,dim=256].max_passing_pairs == 256, found {e_256.get('max_passing_pairs') if e_256 else 'ENTRY NOT FOUND'!r}",
    )


def verify_incontext_mqar() -> None:
    d = load("incontext_mqar.json")

    def find(arm: str, num_pairs: int):
        for entry in d:
            if entry.get("arm") == arm and entry.get("num_pairs") == num_pairs:
                return entry.get("accuracy")
        return None

    t_only = find("transformer-only", 64)
    ok_t = t_only is not None and close(t_only, 0.81640625, 1e-4)
    check(
        "incontext_mqar.json: transformer-only accuracy @ 64 pairs ~= 0.81640625",
        ok_t,
        f"asserted ~=0.81640625 tol 1e-4, found {t_only!r}",
    )

    t_hlm5 = find("transformer+HLM5", 64)
    ok_h = t_hlm5 is not None and close(t_hlm5, 0.11796875, 1e-4)
    check(
        "incontext_mqar.json: transformer+HLM5 accuracy @ 64 pairs ~= 0.11796875",
        ok_h,
        f"asserted ~=0.11796875 tol 1e-4, found {t_hlm5!r}",
    )


def verify_g2c_rare() -> None:
    d = load("g2c_rare.json")
    val = d.get("reachable_fraction")
    ok = val is not None and close(val, 0.01849, 1e-4)
    check(
        "g2c_rare.json: reachable_fraction ~= 0.01849 (tol 1e-4)",
        ok,
        f"asserted ~=0.01849 tol 1e-4, found {val!r}",
    )


def verify_train_summaries() -> None:
    b = load("baseline_train_summary.json")
    h = load("hybrid_train_summary.json")
    check(
        "baseline_train_summary.json: steps == 153000",
        b.get("steps") == 153000,
        f"asserted steps == 153000, found {b.get('steps')!r}",
    )
    check(
        "hybrid_train_summary.json: steps == 153000",
        h.get("steps") == 153000,
        f"asserted steps == 153000, found {h.get('steps')!r}",
    )


def verify_edit_receipts() -> None:
    d = load("edit_receipts.json")
    checks = {
        "receipts_verified": True,
        "tamper_value_detected": True,
        "tamper_prediction_detected": True,
        "tamper_prompt_detected": True,
    }
    ok = all(d.get(k) == v for k, v in checks.items())
    detail = ", ".join(f"{k}={d.get(k)!r}" for k in checks)
    check(
        "edit_receipts.json: receipts_verified and all tamper_*_detected flags == true",
        ok,
        detail,
    )


def verify_lm_kb_inject() -> None:
    d = load("lm_kb_inject.json")
    sweep = {entry.get("K"): entry.get("inject_flip") for entry in d.get("sweep", [])}
    base_acc = d.get("base_acc")
    expected = {16: 1.0, 32: 1.0, 48: 0.97917, 64: 0.984375, 80: 0.9875}

    ok_base = base_acc is not None and close(base_acc, 1.0, 1e-4)
    check(
        "lm_kb_inject.json (D=64): base_acc ~= 1.0",
        ok_base,
        f"asserted base_acc ~=1.0 tol 1e-4, found {base_acc!r}",
    )

    for k, exp in expected.items():
        val = sweep.get(k)
        ok = val is not None and close(val, exp, 1e-4)
        check(
            f"lm_kb_inject.json (D=64): inject_flip @ K={k} ~= {exp}",
            ok,
            f"asserted ~={exp} tol 1e-4, found {val!r}",
        )


def verify_no_tax(name: str) -> None:
    d = load(name)
    baseline = d.get("baseline_val_ppl")
    hybrid = d.get("hybrid_val_ppl")
    ok = baseline is not None and hybrid is not None
    check(
        f"{name}: loadable, contains baseline/hybrid PPL fields",
        ok,
        f"baseline_val_ppl={baseline!r}, hybrid_val_ppl={hybrid!r}, delta_pct={d.get('delta_pct')!r}",
    )


def verify_refresh_contract(
    name: str,
    payload: dict,
    producer: str,
    preflight_sha256: str,
    *,
    counterfact: bool = False,
) -> None:
    contract = payload.get("certificate_contract", {})
    producer_path = os.path.join(ROOT, *producer.split("/"))
    expected = {
        "schema": "hlm5-certificate-contract-v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "eps": 0.0,
        "arithmetic_dtype": "float64",
        "producer": producer,
        "producer_sha256": sha256_path(producer_path),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": "float32",
        "seed": 0,
    }
    if counterfact:
        expected.update(
            {
                "model_id": "gpt2-xl",
                "model_revision": GPT2_XL_REVISION,
                "data_sha256": COUNTERFACT_SHA256,
            }
        )
    else:
        expected.update(
            {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "tokenizer_sha256": TOKENIZER_SHA256,
            }
        )
    mismatches = {
        key: {"found": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    boundary = payload.get("certificate_boundary_dtypes", {})
    boundary_ok = bool(boundary) and all(
        dtype == "float64" for dtype in boundary.values()
    )
    check(
        f"{name}: exact-sign contract, hashes, and runtime float64 boundary",
        not mismatches and boundary_ok,
        f"contract_mismatches={mismatches}, boundary={boundary!r}",
    )


def verify_certificate_refresh() -> None:
    preflight = load("certificate_refresh_preflight.json")
    comparison = load("certificate_refresh_comparison.json")
    manifest = load("certificate_refresh_manifest.json")
    preflight_sha256 = result_sha256("certificate_refresh_preflight.json")

    check(
        "certificate refresh receipts: READY preflight and valid changed-claim verdict",
        preflight_sha256 == PREFLIGHT_SHA256
        and result_sha256("certificate_refresh_comparison.json")
        == COMPARISON_SHA256
        and preflight.get("status") == "READY"
        and preflight.get("implementation_commit")
        == "ab4e28ca9c81055289a5eebbdfda8a8cc3eaed64"
        and comparison.get("preflight_sha256") == preflight_sha256
        and comparison.get("verdict") == "VALID_REFRESH_CHANGED_CLAIMS"
        and comparison.get("validity_failures") == [],
        f"preflight={preflight.get('status')!r}, verdict={comparison.get('verdict')!r}, "
        f"failures={comparison.get('validity_failures')!r}",
    )

    one = load("cert_envelope_synth.json")
    verify_refresh_contract(
        "cert_envelope_synth.json",
        one,
        "scripts/run_1b_certificate.py",
        preflight_sha256,
    )
    one_valid = one.get("validity", {})
    check(
        "cert_envelope_synth.json: 941/1200 reachable and all candidate bars pass",
        one.get("pool") == 1200
        and one.get("envelope", {}).get("reachable_rate") == 0.784
        and one.get("envelope", {}).get("risk_label_counts")
        == {"safe": 888, "narrow": 38, "brittle": 15, "unreachable": 259}
        and one_valid.get("candidate_doses_checked") == 941
        and one_valid.get("all_candidate_doses_inside_open_interval") is True
        and one_valid.get("all_candidate_margins_positive") is True
        and one.get("residual_synthesis", {}).get("rescued") == 40,
        f"validity={one_valid!r}",
    )

    beta = load("betastar.json")
    verify_refresh_contract(
        "betastar.json",
        beta,
        "scripts/run_1b_betastar.py",
        preflight_sha256,
    )
    beta_valid = beta.get("validity", {})
    check(
        "betastar.json: 941 certified doses, median beta 72.134, margin 18.39",
        beta.get("n_reachable") == 941
        and beta.get("median_beta_star") == 72.134
        and beta.get("median_worst_margin_betastar") == 18.39
        and beta_valid.get("grid_failures") == 0
        and beta_valid.get("all_candidate_doses_inside_open_interval") is True
        and beta_valid.get("all_candidate_margins_positive") is True,
        f"validity={beta_valid!r}",
    )

    synth = load("synth_verify.json")
    verify_refresh_contract(
        "synth_verify.json",
        synth,
        "scripts/run_1b_synth_verify.py",
        preflight_sha256,
    )
    check(
        "synth_verify.json: native-boundary synthesis flips 40/40 at median beta 20",
        synth.get("n_unreachable_tested") == 40
        and synth.get("flip_with_unit_Wy") == 0
        and synth.get("flip_with_synth_residual") == 40
        and synth.get("median_beta_needed") == 20
        and synth.get("validity", {}).get("all_synthesis_margins_positive") is True,
        f"validity={synth.get('validity')!r}",
    )

    multi = load("cert_envelope_multikey.json")
    verify_refresh_contract(
        "cert_envelope_multikey.json",
        multi,
        "scripts/run_1b_envelope_multikey.py",
        preflight_sha256,
    )
    multi_valid = multi.get("validity", {})
    check(
        "cert_envelope_multikey.json: 60 keys, 57151 valid candidates, scalar/vector match",
        multi.get("n_keys") == 60
        and multi.get("aggregates", {}).get("reachable_fraction", {}).get("mean")
        == 0.7938
        and multi.get("aggregates", {}).get("reachable_fraction", {}).get("std")
        == 0.0221
        and multi_valid.get("candidate_doses_checked") == 57151
        and multi_valid.get("all_candidate_doses_inside_open_interval") is True
        and multi_valid.get("all_candidate_margins_positive") is True
        and multi_valid.get("original_scalar_matches_staged_one_key") is True
        and multi_valid.get("original_vectorized_matches_scalar") is True,
        f"validity={multi_valid!r}",
    )

    faithful = load("hlm5_1b_faithful_certdosed.json")
    verify_refresh_contract(
        "hlm5_1b_faithful_certdosed.json",
        faithful,
        "scripts/run_1b_faithful_certdosed.py",
        preflight_sha256,
    )
    arms = faithful.get("arms", {})
    faithful_valid = faithful.get("validity", {})
    check(
        "hlm5_1b_faithful_certdosed.json: 9/17 -> 13/17 -> 15/17, locality 8/8",
        arms.get("global_repro", {}).get("eff_count") == "9/17"
        and arms.get("cert_naive", {}).get("eff_count") == "13/17"
        and arms.get("cert_synth", {}).get("eff_count") == "15/17"
        and all(
            arm.get("locality_bit_identical") == "8/8"
            for arm in arms.values()
        )
        and faithful_valid.get(
            "all_candidate_doses_inside_with_positive_margin"
        )
        is True
        and faithful_valid.get("ordinary_forward_naive_agreement") is True
        and faithful_valid.get("ordinary_forward_synth_agreement") is True,
        f"validity={faithful_valid!r}",
    )

    rows = load_jsonl("cert_counterfact_gpt2xl.jsonl")
    counterfact = load("cert_counterfact_gpt2xl_summary.json")
    verify_refresh_contract(
        "cert_counterfact_gpt2xl_summary.json",
        counterfact,
        "scripts/run_cert_counterfact_gpt2xl.py",
        preflight_sha256,
        counterfact=True,
    )
    summary_contract = counterfact.get("certificate_contract", {})
    row_contract = {
        key: value
        for key, value in summary_contract.items()
        if key != "jsonl_sha256"
    }
    row_contract_sha256 = stable_json_sha256(row_contract)
    check(
        "cert_counterfact_gpt2xl.jsonl: 1000 ordered unique contract-bound rows",
        len(rows) == 1000
        and [row.get("usable_index") for row in rows] == list(range(1000))
        and len({row.get("case_id") for row in rows}) == 1000
        and all(
            row.get("certificate_contract_sha256") == row_contract_sha256
            for row in rows
        )
        and all(
            bool(row.get("certificate_boundary_dtypes"))
            and all(
                dtype == "float64"
                for dtype in row["certificate_boundary_dtypes"].values()
            )
            for row in rows
        )
        and summary_contract.get("jsonl_sha256")
        == result_sha256("cert_counterfact_gpt2xl.jsonl"),
        f"rows={len(rows)}, jsonl_sha256={summary_contract.get('jsonl_sha256')!r}",
    )
    check(
        "cert_counterfact_gpt2xl_summary.json: identity and 1000/1000 reachability bars",
        counterfact.get("n_records") == 1000
        and counterfact.get("fraction_reachable") == 1.0
        and counterfact.get("fraction_head_reachable") == 1.0
        and counterfact.get("fraction_hard_blocker") == 0.0
        and counterfact.get("identity_check_first_record", {}).get("allclose")
        is True
        and counterfact.get("identity_check_first_record", {}).get("argmax_match")
        is True,
        f"identity={counterfact.get('identity_check_first_record')!r}",
    )

    primary_names = (
        "cert_envelope_synth.json",
        "betastar.json",
        "synth_verify.json",
        "cert_envelope_multikey.json",
        "hlm5_1b_faithful_certdosed.json",
        "cert_counterfact_gpt2xl.jsonl",
        "cert_counterfact_gpt2xl_summary.json",
    )
    actual_hashes = {
        f"results/{name}": result_sha256(name) for name in primary_names
    }
    check(
        "certificate_refresh_manifest.json: binds exact receipts and promoted payloads",
        manifest.get("schema") == "hlm5-certificate-refresh-promotion-manifest-v1"
        and manifest.get("verdict") == "VALID_REFRESH_CHANGED_CLAIMS"
        and manifest.get("preflight_sha256") == PREFLIGHT_SHA256
        and manifest.get("comparison_sha256") == COMPARISON_SHA256
        and manifest.get("artifact_sha256") == PROMOTED_ARTIFACT_SHA256
        and actual_hashes == PROMOTED_ARTIFACT_SHA256,
        f"artifact_sha256={manifest.get('artifact_sha256')!r}",
    )


def verify_e7_3b_portability() -> None:
    preflight = load("e7_3b_preflight.json")
    receipt = load("e7_3b_execution_receipt.json")
    result = load("e7_3b_result.json")
    verdict = load("e7_3b_verdict.json")
    manifest = load("e7_3b_manifest.json")
    actual_hashes = {
        name: result_sha256(name.removeprefix("results/"))
        for name in E7_ARTIFACT_SHA256
    }
    check(
        "E7 3B manifest: exact promoted artifacts and valid scientific FAIL",
        actual_hashes == E7_ARTIFACT_SHA256
        and manifest.get("schema") == "hlm5-e7-3b-promotion-manifest-v1"
        and manifest.get("artifact_sha256") == E7_ARTIFACT_SHA256
        and manifest.get("verdict") == "FAIL_3B_PORTABILITY"
        and manifest.get("validity_failures") == [],
        f"actual_hashes={actual_hashes!r}",
    )

    identity = preflight.get("baseline_only_head_identity", {})
    invariants = preflight.get("model_invariants", {})
    check(
        "E7 3B preflight: frozen 3.075B BF16 trunk and independent head identity",
        preflight.get("status") == "READY"
        and preflight.get("implementation_commit")
        == "436396deba0234987ec5a0a8b66dbda406a7beea"
        and invariants.get("parameter_count") == 3_075_098_624
        and invariants.get("tied_embeddings") is True
        and invariants.get("head_bias_is_none") is True
        and invariants.get("all_parameters_frozen") is True
        and invariants.get("floating_parameter_dtypes") == ["bfloat16"]
        and identity.get("argmax_equal") is True
        and identity.get("max_abs_logit_difference") == 0.0625
        and preflight.get("candidate_outputs_computed") is False,
        f"invariants={invariants!r}, identity={identity!r}",
    )
    check(
        "E7 3B execution receipt: tests passed before candidate access",
        receipt.get("status") == "READY"
        and receipt.get("preflight_sha256")
        == E7_ARTIFACT_SHA256["results/e7_3b_preflight.json"]
        and receipt.get("tests_returncode") == 0
        and receipt.get("candidate_outputs_computed_before_receipt") is False,
        f"status={receipt.get('status')!r}, tests={receipt.get('tests_returncode')!r}",
    )

    validity = result.get("validity", {})
    contract = result.get("certificate_contract", {})
    check(
        "E7 3B result: every implementation-validity bar passed",
        result.get("schema") == "hlm5-e7-3b-result-v1"
        and result.get("status") == "COMPLETE"
        and validity
        and all(value is True for key, value in validity.items() if key.startswith("all_"))
        and validity.get("original_scalar_vector_match") is True
        and validity.get("keys_checked") == 60
        and validity.get("target_pool_count") == 1200
        and validity.get("gate_rows_checked") == 80
        and contract.get("producer_sha256")
        == sha256_path(os.path.join(ROOT, "scripts", "run_e7_3b.py"))
        and contract.get("preflight_sha256")
        == E7_ARTIFACT_SHA256["results/e7_3b_preflight.json"]
        and contract.get("execution_receipt_sha256")
        == E7_ARTIFACT_SHA256["results/e7_3b_execution_receipt.json"],
        f"validity={validity!r}",
    )

    direct = result.get("direct_certified_injection", {})
    failed_direct = [
        row for row in direct.get("rows", []) if not row.get("ordinary_head_success")
    ]
    faithful = result.get("faithful", {})
    faithful_summary = faithful.get("summary", {})
    check(
        "E7 3B scientific record: 1027/1028 direct, 11/11 faithful, zero false gates",
        direct.get("accepted_count") == 1028
        and direct.get("ordinary_head_success_count") == 1027
        and len(failed_direct) == 1
        and failed_direct[0].get("target_id") == 922
        and close(failed_direct[0].get("float64_margin"), 2.2585672901487284, 1e-12)
        and failed_direct[0].get("ordinary_bfloat16_margin") == 0.0
        and faithful.get("accepted_count") == 11
        and faithful_summary.get("accepted_exact_full_successes") == 11
        and faithful_summary.get("false_gate_applications") == 0
        and faithful_summary.get("neutral_bit_identical") == 8,
        f"direct={direct.get('ordinary_head_success_count')}/{direct.get('accepted_count')}, "
        f"faithful={faithful_summary!r}",
    )
    check(
        "E7 3B verdict: implementation-valid FAIL at the float64-to-BF16 boundary",
        verdict.get("schema") == "hlm5-e7-3b-verdict-v1"
        and verdict.get("binding_bars_passed") is True
        and verdict.get("validity_failures") == []
        and verdict.get("verdict") == "FAIL_3B_PORTABILITY"
        and verdict.get("scientific_failures")
        == ["not every directly certified target succeeded"]
        and verdict.get("result_sha256")
        == E7_ARTIFACT_SHA256["results/e7_3b_result.json"],
        f"verdict={verdict!r}",
    )


def main() -> int:
    verify_capacity_search_report()
    verify_scale_suite_report()
    verify_incontext_mqar()
    verify_g2c_rare()
    verify_train_summaries()
    verify_edit_receipts()
    verify_lm_kb_inject()
    verify_no_tax("g2a_no_tax.json")
    verify_no_tax("g3a_no_tax.json")
    verify_certificate_refresh()
    verify_e7_3b_portability()

    # Count PASS/FAIL lines were already printed; just summarize failures.
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("RESULT: all checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
