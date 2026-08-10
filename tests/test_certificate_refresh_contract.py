"""CPU-only contract tests for the exact-sign certificate refresh."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hlm5 import artifact_contract as contract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "scripts" / "compare_certificate_refresh.py"
COMPARE_SPEC = importlib.util.spec_from_file_location(
    "certificate_refresh_comparison",
    COMPARE_PATH,
)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
compare = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = compare
COMPARE_SPEC.loader.exec_module(compare)
SYNTH_PATH = ROOT / "scripts" / "run_1b_synth_verify.py"
SYNTH_SPEC = importlib.util.spec_from_file_location(
    "certificate_refresh_synthesis",
    SYNTH_PATH,
)
assert SYNTH_SPEC is not None and SYNTH_SPEC.loader is not None
synthesis = importlib.util.module_from_spec(SYNTH_SPEC)
sys.modules[SYNTH_SPEC.name] = synthesis
SYNTH_SPEC.loader.exec_module(synthesis)
FIGURE_PATH = ROOT / "scripts" / "make_figures.py"
FIGURE_SPEC = importlib.util.spec_from_file_location(
    "certificate_refresh_figures",
    FIGURE_PATH,
)
assert FIGURE_SPEC is not None and FIGURE_SPEC.loader is not None
figures = importlib.util.module_from_spec(FIGURE_SPEC)
sys.modules[FIGURE_SPEC.name] = figures
FIGURE_SPEC.loader.exec_module(figures)


def test_registered_sources_cover_runtime_helpers_and_contract_tests() -> None:
    required = {
        "hlm5/artifact_contract.py",
        "hlm5/certify.py",
        "hlm5/io.py",
        "hlm5/memory.py",
        "hlm5/model.py",
        "scripts/compare_certificate_refresh.py",
        "scripts/preflight_certificate_refresh.py",
        "tests/test_certificate_refresh_contract.py",
    }
    assert required <= set(contract.REGISTERED_SOURCE_PATHS)


def test_registered_refresh_commands_bootstrap_repo_before_package_import() -> None:
    scripts = (
        "scripts/preflight_certificate_refresh.py",
        "scripts/run_1b_certificate.py",
        "scripts/run_1b_betastar.py",
        "scripts/run_1b_synth_verify.py",
        "scripts/run_1b_envelope_multikey.py",
        "scripts/run_1b_faithful_certdosed.py",
        "scripts/run_cert_counterfact_gpt2xl.py",
        "scripts/compare_certificate_refresh.py",
        "scripts/analyze_cert_vs_editors.py",
        "scripts/make_figures.py",
    )
    for relative in scripts:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert source.index("sys.path.insert") < source.index(
            "from hlm5"
        ), relative


def test_registered_sha256_values_are_well_formed_and_tokenizer_matches() -> None:
    for digest in (
        contract.PROTOCOL_SHA256,
        contract.EXPECTED_CHECKPOINT_SHA256,
        contract.EXPECTED_TOKENIZER_SHA256,
        contract.EXPECTED_COUNTERFACT_SHA256,
    ):
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
    assert contract.sha256_file(contract.TOKENIZER_PATH) == (
        contract.EXPECTED_TOKENIZER_SHA256
    )


def test_registered_source_verification_detects_post_preflight_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    monkeypatch.setattr(contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        contract,
        "REGISTERED_SOURCE_PATHS",
        ("first.py", "second.py"),
    )
    receipt = {
        "source_sha256": {
            "first.py": contract.sha256_file(first),
            "second.py": contract.sha256_file(second),
        }
    }

    contract.verify_registered_sources(receipt)
    second.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        contract.verify_registered_sources(receipt)


def test_registered_source_verification_rejects_incomplete_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "only.py"
    path.write_text("only\n", encoding="utf-8")
    monkeypatch.setattr(contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        contract,
        "REGISTERED_SOURCE_PATHS",
        ("only.py", "missing.py"),
    )
    receipt = {"source_sha256": {"only.py": contract.sha256_file(path)}}

    with pytest.raises(RuntimeError, match="source registry mismatch"):
        contract.verify_registered_sources(receipt)


def test_atomic_scientific_json_is_staging_only_and_finite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    monkeypatch.setattr(contract, "STAGING_DIR", staging)
    output = staging / "result.json"

    contract.atomic_write_json(output, {"value": 1.25})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1.25}
    assert output.read_bytes().endswith(b"\n")

    with pytest.raises(ValueError, match="Out of range float values"):
        contract.atomic_write_json(staging / "nan.json", {"value": float("nan")})
    with pytest.raises(ValueError, match="must stay under"):
        contract.atomic_write_json(tmp_path / "outside.json", {"value": 1})


def test_json_object_loader_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        contract.load_json_object(path)


def _row(index: int, contract_sha256: str) -> dict[str, object]:
    return {
        "certificate_contract_sha256": contract_sha256,
        "case_id": f"case-{index}",
        "usable_index": index,
    }


def test_contract_jsonl_accepts_only_an_exact_unique_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    expected = "a" * 64
    rows = [_row(0, expected), _row(1, expected)]
    path.write_text(
        "".join(f"{contract.stable_json(row)}\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )

    assert contract.load_contract_jsonl(
        path,
        expected,
        require_prefix=True,
    ) == rows

    gap = [_row(0, expected), _row(2, expected)]
    path.write_text(
        "".join(f"{contract.stable_json(row)}\n" for row in gap),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(RuntimeError, match="ordered prefix"):
        contract.load_contract_jsonl(path, expected, require_prefix=True)


@pytest.mark.parametrize(
    "payload, message",
    [
        ('{"usable_index":0}\n', "different certificate contract"),
        ('{"certificate_contract_sha256":"x",', "malformed staged JSONL"),
        ("\n", "blank JSONL row"),
    ],
)
def test_contract_jsonl_fails_closed_on_invalid_rows(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(payload, encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match=message):
        contract.load_contract_jsonl(path, "expected", require_prefix=True)


def test_contract_jsonl_rejects_duplicate_case_or_index(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = "b" * 64
    duplicate = [_row(0, expected), {**_row(1, expected), "case_id": "case-0"}]
    path.write_text(
        "".join(f"{contract.stable_json(row)}\n" for row in duplicate),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(RuntimeError, match="duplicate case_id/usable_index"):
        contract.load_contract_jsonl(path, expected, require_prefix=True)


def test_float64_boundary_record_is_runtime_evidence() -> None:
    record = contract.float64_boundary_record(
        head=torch.zeros(2, 2, dtype=torch.float64),
        hidden=torch.zeros(2, dtype=torch.float64),
        direction=torch.ones(2, dtype=torch.float64),
    )
    assert record == {
        "head": "float64",
        "hidden": "float64",
        "direction": "float64",
    }
    with pytest.raises(RuntimeError, match="not float64"):
        contract.float64_boundary_record(
            head=torch.zeros(2, 2, dtype=torch.float32)
        )


def test_comparison_boundary_requires_head_hidden_and_direction() -> None:
    assert compare.check_boundary(
        {
            "certificate_boundary_dtypes": {
                "head": "float64",
                "hidden": "float64",
                "direction": "float64",
            }
        },
        "valid",
    ) == []
    failures = compare.check_boundary(
        {"certificate_boundary_dtypes": {"head": "float32"}},
        "invalid",
    )
    assert any("float32" in failure for failure in failures)
    assert any("hidden" in failure for failure in failures)
    assert any("direction" in failure for failure in failures)


def test_synthesis_flip_uses_native_injection_rounding() -> None:
    head = torch.eye(2, dtype=torch.float32)
    hidden = torch.zeros(2, dtype=torch.float32)
    direction = torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.float64)

    assert int((head.to(torch.float64) @ direction).argmax()) == 1
    assert synthesis.native_dosed_argmax(head, hidden, direction, 1.0) == 0


def test_certificate_figure_accepts_registered_null_upper_bound() -> None:
    assert figures._upper_bound({"U": None}) == float("inf")
    assert figures._upper_bound({"U": "inf"}) == float("inf")
    assert figures._upper_bound({"U": 3.5}) == 3.5


def test_all_registered_producers_have_exact_float64_source_boundaries() -> None:
    producers = [producer for _, _, producer in compare.HLM_SPECS]
    producers.append(compare.COUNTERFACT_PRODUCER)
    assert {
        producer: compare.check_producer_source(producer)
        for producer in producers
    } == {producer: [] for producer in producers}


def test_comparison_checks_the_full_hlm5_contract() -> None:
    producer = "scripts/run_1b_certificate.py"
    sample = compare.HLM_SAMPLE_CONTRACTS["one_key"]
    preflight = {
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        }
    }
    preflight_sha256 = "c" * 64
    payload = {
        "certificate_contract": {
            "schema": contract.CONTRACT_SCHEMA,
            "protocol_commit": contract.PROTOCOL_COMMIT,
            "protocol_sha256": contract.PROTOCOL_SHA256,
            "eps": 0.0,
            "arithmetic_dtype": "float64",
            "producer": producer,
            "producer_sha256": contract.sha256_file(ROOT / producer),
            "preflight_sha256": preflight_sha256,
            "model_forward_dtype": "float32",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "seed": 0,
            "sample_contract": sample,
            "upstream_sha256": {},
            "checkpoint": contract.repo_relative(contract.CHECKPOINT_PATH),
            "checkpoint_sha256": contract.EXPECTED_CHECKPOINT_SHA256,
            "tokenizer": contract.repo_relative(contract.TOKENIZER_PATH),
            "tokenizer_sha256": contract.EXPECTED_TOKENIZER_SHA256,
        }
    }

    assert compare.check_contract(
        payload,
        producer,
        sample,
        preflight,
        preflight_sha256,
        "hlm5",
    ) == []
    payload["certificate_contract"]["sample_contract"] = {"facts": "changed"}
    failures = compare.check_contract(
        payload,
        producer,
        sample,
        preflight,
        preflight_sha256,
        "hlm5",
    )
    assert any("sample_contract" in failure for failure in failures)


def test_row_level_comparison_records_field_changes() -> None:
    result = compare.compare_categorical_rows(
        [{"i": 0, "risk": "safe"}, {"i": 1, "risk": "narrow"}],
        [{"i": 0, "risk": "safe"}, {"i": 1, "risk": "brittle"}],
        key="i",
        fields=("risk",),
    )
    assert result["changed_rows"] == 1
    assert result["changes"] == [
        {
            "i": 1,
            "changes": {"risk": {"old": "narrow", "new": "brittle"}},
        }
    ]


def test_promoted_headlines_preserve_all_but_registered_changed_claim() -> None:
    promoted = compare.load_release_headlines()
    expected = copy.deepcopy(compare.OLD_HEADLINES)
    expected["faithful_synth"] = "15/17"
    assert promoted == expected


def test_open_interval_helper_handles_unbounded_upper_limit() -> None:
    assert compare.inside_open_interval(1.0, None, 2.0)
    assert compare.inside_open_interval(1.0, 3.0, 2.0)
    assert not compare.inside_open_interval(1.0, 3.0, 1.0)
    assert not compare.inside_open_interval(1.0, 3.0, 3.0)


def _runtime_boundary() -> dict[str, str]:
    return {
        "head": "float64",
        "hidden": "float64",
        "direction": "float64",
    }


def _synthetic_contract(
    producer: str,
    sample_contract: dict[str, object],
    preflight_sha256: str,
    upstream: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "schema": contract.CONTRACT_SCHEMA,
        "protocol_commit": contract.PROTOCOL_COMMIT,
        "protocol_sha256": contract.PROTOCOL_SHA256,
        "eps": 0.0,
        "arithmetic_dtype": "float64",
        "producer": producer,
        "producer_sha256": contract.sha256_file(ROOT / producer),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": "float32",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "seed": 0,
        "sample_contract": sample_contract,
        "upstream_sha256": dict(sorted((upstream or {}).items())),
        "checkpoint": contract.repo_relative(contract.CHECKPOINT_PATH),
        "checkpoint_sha256": contract.EXPECTED_CHECKPOINT_SHA256,
        "tokenizer": contract.repo_relative(contract.TOKENIZER_PATH),
        "tokenizer_sha256": contract.EXPECTED_TOKENIZER_SHA256,
    }


def _synthetic_hlm_payloads(
    preflight_sha256: str,
    one_key_upstream: dict[str, str],
) -> dict[str, dict[str, object]]:
    envelope_rows = []
    for target_id in range(1200):
        reachable = target_id < 800
        envelope_rows.append(
            {
                "target_id": target_id,
                "token": f" token-{target_id}",
                "reachable": reachable,
                "L": 1.0,
                "U": 12.0 if reachable else 0.5,
                "slack": 11.0 if reachable else -0.5,
                "risk": "safe" if reachable else "unreachable",
                "beta_candidate": 2.05 if reachable else None,
                "margin_at_candidate": 1.0 if reachable else None,
                "blocker": " blocker",
                "blocker_a": None,
                "blocker_b": None,
            }
        )
    synthesis_rows = [
        {
            "target_id": target_id,
            "token": f" token-{target_id}",
            "minimum_margin": 1.0,
            "rescued": True,
        }
        for target_id in range(800, 840)
    ]
    one_validity = {
        "candidate_doses_checked": 800,
        "all_candidate_doses_inside_open_interval": True,
        "all_candidate_margins_positive": True,
        "minimum_candidate_margin": 1.0,
        "synthesis_directions_checked": 40,
        "all_synthesis_margins_positive": True,
        "minimum_synthesis_margin": 1.0,
    }
    one = {
        "certificate_contract": _synthetic_contract(
            "scripts/run_1b_certificate.py",
            contract.ONE_KEY_SAMPLE_CONTRACT,
            preflight_sha256,
        ),
        "certificate_boundary_dtypes": _runtime_boundary(),
        "pool": 1200,
        "envelope": {
            "reachable_rate": 0.667,
            "unreachable_rate": 0.333,
            "median_slack_reachable": 11.0,
            "risk_label_counts": {
                "safe": 800,
                "narrow": 0,
                "brittle": 0,
                "unreachable": 400,
            },
            "per_target": envelope_rows,
        },
        "residual_synthesis": {
            "unreachable_tested": 40,
            "rescued": 40,
            "rescue_rate": 1.0,
            "examples": [],
            "per_target": synthesis_rows,
        },
        "validity": one_validity,
    }

    beta_rows = [
        {
            "target_id": target_id,
            "L": 1.0,
            "U": 3.0,
            "beta_star": 2.0,
            "worst_margin_beta_star": 1.0,
            "heuristic_beta": 2.05,
            "worst_margin_heuristic": 0.5,
        }
        for target_id in range(2)
    ]
    beta = {
        "certificate_contract": _synthetic_contract(
            "scripts/run_1b_betastar.py",
            contract.BETA_STAR_SAMPLE_CONTRACT,
            preflight_sha256,
        ),
        "certificate_boundary_dtypes": _runtime_boundary(),
        "n_reachable": 2,
        "median_beta_star": 2.0,
        "median_worst_margin_betastar": 1.0,
        "median_worst_margin_heuristic": 0.5,
        "median_margin_gain": 0.5,
        "per_target": beta_rows,
        "validity": {
            "candidate_doses_checked": 2,
            "envelope_reachable": 2,
            "grid_failures": 0,
            "all_candidate_doses_inside_open_interval": True,
            "all_candidate_margins_positive": True,
            "minimum_candidate_margin": 1.0,
        },
    }

    synth_rows = [
        {
            "target_id": target_id,
            "unit_value_flipped": False,
            "synthesis_min_margin": 1.0,
            "synth_value_flipped": True,
            "first_flip_beta": 10,
        }
        for target_id in range(800, 840)
    ]
    synthesis_flip = {
        "certificate_contract": _synthetic_contract(
            "scripts/run_1b_synth_verify.py",
            contract.SYNTHESIS_FLIP_SAMPLE_CONTRACT,
            preflight_sha256,
        ),
        "certificate_boundary_dtypes": _runtime_boundary(),
        "n_unreachable_tested": 40,
        "flip_with_unit_Wy": 0,
        "flip_with_synth_residual": 40,
        "rescue_flip_rate": 1.0,
        "median_beta_needed": 10,
        "max_beta_needed": 10,
        "min_beta_needed": 10,
        "per_target": synth_rows,
        "validity": {
            "synthesis_directions_checked": 40,
            "all_synthesis_margins_positive": True,
            "minimum_synthesis_margin": 1.0,
        },
    }

    key_rows = []
    prompts: dict[str, list[str]] = {
        "novel": [],
        "real": [],
        "generic": [],
    }
    for index in range(60):
        category = ("novel", "real", "generic")[index // 20]
        prompt = (
            contract.MULTI_KEY_SAMPLE_CONTRACT["original_key"]
            if index == 0
            else f"prompt-{index}"
        )
        prompts[category].append(prompt)
        key_rows.append(
            {
                "prompt": prompt,
                "category": category,
                "is_original_key": index == 0,
                "base_argmax": " base",
                "n_reachable": 800,
                "reachable_fraction": 0.6667,
                "risk_counts": {
                    "safe": 800,
                    "narrow": 0,
                    "brittle": 0,
                    "unreachable": 400,
                },
                "median_slack_reachable": 11.0,
                "unreachable_hard_blocker": 400,
                "unreachable_interval_only": 0,
                "candidate_doses_checked": 800,
                "all_candidate_doses_inside_open_interval": True,
                "all_candidate_margins_positive": True,
                "minimum_candidate_margin": 1.0,
            }
        )
    risk_counts = {
        "safe": 800,
        "narrow": 0,
        "brittle": 0,
        "unreachable": 400,
    }
    repeated_stats = {
        "reachable_fraction": compare.rounded_population_stats([0.6667] * 60),
        "median_slack_reachable": compare.rounded_population_stats([11.0] * 60),
        "unreachable_hard_blocker": compare.rounded_population_stats([400] * 60),
        "unreachable_interval_only": compare.rounded_population_stats([0] * 60),
    }
    repeated_stats["per_category"] = {
        category: {
            "n": 20,
            "mean_reachable_fraction": 0.6667,
            "mean_median_slack": 11.0,
            "mean_unreachable": 400.0,
            "mean_hard_blocker": 400.0,
            "mean_interval_only": 0.0,
        }
        for category in ("novel", "real", "generic")
    }
    repeated_stats["original_key_typicality"] = {
        "original_reachable_fraction": 0.6667,
        "percentile_rank": 1.0,
        "z_score": None,
        "within_1_std": True,
    }
    multi = {
        "certificate_contract": _synthetic_contract(
            "scripts/run_1b_envelope_multikey.py",
            contract.MULTI_KEY_SAMPLE_CONTRACT,
            preflight_sha256,
            one_key_upstream,
        ),
        "certificate_boundary_dtypes": _runtime_boundary(),
        "pool": 1200,
        "n_keys": 60,
        "original_key": {
            "stored": {
                "reachable_rate": 0.667,
                "median_slack_reachable": 11.0,
                "risk_label_counts": risk_counts,
            },
            "recomputed_scalar": {
                "reachable_fraction": 0.6667,
                "risk_counts": risk_counts,
                "median_slack_reachable": 11.0,
                "unreachable_total": 400,
                "unreachable_hard_blocker": 400,
                "unreachable_interval_only": 0,
            },
            "recompute_matches_stored": True,
            "vectorized_matches_scalar": True,
        },
        "aggregates": repeated_stats,
        "keys": key_rows,
        "prompt_list": prompts,
        "validity": {
            "keys_checked": 60,
            "candidate_doses_checked": 48000,
            "all_candidate_doses_inside_open_interval": True,
            "all_candidate_margins_positive": True,
            "minimum_candidate_margin": 1.0,
            "all_risk_counts_sum_to_pool": True,
            "original_scalar_matches_staged_one_key": True,
            "original_vectorized_matches_scalar": True,
        },
    }

    locality_rows = [
        {
            "prompt": f"neutral-{index}",
            "argmax_match": True,
            "logits_bit_identical": True,
        }
        for index in range(8)
    ]
    fact_rows = []
    for index in range(17):
        fact_rows.append(
            {
                "i": index,
                "L": 1.0,
                "U": 3.0,
                "reachable": True,
                "rescued": False,
                "beta_star": 2.0,
                "margin_after": 1.0,
                "flip_global": False,
                "pred_global": " base",
                "generalization_global": 0.0,
                "flip_cert_naive": True,
                "pred_cert_naive": " target",
                "generalization_cert_naive": 0.0,
                "flip_cert_synth": True,
                "pred_cert_synth": " target",
                "generalization_cert_synth": 0.0,
            }
        )

    def arm(efficacy: float, count: str) -> dict[str, object]:
        return {
            "boost": 1.0,
            "summary": {
                "eff": [efficacy, efficacy, efficacy],
                "gen": [0.0, 0.0, 0.0],
                "loc": [1.0, 1.0, 1.0],
            },
            "eff_count": count,
            "gen_paraphrase_rate": 0.0,
            "paraphrase_gate_fire_rate": 0.0,
            "locality_argmax": "8/8",
            "locality_bit_identical": "8/8",
            "locality_rows": locality_rows,
        }

    faithful = {
        "certificate_contract": _synthetic_contract(
            "scripts/run_1b_faithful_certdosed.py",
            contract.FAITHFUL_SAMPLE_CONTRACT,
            preflight_sha256,
        ),
        "certificate_boundary_dtypes": _runtime_boundary(),
        "n_facts": 17,
        "paraphrase_gate_fire_rate": 0.0,
        "arms": {
            "global_repro": arm(0.0, "0/17"),
            "cert_naive": arm(1.0, "17/17"),
            "cert_synth": arm(1.0, "17/17"),
        },
        "per_fact": fact_rows,
        "validity": {
            "candidate_doses_checked": 17,
            "all_candidate_doses_inside_with_positive_margin": True,
            "ordinary_forward_naive_agreement": True,
            "ordinary_forward_synth_agreement": True,
            "all_arms_locality_bit_identical": True,
        },
    }
    return {
        "one_key": one,
        "beta_star": beta,
        "synthesis_flip": synthesis_flip,
        "multi_key": multi,
        "faithful": faithful,
    }


def test_hlm_comparator_accepts_coherent_rows_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compare, "STAGING_DIR", ROOT / "results")
    one_key_path = ROOT / "results" / "cert_envelope_synth.json"
    upstream = {
        contract.repo_relative(one_key_path): contract.sha256_file(one_key_path)
    }
    preflight_sha256 = "d" * 64
    preflight = {
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        }
    }
    payloads = _synthetic_hlm_payloads(preflight_sha256, upstream)

    assert compare.validate_hlm(payloads, preflight, preflight_sha256) == []

    tampered = copy.deepcopy(payloads)
    tampered["beta_star"]["per_target"][0]["worst_margin_beta_star"] = -1.0
    failures = compare.validate_hlm(tampered, preflight, preflight_sha256)
    assert any("beta_star[0]: margin is not positive" in item for item in failures)
    assert any("validity summary does not match rows" in item for item in failures)

    tampered_multi = copy.deepcopy(payloads)
    tampered_multi["multi_key"]["keys"][0][
        "all_candidate_margins_positive"
    ] = False
    failures = compare.validate_hlm(
        tampered_multi,
        preflight,
        preflight_sha256,
    )
    assert any("multi_key: a candidate dose has non-positive margin" in item for item in failures)


def _synthetic_counterfact_contract(
    preflight_sha256: str,
) -> dict[str, object]:
    producer = "scripts/run_cert_counterfact_gpt2xl.py"
    return {
        "schema": contract.CONTRACT_SCHEMA,
        "protocol_commit": contract.PROTOCOL_COMMIT,
        "protocol_sha256": contract.PROTOCOL_SHA256,
        "eps": 0.0,
        "arithmetic_dtype": "float64",
        "producer": producer,
        "producer_sha256": contract.sha256_file(ROOT / producer),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": "float32",
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "seed": 0,
        "sample_contract": contract.COUNTERFACT_SAMPLE_CONTRACT,
        "upstream_sha256": {},
        "model_id": contract.GPT2_XL_MODEL_ID,
        "model_revision": contract.GPT2_XL_REVISION,
        "data": "scripts/data/counterfact.json",
        "data_sha256": contract.EXPECTED_COUNTERFACT_SHA256,
    }


def test_counterfact_comparator_validates_all_rows_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_sha256 = "e" * 64
    base_contract = _synthetic_counterfact_contract(preflight_sha256)
    row_contract_sha256 = contract.stable_json_sha256(base_contract)
    rows = [
        {
            "certificate_contract_sha256": row_contract_sha256,
            "certificate_boundary_dtypes": _runtime_boundary(),
            "case_id": index,
            "usable_index": index,
            "reachable": True,
            "L": 1.0,
            "U": 3.0,
            "slack": 2.0,
            "risk": "safe",
            "hard_blocker": False,
            "beta_star": 2.0,
            "margin_at_beta_star": 1.0,
            "head_reachable": True,
            "already_correct": False,
            "single_token_target": True,
        }
        for index in range(1000)
    ]
    jsonl_path = tmp_path / "counterfact.jsonl"
    jsonl_path.write_text(
        "".join(f"{contract.stable_json(row)}\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(compare, "COUNTERFACT_ROWS", jsonl_path)
    summary_contract = {
        **base_contract,
        "jsonl_sha256": contract.sha256_file(jsonl_path),
    }
    summary = {
        "certificate_contract": summary_contract,
        "certificate_boundary_dtypes": _runtime_boundary(),
        "model": "gpt2-xl",
        "dtype": "float32",
        "device": "cpu",
        "n_records": 1000,
        "identity_check_first_record": {
            "case_id": 0,
            "argmax_match": True,
            "allclose": True,
            "rtol": 1e-5,
            "atol": 2e-5,
            "max_abs_diff_lmhead_vs_logits": 0.0,
            "dtype": "float32",
        },
        "risk_label_counts": {"safe": 1000},
        "fraction_reachable": 1.0,
        "fraction_hard_blocker": 0.0,
        "fraction_infinite_slack": 0.0,
        "median_slack_finite_all": 2.0,
        "median_slack_reachable_finite": 2.0,
        "fraction_head_reachable": 1.0,
        "fraction_already_correct": 0.0,
        "fraction_single_token_target": 1.0,
        "case_id_first": 0,
        "case_id_last": 999,
    }
    preflight = {
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "cuda_available": False,
        }
    }

    assert compare.validate_counterfact(
        summary,
        rows,
        preflight,
        preflight_sha256,
    ) == []

    tampered = copy.deepcopy(rows)
    tampered[0]["margin_at_beta_star"] = -1.0
    failures = compare.validate_counterfact(
        summary,
        tampered,
        preflight,
        preflight_sha256,
    )
    assert any("counterfact[0]: non-positive certified margin" in item for item in failures)

    identity_tampered = copy.deepcopy(summary)
    identity_tampered["identity_check_first_record"]["allclose"] = False
    failures = compare.validate_counterfact(
        identity_tampered,
        rows,
        preflight,
        preflight_sha256,
    )
    assert any("native identity check failed" in item for item in failures)
