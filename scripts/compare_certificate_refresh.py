"""Validate staged certificate artifacts and compare them with release anchors."""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from hlm5.artifact_contract import (
    BETA_STAR_SAMPLE_CONTRACT,
    CERTIFICATE_ARITHMETIC_DTYPE,
    CERTIFICATE_EPS,
    CHECKPOINT_PATH,
    CONTRACT_SCHEMA,
    COUNTERFACT_SAMPLE_CONTRACT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_COUNTERFACT_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    FAITHFUL_SAMPLE_CONTRACT,
    GPT2_XL_MODEL_ID,
    GPT2_XL_REVISION,
    MULTI_KEY_SAMPLE_CONTRACT,
    ONE_KEY_SAMPLE_CONTRACT,
    PROTOCOL_COMMIT,
    PROTOCOL_SHA256,
    REPO_ROOT,
    STAGING_DIR,
    SYNTHESIS_FLIP_SAMPLE_CONTRACT,
    TOKENIZER_PATH,
    atomic_write_json,
    load_contract_jsonl,
    load_json_object,
    load_preflight,
    repo_relative,
    require_staged_artifact,
    sha256_file,
    stable_json_sha256,
    verify_registered_sources,
)


COMPARISON_PATH = STAGING_DIR / "comparison.json"
MANIFEST_PATH = STAGING_DIR / "manifest.json"
HLM_SPECS = (
    (
        "one_key",
        "cert_envelope_synth.json",
        "scripts/run_1b_certificate.py",
    ),
    ("beta_star", "betastar.json", "scripts/run_1b_betastar.py"),
    ("synthesis_flip", "synth_verify.json", "scripts/run_1b_synth_verify.py"),
    (
        "multi_key",
        "cert_envelope_multikey.json",
        "scripts/run_1b_envelope_multikey.py",
    ),
    (
        "faithful",
        "hlm5_1b_faithful_certdosed.json",
        "scripts/run_1b_faithful_certdosed.py",
    ),
)
COUNTERFACT_ROWS = STAGING_DIR / "cert_counterfact_gpt2xl.jsonl"
COUNTERFACT_SUMMARY = STAGING_DIR / "cert_counterfact_gpt2xl_summary.json"
COUNTERFACT_PRODUCER = "scripts/run_cert_counterfact_gpt2xl.py"
RELEASE_ARTIFACTS = (
    "results/cert_envelope_synth.json",
    "results/betastar.json",
    "results/synth_verify.json",
    "results/cert_envelope_multikey.json",
    "results/hlm5_1b_faithful_certdosed.json",
    "results/cert_counterfact_gpt2xl.jsonl",
    "results/cert_counterfact_gpt2xl_summary.json",
)
HLM_SAMPLE_CONTRACTS = {
    "one_key": ONE_KEY_SAMPLE_CONTRACT,
    "beta_star": BETA_STAR_SAMPLE_CONTRACT,
    "synthesis_flip": SYNTHESIS_FLIP_SAMPLE_CONTRACT,
    "multi_key": MULTI_KEY_SAMPLE_CONTRACT,
    "faithful": FAITHFUL_SAMPLE_CONTRACT,
}
EPS_PATTERN = re.compile(r"^\s*EPS\s*=\s*([0-9.eE+-]+)", re.MULTILINE)
DTYPE_PATTERN = re.compile(
    r"^\s*CERTIFICATE_ARITHMETIC_DTYPE\s*=\s*[\"']([^\"']+)[\"']",
    re.MULTILINE,
)


def load_json(path: Path) -> dict[str, Any]:
    return load_json_object(path)


def check_contract(
    payload: dict[str, Any],
    producer: str,
    sample_contract: dict[str, Any],
    preflight: dict[str, Any],
    preflight_sha256: str,
    lane: str,
    upstream_sha256: dict[str, str] | None = None,
) -> list[str]:
    contract = payload.get("certificate_contract")
    if not isinstance(contract, dict):
        return [f"{producer}: missing certificate_contract"]
    expected = {
        "schema": CONTRACT_SCHEMA,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "eps": CERTIFICATE_EPS,
        "arithmetic_dtype": CERTIFICATE_ARITHMETIC_DTYPE,
        "producer": producer,
        "producer_sha256": sha256_file(REPO_ROOT / producer),
        "preflight_sha256": preflight_sha256,
        "model_forward_dtype": "float32",
        "torch_version": preflight["environment"]["torch"],
        "python_version": preflight["environment"]["python"],
        "seed": 0,
        "sample_contract": sample_contract,
        "upstream_sha256": dict(sorted((upstream_sha256 or {}).items())),
    }
    if lane == "hlm5":
        expected.update(
            {
                "checkpoint": repo_relative(CHECKPOINT_PATH),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "tokenizer": repo_relative(TOKENIZER_PATH),
                "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            }
        )
    else:
        expected.update(
            {
                "model_id": GPT2_XL_MODEL_ID,
                "model_revision": GPT2_XL_REVISION,
                "data": "scripts/data/counterfact.json",
                "data_sha256": EXPECTED_COUNTERFACT_SHA256,
            }
        )
    return [
        f"{producer}: {key}={contract.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if contract.get(key) != value
    ]


def check_boundary(payload: dict[str, Any], label: str) -> list[str]:
    boundary = payload.get("certificate_boundary_dtypes")
    if not isinstance(boundary, dict) or not boundary:
        return [f"{label}: missing runtime certificate boundary dtypes"]
    failures = [
        f"{label}: certificate tensor {name} has dtype {dtype!r}"
        for name, dtype in boundary.items()
        if dtype != "float64"
    ]
    for required in ("head", "hidden", "direction"):
        if required not in boundary and not (
            required == "direction" and "direction_matrix" in boundary
        ):
            failures.append(f"{label}: boundary does not report {required} dtype")
    return failures


def check_producer_source(producer: str) -> list[str]:
    source = (REPO_ROOT / producer).read_text(encoding="utf-8")
    eps_values = [float(value) for value in EPS_PATTERN.findall(source)]
    dtype_match = DTYPE_PATTERN.search(source)
    failures = []
    if not eps_values or any(value != 0.0 for value in eps_values):
        failures.append(f"{producer}: source does not use exact EPS=0 signs")
    if dtype_match is None or dtype_match.group(1) != "float64":
        failures.append(f"{producer}: source has no float64 arithmetic marker")
    for line_number, line in enumerate(source.splitlines(), start=1):
        if ".float()" in line and ("W =" in line or "h =" in line):
            failures.append(
                f"{producer}:{line_number}: float32 certificate boundary"
            )
    return failures


def inside_open_interval(lower: float, upper: float | None, value: float) -> bool:
    return lower < value and (upper is None or value < upper)


def upper_median(values: list[float]) -> float | None:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else None


def rounded_population_stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    std = (sum((value - mean) ** 2 for value in values) / n) ** 0.5
    ordered = sorted(values)
    median = (
        ordered[n // 2]
        if n % 2
        else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    )
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "median": round(median, 4),
        "max": round(max(values), 4),
    }


def bootstrap_summary(
    values: list[float],
    rng: random.Random,
    resamples: int = 2000,
) -> list[float]:
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    return [
        round(sum(values) / n, 3),
        round(means[int(0.025 * resamples)], 3),
        round(means[int(0.975 * resamples)], 3),
    ]


def validate_hlm(
    payloads: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
    preflight_sha256: str,
) -> list[str]:
    failures: list[str] = []
    producers = {name: producer for name, _, producer in HLM_SPECS}
    for name, payload in payloads.items():
        upstream = None
        if name == "multi_key":
            one_key_path = STAGING_DIR / "cert_envelope_synth.json"
            upstream = {repo_relative(one_key_path): sha256_file(one_key_path)}
        failures.extend(
            check_contract(
                payload,
                producers[name],
                HLM_SAMPLE_CONTRACTS[name],
                preflight,
                preflight_sha256,
                "hlm5",
                upstream,
            )
        )
        failures.extend(check_boundary(payload, name))
        failures.extend(check_producer_source(producers[name]))

    one = payloads["one_key"]
    if one.get("pool") != 1200:
        failures.append("one_key: pool must be 1200")
    one_envelope = one.get("envelope", {})
    one_rows = one_envelope.get("per_target", [])
    if len(one_rows) != 1200 or len(
        {row.get("target_id") for row in one_rows}
    ) != 1200:
        failures.append("one_key: expected 1200 unique per-target rows")
    derived_risk = {
        label: sum(row.get("risk") == label for row in one_rows)
        for label in ("safe", "narrow", "brittle", "unreachable")
    }
    if one_envelope.get("risk_label_counts") != derived_risk:
        failures.append("one_key: risk counts do not match per-target rows")
    if sum(derived_risk.values()) != 1200:
        failures.append("one_key: risk counts do not sum to 1200")
    reachable_rows = [row for row in one_rows if row.get("reachable")]
    if one_envelope.get("reachable_rate") != round(len(reachable_rows) / 1200, 3):
        failures.append("one_key: reachable rate does not match per-target rows")
    if one_envelope.get("unreachable_rate") != round(
        (1200 - len(reachable_rows)) / 1200,
        3,
    ):
        failures.append("one_key: unreachable rate does not match per-target rows")
    finite_slacks = [
        row["slack"]
        for row in reachable_rows
        if row.get("slack") is not None
    ]
    expected_median_slack = (
        round(upper_median(finite_slacks), 2) if finite_slacks else None
    )
    if one_envelope.get("median_slack_reachable") != expected_median_slack:
        failures.append("one_key: median slack does not match per-target rows")
    for index, row in enumerate(reachable_rows):
        if not inside_open_interval(
            row.get("L"),
            row.get("U"),
            row.get("beta_candidate"),
        ):
            failures.append(f"one_key[{index}]: candidate outside open interval")
        if not row.get("margin_at_candidate", 0.0) > 0.0:
            failures.append(f"one_key[{index}]: candidate margin is not positive")
    synthesis_block = one.get("residual_synthesis", {})
    synthesis_rows = synthesis_block.get("per_target", [])
    if synthesis_block.get("unreachable_tested") != 40:
        failures.append("one_key: synthesis sample must contain 40 targets")
    if len(synthesis_rows) != 40:
        failures.append("one_key: expected 40 synthesis rows")
    expected_synthesis_targets = [
        row.get("target_id")
        for row in one_rows
        if not row.get("reachable")
    ][:40]
    if [row.get("target_id") for row in synthesis_rows] != expected_synthesis_targets:
        failures.append("one_key: synthesis rows are not the first 40 unreachable")
    rescued = sum(bool(row.get("rescued")) for row in synthesis_rows)
    if synthesis_block.get("rescued") != rescued:
        failures.append("one_key: synthesis rescue count mismatch")
    if synthesis_block.get("rescue_rate") != round(rescued / 40, 3):
        failures.append("one_key: synthesis rescue rate mismatch")
    one_validity = one.get("validity", {})
    if one_validity.get("candidate_doses_checked") != sum(
        one.get("envelope", {}).get("risk_label_counts", {}).get(label, 0)
        for label in ("safe", "narrow", "brittle")
    ):
        failures.append("one_key: candidate validation count mismatch")
    if one_validity.get("all_candidate_doses_inside_open_interval") is not True:
        failures.append("one_key: a candidate dose is outside its open interval")
    if one_validity.get("all_candidate_margins_positive") is not True:
        failures.append("one_key: a candidate dose has non-positive margin")
    if one_validity.get("synthesis_directions_checked") != 40:
        failures.append("one_key: did not validate all 40 synthesis directions")
    expected_one_validity = {
        "candidate_doses_checked": len(reachable_rows),
        "all_candidate_doses_inside_open_interval": all(
            inside_open_interval(
                row["L"],
                row["U"],
                row["beta_candidate"],
            )
            for row in reachable_rows
        ),
        "all_candidate_margins_positive": all(
            row["margin_at_candidate"] > 0.0 for row in reachable_rows
        ),
        "minimum_candidate_margin": min(
            row["margin_at_candidate"] for row in reachable_rows
        ),
        "synthesis_directions_checked": len(synthesis_rows),
        "all_synthesis_margins_positive": all(
            row["minimum_margin"] > 0.0 for row in synthesis_rows
        ),
        "minimum_synthesis_margin": min(
            row["minimum_margin"] for row in synthesis_rows
        ),
    }
    if one_validity != expected_one_validity:
        failures.append("one_key: validity summary does not match per-target rows")

    synth = payloads["synthesis_flip"]
    if synth.get("n_unreachable_tested") != 40:
        failures.append("synthesis_flip: sample must contain 40 targets")
    if synth.get("validity", {}).get("synthesis_directions_checked") != 40:
        failures.append("synthesis_flip: did not validate all 40 directions")
    synth_rows = synth.get("per_target", [])
    if len(synth_rows) != 40:
        failures.append("synthesis_flip: expected 40 per-target rows")
    if [row.get("target_id") for row in synth_rows] != expected_synthesis_targets:
        failures.append("synthesis_flip: target sample differs from one-key sample")
    unit_flips = sum(bool(row.get("unit_value_flipped")) for row in synth_rows)
    synth_flips = sum(bool(row.get("synth_value_flipped")) for row in synth_rows)
    used_betas = [
        row["first_flip_beta"]
        for row in synth_rows
        if row.get("first_flip_beta") is not None
    ]
    synth_expected = {
        "flip_with_unit_Wy": unit_flips,
        "flip_with_synth_residual": synth_flips,
        "rescue_flip_rate": round(synth_flips / 40, 3),
        "median_beta_needed": upper_median(used_betas),
        "max_beta_needed": max(used_betas) if used_betas else None,
        "min_beta_needed": min(used_betas) if used_betas else None,
    }
    for key, expected in synth_expected.items():
        if synth.get(key) != expected:
            failures.append(
                f"synthesis_flip: {key}={synth.get(key)!r}, expected {expected!r}"
            )
    expected_synth_validity = {
        "synthesis_directions_checked": len(synth_rows),
        "all_synthesis_margins_positive": all(
            row["synthesis_min_margin"] > 0.0 for row in synth_rows
        ),
        "minimum_synthesis_margin": min(
            row["synthesis_min_margin"] for row in synth_rows
        ),
    }
    if synth.get("validity") != expected_synth_validity:
        failures.append("synthesis_flip: validity summary does not match rows")

    beta = payloads["beta_star"]
    beta_validity = beta.get("validity", {})
    beta_rows = beta.get("per_target", [])
    if len(beta_rows) != beta.get("n_reachable") or len(
        {row.get("target_id") for row in beta_rows}
    ) != len(beta_rows):
        failures.append("beta_star: per-target rows are missing or duplicated")
    for index, row in enumerate(beta_rows):
        if not inside_open_interval(row["L"], row["U"], row["beta_star"]):
            failures.append(f"beta_star[{index}]: dose outside open interval")
        if not row.get("worst_margin_beta_star", 0.0) > 0.0:
            failures.append(f"beta_star[{index}]: margin is not positive")
    beta_expected = {
        "median_beta_star": (
            round(upper_median([row["beta_star"] for row in beta_rows]), 3)
            if beta_rows
            else None
        ),
        "median_worst_margin_betastar": (
            round(
                upper_median(
                    [row["worst_margin_beta_star"] for row in beta_rows]
                ),
                3,
            )
            if beta_rows
            else None
        ),
        "median_worst_margin_heuristic": (
            round(
                upper_median(
                    [row["worst_margin_heuristic"] for row in beta_rows]
                ),
                3,
            )
            if beta_rows
            else None
        ),
    }
    for key, expected in beta_expected.items():
        if beta.get(key) != expected:
            failures.append(
                f"beta_star: {key}={beta.get(key)!r}, expected {expected!r}"
            )
    expected_gain = round(
        (beta_expected["median_worst_margin_betastar"] or 0.0)
        - (beta_expected["median_worst_margin_heuristic"] or 0.0),
        3,
    )
    if beta.get("median_margin_gain") != expected_gain:
        failures.append(
            "beta_star: median margin gain does not match per-target rows"
        )
    expected_beta_validity = {
        "candidate_doses_checked": len(beta_rows),
        "envelope_reachable": len(beta_rows),
        "grid_failures": 0,
        "all_candidate_doses_inside_open_interval": all(
            inside_open_interval(row["L"], row["U"], row["beta_star"])
            for row in beta_rows
        ),
        "all_candidate_margins_positive": all(
            row["worst_margin_beta_star"] > 0.0 for row in beta_rows
        ),
        "minimum_candidate_margin": min(
            row["worst_margin_beta_star"] for row in beta_rows
        ),
    }
    if beta_validity != expected_beta_validity:
        failures.append("beta_star: validity summary does not match rows")
    if beta_validity.get("candidate_doses_checked") != beta.get("n_reachable"):
        failures.append("beta_star: candidate validation count mismatch")
    if beta_validity.get("envelope_reachable") != beta.get("n_reachable"):
        failures.append("beta_star: a reachable envelope has no certified dose")
    if beta_validity.get("grid_failures") != 0:
        failures.append("beta_star: dose grid failed on a reachable envelope")
    if beta_validity.get("all_candidate_doses_inside_open_interval") is not True:
        failures.append("beta_star: a dose is outside its open interval")
    if beta_validity.get("all_candidate_margins_positive") is not True:
        failures.append("beta_star: a dose has non-positive margin")

    multi = payloads["multi_key"]
    if multi.get("pool") != 1200 or multi.get("n_keys") != 60:
        failures.append("multi_key: expected 1200 targets across 60 keys")
    original = multi.get("original_key", {})
    if original.get("recompute_matches_stored") is not True:
        failures.append("multi_key: scalar result does not match staged one-key result")
    if original.get("vectorized_matches_scalar") is not True:
        failures.append("multi_key: vectorized result does not match scalar result")
    expected_stored = {
        "reachable_rate": one_envelope.get("reachable_rate"),
        "median_slack_reachable": one_envelope.get("median_slack_reachable"),
        "risk_label_counts": one_envelope.get("risk_label_counts"),
    }
    if original.get("stored") != expected_stored:
        failures.append("multi_key: embedded one-key summary does not match upstream")
    keys = multi.get("keys", [])
    if len(keys) != 60:
        failures.append("multi_key: expected exactly 60 key rows")
    categories = {
        category: sum(row.get("category") == category for row in keys)
        for category in ("novel", "real", "generic")
    }
    if categories != {"novel": 20, "real": 20, "generic": 20}:
        failures.append(f"multi_key: key category counts mismatch: {categories}")
    if len({row.get("prompt") for row in keys}) != 60:
        failures.append("multi_key: prompts are not unique")
    original_rows = [row for row in keys if row.get("is_original_key")]
    if len(original_rows) != 1:
        failures.append("multi_key: expected exactly one original-key row")
    else:
        original_row = original_rows[0]
        if original_row.get("prompt") != MULTI_KEY_SAMPLE_CONTRACT["original_key"]:
            failures.append("multi_key: original-key prompt mismatch")
        expected_recomputed = {
            "reachable_fraction": original_row.get("reachable_fraction"),
            "risk_counts": original_row.get("risk_counts"),
            "median_slack_reachable": original_row.get(
                "median_slack_reachable"
            ),
            "unreachable_total": original_row.get("risk_counts", {}).get(
                "unreachable"
            ),
            "unreachable_hard_blocker": original_row.get(
                "unreachable_hard_blocker"
            ),
            "unreachable_interval_only": original_row.get(
                "unreachable_interval_only"
            ),
        }
        if original.get("recomputed_scalar") != expected_recomputed:
            failures.append(
                "multi_key: scalar original summary does not match original-key row"
            )
    prompt_list = multi.get("prompt_list", {})
    for category in ("novel", "real", "generic"):
        if prompt_list.get(category) != [
            row.get("prompt") for row in keys if row.get("category") == category
        ]:
            failures.append(f"multi_key: prompt list mismatch for {category}")
    if any(sum(row.get("risk_counts", {}).values()) != 1200 for row in keys):
        failures.append("multi_key: at least one risk-count row does not sum to 1200")
    multi_aggregates = multi.get("aggregates", {})
    aggregate_inputs = {
        "reachable_fraction": [row["reachable_fraction"] for row in keys],
        "median_slack_reachable": [
            row["median_slack_reachable"] for row in keys
        ],
        "unreachable_hard_blocker": [
            row["unreachable_hard_blocker"] for row in keys
        ],
        "unreachable_interval_only": [
            row["unreachable_interval_only"] for row in keys
        ],
    }
    for key, values in aggregate_inputs.items():
        if keys and multi_aggregates.get(key) != rounded_population_stats(values):
            failures.append(f"multi_key: aggregate {key} does not match key rows")
    expected_per_category = {}
    for category in ("novel", "real", "generic"):
        subset = [row for row in keys if row.get("category") == category]
        if not subset:
            continue
        expected_per_category[category] = {
            "n": len(subset),
            "mean_reachable_fraction": round(
                sum(row["reachable_fraction"] for row in subset) / len(subset),
                4,
            ),
            "mean_median_slack": round(
                sum(row["median_slack_reachable"] for row in subset)
                / len(subset),
                2,
            ),
            "mean_unreachable": round(
                sum(row["risk_counts"]["unreachable"] for row in subset)
                / len(subset),
                1,
            ),
            "mean_hard_blocker": round(
                sum(row["unreachable_hard_blocker"] for row in subset)
                / len(subset),
                1,
            ),
            "mean_interval_only": round(
                sum(row["unreachable_interval_only"] for row in subset)
                / len(subset),
                1,
            ),
        }
    if multi_aggregates.get("per_category") != expected_per_category:
        failures.append("multi_key: per-category aggregates do not match key rows")
    if keys:
        reachable_fractions = aggregate_inputs["reachable_fraction"]
        reachable_stats = rounded_population_stats(reachable_fractions)
        original_fraction = original.get("recomputed_scalar", {}).get(
            "reachable_fraction"
        )
        std = reachable_stats["std"]
        expected_typicality = {
            "original_reachable_fraction": original_fraction,
            "percentile_rank": round(
                sum(value <= original_fraction for value in reachable_fractions)
                / len(reachable_fractions),
                3,
            ),
            "z_score": (
                round((original_fraction - reachable_stats["mean"]) / std, 2)
                if std > 0
                else None
            ),
            "within_1_std": bool(
                abs(original_fraction - reachable_stats["mean"]) <= std
            ),
        }
        if multi_aggregates.get("original_key_typicality") != expected_typicality:
            failures.append("multi_key: original-key typicality does not match rows")
    expected_multi_validity = {
        "keys_checked": len(keys),
        "candidate_doses_checked": sum(
            row.get("candidate_doses_checked", 0) for row in keys
        ),
        "all_candidate_doses_inside_open_interval": all(
            row.get("all_candidate_doses_inside_open_interval") is True
            for row in keys
        ),
        "all_candidate_margins_positive": all(
            row.get("all_candidate_margins_positive") is True for row in keys
        ),
        "minimum_candidate_margin": min(
            row["minimum_candidate_margin"]
            for row in keys
            if row.get("minimum_candidate_margin") is not None
        ),
        "all_risk_counts_sum_to_pool": all(
            sum(row.get("risk_counts", {}).values()) == 1200 for row in keys
        ),
        "original_scalar_matches_staged_one_key": (
            original.get("recompute_matches_stored") is True
        ),
        "original_vectorized_matches_scalar": (
            original.get("vectorized_matches_scalar") is True
        ),
    }
    if multi.get("validity") != expected_multi_validity:
        failures.append("multi_key: validity summary does not match key rows")
    if expected_multi_validity[
        "all_candidate_doses_inside_open_interval"
    ] is not True:
        failures.append("multi_key: a candidate dose is outside its open interval")
    if expected_multi_validity["all_candidate_margins_positive"] is not True:
        failures.append("multi_key: a candidate dose has non-positive margin")

    faithful = payloads["faithful"]
    rows = faithful.get("per_fact", [])
    if faithful.get("n_facts") != 17 or len(rows) != 17:
        failures.append("faithful: expected exactly 17 fact rows")
    for index, row in enumerate(rows):
        if row.get("reachable"):
            if not inside_open_interval(
                row["L"], row["U"], row["beta_star"]
            ):
                failures.append(f"faithful[{index}]: naive dose outside open interval")
            if not row.get("margin_after", 0.0) > 0.0:
                failures.append(f"faithful[{index}]: naive margin is not positive")
        if row.get("rescued"):
            if not inside_open_interval(
                row["L_synth"], row["U_synth"], row["beta_star_synth"]
            ):
                failures.append(f"faithful[{index}]: synth dose outside open interval")
            if not row.get("margin_after_synth", 0.0) > 0.0:
                failures.append(f"faithful[{index}]: synth margin is not positive")
        if bool(row.get("flip_cert_naive")) != bool(row.get("reachable")):
            failures.append(
                f"faithful[{index}]: ordinary naive-arm flip disagrees with certificate"
            )
        expected_synth_flip = bool(row.get("reachable") or row.get("rescued"))
        if bool(row.get("flip_cert_synth")) != expected_synth_flip:
            failures.append(
                f"faithful[{index}]: ordinary synth-arm flip disagrees with certificate"
            )
    arms = faithful.get("arms", {})
    expected_arm_names = {"global_repro", "cert_naive", "cert_synth"}
    if set(arms) != expected_arm_names:
        failures.append(f"faithful: arm set mismatch: {sorted(arms)}")
    if [row.get("i") for row in rows] != list(range(17)):
        failures.append("faithful: fact indices are not exactly 0..16")
    for arm_name, arm in arms.items():
        if arm.get("locality_bit_identical") != "8/8":
            failures.append(f"faithful {arm_name}: locality is not bit-identical 8/8")
        locality_rows = arm.get("locality_rows", [])
        if len(locality_rows) != 8 or any(
            row.get("logits_bit_identical") is not True for row in locality_rows
        ):
            failures.append(
                f"faithful {arm_name}: locality rows are not all bit-identical"
            )
    rng = random.Random(0)
    arm_raw_fields = {
        "global_repro": (
            "flip_global",
            "generalization_global",
        ),
        "cert_naive": (
            "flip_cert_naive",
            "generalization_cert_naive",
        ),
        "cert_synth": (
            "flip_cert_synth",
            "generalization_cert_synth",
        ),
    }
    for arm_name in ("global_repro", "cert_naive", "cert_synth"):
        arm = arms.get(arm_name, {})
        flip_field, generalization_field = arm_raw_fields[arm_name]
        efficacy = [float(bool(row.get(flip_field))) for row in rows]
        generalization = [
            float(row.get(generalization_field, 0.0)) for row in rows
        ]
        locality_rows = arm.get("locality_rows", [])
        locality_fraction = (
            sum(bool(row.get("argmax_match")) for row in locality_rows)
            / len(locality_rows)
            if locality_rows
            else 0.0
        )
        locality = [locality_fraction] * len(rows)
        expected_summary = {
            "eff": bootstrap_summary(efficacy, rng),
            "gen": bootstrap_summary(generalization, rng),
            "loc": bootstrap_summary(locality, rng),
        }
        if arm.get("summary") != expected_summary:
            failures.append(f"faithful {arm_name}: summary does not match raw rows")
        if arm.get("eff_count") != f"{sum(efficacy):.0f}/17":
            failures.append(f"faithful {arm_name}: efficacy count mismatch")
        expected_gen_rate = round(sum(generalization) / len(generalization), 3)
        if arm.get("gen_paraphrase_rate") != expected_gen_rate:
            failures.append(f"faithful {arm_name}: generalization rate mismatch")
        expected_argmax = sum(
            bool(row.get("argmax_match")) for row in locality_rows
        )
        expected_bit_identity = sum(
            bool(row.get("logits_bit_identical")) for row in locality_rows
        )
        if arm.get("locality_argmax") != f"{expected_argmax}/8":
            failures.append(f"faithful {arm_name}: locality argmax count mismatch")
        if arm.get("locality_bit_identical") != f"{expected_bit_identity}/8":
            failures.append(f"faithful {arm_name}: locality bit count mismatch")
        if arm.get("paraphrase_gate_fire_rate") != faithful.get(
            "paraphrase_gate_fire_rate"
        ):
            failures.append(f"faithful {arm_name}: gate-fire rate mismatch")
    faithful_validity = faithful.get("validity", {})
    for key in (
        "all_candidate_doses_inside_with_positive_margin",
        "ordinary_forward_naive_agreement",
        "ordinary_forward_synth_agreement",
        "all_arms_locality_bit_identical",
    ):
        if faithful_validity.get(key) is not True:
            failures.append(f"faithful: failed reported validity bar {key}")
    expected_faithful_validity = {
        "candidate_doses_checked": sum(
            bool(row.get("reachable")) + bool(row.get("rescued"))
            for row in rows
        ),
        "all_candidate_doses_inside_with_positive_margin": all(
            (
                not row.get("reachable")
                or (
                    inside_open_interval(
                        row["L"],
                        row["U"],
                        row["beta_star"],
                    )
                    and row["margin_after"] > 0.0
                )
            )
            and (
                not row.get("rescued")
                or (
                    inside_open_interval(
                        row["L_synth"],
                        row["U_synth"],
                        row["beta_star_synth"],
                    )
                    and row["margin_after_synth"] > 0.0
                )
            )
            for row in rows
        ),
        "ordinary_forward_naive_agreement": all(
            bool(row.get("flip_cert_naive")) == bool(row.get("reachable"))
            for row in rows
        ),
        "ordinary_forward_synth_agreement": all(
            bool(row.get("flip_cert_synth"))
            == bool(row.get("reachable") or row.get("rescued"))
            for row in rows
        ),
        "all_arms_locality_bit_identical": all(
            arm.get("locality_bit_identical") == "8/8"
            for arm in arms.values()
        )
        and set(arms) == expected_arm_names,
    }
    if faithful_validity != expected_faithful_validity:
        failures.append("faithful: validity summary does not match raw rows")
    return failures


def validate_counterfact(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    preflight: dict[str, Any],
    preflight_sha256: str,
) -> list[str]:
    failures = check_contract(
        summary,
        COUNTERFACT_PRODUCER,
        COUNTERFACT_SAMPLE_CONTRACT,
        preflight,
        preflight_sha256,
        "counterfact",
    )
    failures.extend(check_producer_source(COUNTERFACT_PRODUCER))
    failures.extend(check_boundary(summary, "counterfact summary"))
    contract = summary.get("certificate_contract", {})
    base_contract = {key: value for key, value in contract.items() if key != "jsonl_sha256"}
    contract_sha256 = stable_json_sha256(base_contract)
    if len(rows) != 1000:
        failures.append(f"counterfact: expected 1000 rows, found {len(rows)}")
    if not rows:
        return failures
    if [row.get("usable_index") for row in rows] != list(range(1000)):
        failures.append("counterfact: usable indices are not exactly 0..999")
    if len({row.get("case_id") for row in rows}) != len(rows):
        failures.append("counterfact: duplicate case IDs")
    if any(
        row.get("certificate_contract_sha256") != contract_sha256 for row in rows
    ):
        failures.append("counterfact: row contract hash mismatch")
    if contract.get("jsonl_sha256") != sha256_file(COUNTERFACT_ROWS):
        failures.append("counterfact: summary does not bind current JSONL")
    identity = summary.get("identity_check_first_record", {})
    if (
        not isinstance(identity, dict)
        or identity.get("argmax_match") is not True
        or identity.get("allclose") is not True
        or identity.get("rtol") != 1e-5
        or identity.get("atol") != 2e-5
        or identity.get("dtype") != "float32"
    ):
        failures.append("counterfact: first-record native identity check failed")
    elif identity.get("case_id") != rows[0].get("case_id"):
        failures.append("counterfact: identity check is not bound to the first row")
    for index, row in enumerate(rows):
        failures.extend(check_boundary(row, f"counterfact[{index}]"))
        if row.get("reachable"):
            beta = row.get("beta_star")
            margin = row.get("margin_at_beta_star")
            lower = row.get("L")
            upper = row.get("U")
            if not all(isinstance(value, (int, float)) for value in (beta, margin, lower)):
                failures.append(f"counterfact[{index}]: missing certified dose fields")
            elif not inside_open_interval(lower, upper, beta):
                failures.append(f"counterfact[{index}]: dose outside open interval")
            elif not margin > 0.0:
                failures.append(f"counterfact[{index}]: non-positive certified margin")
    if summary.get("n_records") != len(rows):
        failures.append("counterfact: summary row count mismatch")
    risk_labels = [row.get("risk") for row in rows]
    expected_risk_counts = {
        label: risk_labels.count(label)
        for label in ("safe", "narrow", "brittle", "unreachable")
        if risk_labels.count(label)
    }
    if summary.get("risk_label_counts") != expected_risk_counts:
        failures.append("counterfact: summary risk counts do not match rows")
    finite_slacks = sorted(
        row["slack"] for row in rows if row.get("slack") is not None
    )
    reachable_finite_slacks = sorted(
        row["slack"]
        for row in rows
        if row.get("reachable") and row.get("slack") is not None
    )
    aggregate_checks = {
        "fraction_reachable": round(
            sum(bool(row.get("reachable")) for row in rows) / len(rows),
            4,
        ),
        "fraction_hard_blocker": round(
            sum(bool(row.get("hard_blocker")) for row in rows) / len(rows),
            4,
        ),
        "fraction_head_reachable": round(
            sum(bool(row.get("head_reachable")) for row in rows) / len(rows),
            4,
        ),
        "fraction_infinite_slack": round(
            sum(row.get("slack") is None for row in rows) / len(rows),
            4,
        ),
        "median_slack_finite_all": (
            round(upper_median(finite_slacks), 4) if finite_slacks else None
        ),
        "median_slack_reachable_finite": (
            round(upper_median(reachable_finite_slacks), 4)
            if reachable_finite_slacks
            else None
        ),
        "fraction_already_correct": round(
            sum(bool(row.get("already_correct")) for row in rows) / len(rows),
            4,
        ),
        "fraction_single_token_target": round(
            sum(bool(row.get("single_token_target")) for row in rows)
            / len(rows),
            4,
        ),
        "case_id_first": rows[0].get("case_id"),
        "case_id_last": rows[-1].get("case_id"),
    }
    for key, expected in aggregate_checks.items():
        if summary.get(key) != expected:
            failures.append(
                f"counterfact: {key}={summary.get(key)!r}, expected {expected!r}"
            )
    if summary.get("certificate_boundary_dtypes") != rows[0].get(
        "certificate_boundary_dtypes"
    ):
        failures.append("counterfact: summary boundary does not match first row")
    expected_device = (
        "cuda" if preflight["environment"].get("cuda_available") else "cpu"
    )
    if summary.get("model") != GPT2_XL_MODEL_ID:
        failures.append("counterfact: model id mismatch")
    if summary.get("dtype") != "float32":
        failures.append("counterfact: model-forward dtype mismatch")
    if summary.get("device") != expected_device:
        failures.append("counterfact: device does not match preflight")
    return failures


def compare_categorical_rows(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    key: str,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    old_by_key = {row.get(key): row for row in old_rows}
    new_by_key = {row.get(key): row for row in new_rows}
    changes = []
    for row_key in sorted(set(old_by_key) | set(new_by_key), key=str):
        old = old_by_key.get(row_key)
        new = new_by_key.get(row_key)
        field_changes = {
            field: {
                "old": old.get(field) if old is not None else None,
                "new": new.get(field) if new is not None else None,
            }
            for field in fields
            if old is None or new is None or old.get(field) != new.get(field)
        }
        if field_changes:
            changes.append({key: row_key, "changes": field_changes})
    return {
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "changed_rows": len(changes),
        "changes": changes,
    }


def row_level_changes(
    payloads: dict[str, dict[str, Any]],
    counterfact_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    release_results = REPO_ROOT / "results"
    old_one = load_json(release_results / "cert_envelope_synth.json")
    old_multi = load_json(release_results / "cert_envelope_multikey.json")
    old_faithful = load_json(
        release_results / "hlm5_1b_faithful_certdosed.json"
    )
    changes = {
        "one_key_synthesis_examples": compare_categorical_rows(
            old_one.get("residual_synthesis", {}).get("examples", []),
            payloads["one_key"].get("residual_synthesis", {}).get("examples", []),
            key="token",
            fields=("blocker", "rescued"),
        ),
        "multi_key": compare_categorical_rows(
            old_multi.get("keys", []),
            payloads["multi_key"].get("keys", []),
            key="prompt",
            fields=("category", "base_argmax", "risk_counts"),
        ),
        "faithful": compare_categorical_rows(
            old_faithful.get("per_fact", []),
            payloads["faithful"].get("per_fact", []),
            key="i",
            fields=(
                "risk",
                "reachable",
                "rescued",
                "flip_global",
                "flip_cert_naive",
                "flip_cert_synth",
                "pred_global",
                "generalization_global",
                "pred_cert_naive",
                "generalization_cert_naive",
                "pred_cert_synth",
                "generalization_cert_synth",
            ),
        ),
    }
    if counterfact_rows is not None:
        old_rows = [
            json.loads(line)
            for line in (
                release_results / "cert_counterfact_gpt2xl.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        changes["counterfact"] = compare_categorical_rows(
            old_rows,
            counterfact_rows,
            key="usable_index",
            fields=(
                "risk",
                "reachable",
                "hard_blocker",
                "head_reachable",
                "base_argmax_id",
            ),
        )
    return changes


def headline_values(
    payloads: dict[str, dict[str, Any]],
    counterfact: dict[str, Any] | None,
) -> dict[str, Any]:
    one = payloads["one_key"]
    multi = payloads["multi_key"]
    beta = payloads["beta_star"]
    synth = payloads["synthesis_flip"]
    faithful = payloads["faithful"]
    values = {
        "one_key_reachable_rate": one["envelope"]["reachable_rate"],
        "one_key_risk_counts": one["envelope"]["risk_label_counts"],
        "multi_key_mean_reachable": multi["aggregates"]["reachable_fraction"]["mean"],
        "multi_key_population_std": multi["aggregates"]["reachable_fraction"]["std"],
        "beta_star_n_reachable": beta["n_reachable"],
        "beta_star_median": beta["median_beta_star"],
        "beta_star_median_worst_margin": beta["median_worst_margin_betastar"],
        "synthesis_flips": synth["flip_with_synth_residual"],
        "synthesis_median_beta": synth["median_beta_needed"],
        "faithful_global": faithful["arms"]["global_repro"]["eff_count"],
        "faithful_naive": faithful["arms"]["cert_naive"]["eff_count"],
        "faithful_synth": faithful["arms"]["cert_synth"]["eff_count"],
        "faithful_locality": faithful["arms"]["cert_synth"][
            "locality_bit_identical"
        ],
    }
    if counterfact is not None:
        values.update(
            {
                "counterfact_reachable": counterfact["fraction_reachable"],
                "counterfact_hard_blocker": counterfact["fraction_hard_blocker"],
                "counterfact_head_reachable": counterfact[
                    "fraction_head_reachable"
                ],
            }
        )
    return values


def load_release_headlines() -> dict[str, Any]:
    payloads = {
        name: load_json(REPO_ROOT / "results" / filename)
        for name, filename, _ in HLM_SPECS
    }
    counterfact = load_json(
        REPO_ROOT / "results" / "cert_counterfact_gpt2xl_summary.json"
    )
    return headline_values(payloads, counterfact)


OLD_HEADLINES = {
    "one_key_reachable_rate": 0.784,
    "one_key_risk_counts": {
        "safe": 888,
        "narrow": 38,
        "brittle": 15,
        "unreachable": 259,
    },
    "multi_key_mean_reachable": 0.7938,
    "multi_key_population_std": 0.0221,
    "beta_star_n_reachable": 941,
    "beta_star_median": 72.134,
    "beta_star_median_worst_margin": 18.39,
    "synthesis_flips": 40,
    "synthesis_median_beta": 20,
    "faithful_global": "9/17",
    "faithful_naive": "13/17",
    "faithful_synth": "17/17",
    "faithful_locality": "8/8",
    "counterfact_reachable": 1.0,
    "counterfact_hard_blocker": 0.0,
    "counterfact_head_reachable": 1.0,
}


def main() -> None:
    preflight, preflight_sha256 = load_preflight()
    verify_registered_sources(preflight)
    missing_hlm = [
        repo_relative(STAGING_DIR / filename)
        for _, filename, _ in HLM_SPECS
        if not (STAGING_DIR / filename).is_file()
    ]
    if missing_hlm:
        present_hashes = {
            repo_relative(STAGING_DIR / filename): sha256_file(
                STAGING_DIR / filename
            )
            for _, filename, _ in HLM_SPECS
            if (STAGING_DIR / filename).is_file()
        }
        report = {
            "schema": "hlm5-certificate-refresh-comparison-v1",
            "protocol_commit": PROTOCOL_COMMIT,
            "preflight_sha256": preflight_sha256,
            "preflight_status": preflight["status"],
            "verdict": "INCOMPLETE",
            "hlm5_status": "INCOMPLETE",
            "counterfact_complete": False,
            "missing_artifacts": missing_hlm,
            "validity_failures": [],
            "artifact_sha256": dict(sorted(present_hashes.items())),
        }
        atomic_write_json(COMPARISON_PATH, report)
        comparison_sha256 = sha256_file(COMPARISON_PATH)
        manifest = {
            "schema": "hlm5-certificate-refresh-manifest-v1",
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_sha256": PROTOCOL_SHA256,
            "preflight_sha256": preflight_sha256,
            "comparison_sha256": comparison_sha256,
            "artifact_sha256": dict(sorted(present_hashes.items())),
            "verdict": "INCOMPLETE",
            "hlm5_status": "INCOMPLETE",
        }
        atomic_write_json(MANIFEST_PATH, manifest)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        print(f"wrote={repo_relative(COMPARISON_PATH)}")
        print(f"wrote={repo_relative(MANIFEST_PATH)}")
        return
    payloads: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, str] = {}
    for name, filename, producer in HLM_SPECS:
        path = STAGING_DIR / filename
        expected_upstream = {}
        if name == "multi_key":
            one_key_path = STAGING_DIR / "cert_envelope_synth.json"
            expected_upstream = {
                repo_relative(one_key_path): sha256_file(one_key_path)
            }
        payload, digest = require_staged_artifact(
            path,
            producer,
            HLM_SAMPLE_CONTRACTS[name],
            expected_upstream,
        )
        payloads[name] = payload
        artifact_hashes[repo_relative(path)] = digest

    hlm_failures = validate_hlm(payloads, preflight, preflight_sha256)
    counterfact_failures: list[str] = []
    counterfact_summary = None
    counterfact_rows = None
    counterfact_complete = COUNTERFACT_ROWS.is_file() and COUNTERFACT_SUMMARY.is_file()
    if counterfact_complete:
        counterfact_summary, summary_sha256 = require_staged_artifact(
            COUNTERFACT_SUMMARY,
            COUNTERFACT_PRODUCER,
            COUNTERFACT_SAMPLE_CONTRACT,
            {},
        )
        summary_contract = counterfact_summary.get("certificate_contract", {})
        row_contract = {
            key: value
            for key, value in summary_contract.items()
            if key != "jsonl_sha256"
        }
        counterfact_rows = load_contract_jsonl(
            COUNTERFACT_ROWS,
            stable_json_sha256(row_contract),
            require_prefix=True,
        )
        counterfact_failures.extend(
            validate_counterfact(
                counterfact_summary,
                counterfact_rows,
                preflight,
                preflight_sha256,
            )
        )
        artifact_hashes[repo_relative(COUNTERFACT_ROWS)] = sha256_file(
            COUNTERFACT_ROWS
        )
        artifact_hashes[repo_relative(COUNTERFACT_SUMMARY)] = summary_sha256

    new_headlines = headline_values(payloads, counterfact_summary)
    release_headlines = load_release_headlines()
    anchor_mismatches = {
        key: {"release": release_headlines.get(key), "registered": value}
        for key, value in OLD_HEADLINES.items()
        if release_headlines.get(key) != value
    }
    if anchor_mismatches:
        hlm_anchor_mismatches = {
            key: value
            for key, value in anchor_mismatches.items()
            if not key.startswith("counterfact_")
        }
        counterfact_anchor_mismatches = {
            key: value
            for key, value in anchor_mismatches.items()
            if key.startswith("counterfact_")
        }
        if hlm_anchor_mismatches:
            hlm_failures.append(
                "registered HLM5 release headlines do not match tracked artifacts: "
                f"{hlm_anchor_mismatches}"
            )
        if counterfact_anchor_mismatches:
            counterfact_failures.append(
                "registered CounterFact release headlines do not match tracked "
                f"artifacts: {counterfact_anchor_mismatches}"
            )
    categorical_changes = row_level_changes(payloads, counterfact_rows)
    release_artifact_hashes = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in RELEASE_ARTIFACTS
    }
    comparisons = {
        key: {
            "old": OLD_HEADLINES[key],
            "new": value,
            "equal": OLD_HEADLINES[key] == value,
        }
        for key, value in new_headlines.items()
    }
    stable = all(item["equal"] for item in comparisons.values())
    failures = hlm_failures + counterfact_failures
    if failures:
        verdict = "IMPLEMENTATION_INVALID"
    elif not counterfact_complete:
        verdict = "INCOMPLETE"
    elif stable:
        verdict = "VALID_REFRESH_STABLE_CLAIMS"
    else:
        verdict = "VALID_REFRESH_CHANGED_CLAIMS"
    hlm5_stable = all(
        item["equal"]
        for key, item in comparisons.items()
        if not key.startswith("counterfact_")
    )
    hlm5_status = (
        "IMPLEMENTATION_INVALID"
        if hlm_failures
        else "VALID_REFRESH_STABLE_CLAIMS"
        if hlm5_stable
        else "VALID_REFRESH_CHANGED_CLAIMS"
    )
    report = {
        "schema": "hlm5-certificate-refresh-comparison-v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "preflight_sha256": preflight_sha256,
        "preflight_status": preflight["status"],
        "verdict": verdict,
        "hlm5_status": hlm5_status,
        "counterfact_complete": counterfact_complete,
        "validity_failures": failures,
        "hlm5_validity_failures": hlm_failures,
        "counterfact_validity_failures": counterfact_failures,
        "headline_comparisons": comparisons,
        "tracked_release_headlines": release_headlines,
        "row_level_categorical_changes": categorical_changes,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "release_artifact_sha256": dict(sorted(release_artifact_hashes.items())),
    }
    atomic_write_json(COMPARISON_PATH, report)
    comparison_sha256 = sha256_file(COMPARISON_PATH)
    manifest = {
        "schema": "hlm5-certificate-refresh-manifest-v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "preflight_sha256": preflight_sha256,
        "comparison_sha256": comparison_sha256,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "release_artifact_sha256": dict(sorted(release_artifact_hashes.items())),
        "verdict": verdict,
        "hlm5_status": hlm5_status,
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(f"wrote={repo_relative(COMPARISON_PATH)}")
    print(f"wrote={repo_relative(MANIFEST_PATH)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
