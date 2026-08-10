"""CPU contract tests for the frozen-3B E7 portability study."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from hlm5.e7_contract import (
    TARGET_POOL_COUNT,
    _require_staging_path,
    registered_samples,
    scientific_failures,
    stable_json_sha256,
    target_pool,
)
from hlm5.memory import EditableHLM5Memory, unit
from hlm5.public_adapter import HLM5PreHeadAdapter

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_e7_3b.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("hlm5_e7_runner", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(_RUNNER)
build_slope_matrix = _RUNNER.build_slope_matrix
envelope_for_key = _RUNNER.envelope_for_key
grid_dose = _RUNNER.grid_dose
scalar_original_check = _RUNNER.scalar_original_check


class _FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            "Ġalpha": 9,
            "Ġbeta": 2,
            "Ġab": 1,
            "plain": 3,
            "Ġtwo2": 4,
            "ĠGamma": 7,
        }


def _memory() -> EditableHLM5Memory:
    torch.manual_seed(7)
    memory = EditableHLM5Memory(
        dim=4,
        memory_size=3,
        degree=5,
        temperature=0.10,
        learnable_memory=False,
    )
    first = memory.inject(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        value=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    second = memory.inject(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        value=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )
    memory.values[first].copy_(torch.tensor([0.0, 2.0, 0.0, 0.0]))
    memory.values[second].copy_(torch.tensor([0.0, 0.0, 3.0, 0.0]))
    return memory


def test_adapter_matches_released_hlm5_hook() -> None:
    memory = _memory()
    mean = torch.tensor([0.2, -0.1, 0.0, 0.0])
    transform = torch.tensor(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 1.2],
        ]
    )
    hidden = torch.tensor(
        [[[1.2, -0.1, 0.0, 0.0], [0.2, 0.9, 0.0, 0.0]]]
    )
    adapter = HLM5PreHeadAdapter(
        memory,
        mean,
        transform,
        boost=1.0,
        gate_thresh=0.50,
    )

    actual, audit = adapter(hidden, return_audit=True)
    query = (hidden - mean) @ transform
    scores = memory.score(query)
    gain = torch.relu(scores).amax(-1, keepdim=True)
    gain = torch.where(gain >= 0.50, gain, torch.zeros_like(gain))
    attention = F.softmax(scores / memory.temperature, dim=-1)
    expected = hidden + gain * (attention @ memory.values)

    assert torch.equal(actual, expected)
    assert audit is not None
    assert torch.equal(audit.gate_open, gain.squeeze(-1) > 0)
    assert torch.allclose(audit.delta, expected - hidden, atol=1e-7, rtol=0.0)


def test_gate_off_is_bit_identical_and_forced_on_control_changes_output() -> None:
    memory = _memory()
    hidden = torch.tensor([[[0.8, 0.6, 0.0, 0.0]]], dtype=torch.float32)
    head = torch.nn.Linear(4, 6, bias=False)
    off = HLM5PreHeadAdapter(
        memory,
        torch.zeros(4),
        torch.eye(4),
        gate_thresh=0.95,
    )
    forced = HLM5PreHeadAdapter(
        memory,
        torch.zeros(4),
        torch.eye(4),
        gate_thresh=0.0,
    )

    off_hidden, off_audit = off(hidden, return_audit=True)
    forced_hidden, forced_audit = forced(hidden, return_audit=True)
    baseline = head(hidden)

    assert off_audit is not None and not bool(off_audit.gate_open.item())
    assert torch.equal(off_hidden, hidden)
    assert torch.equal(head(off_hidden), baseline)
    assert forced_audit is not None and bool(forced_audit.gate_open.item())
    assert not torch.equal(head(forced_hidden), baseline)


def test_empty_memory_is_exact_noop() -> None:
    memory = EditableHLM5Memory(
        dim=4,
        memory_size=2,
        learnable_memory=False,
    )
    hidden = torch.randn(2, 3, 4)
    adapter = HLM5PreHeadAdapter(memory, torch.zeros(4), torch.eye(4))
    actual, audit = adapter(hidden, return_audit=True)

    assert torch.equal(actual, hidden)
    assert audit is not None
    assert not bool(audit.gate_open.any())
    assert bool(torch.isneginf(audit.scores).all())


def test_registered_samples_are_recovered_from_pinned_sources() -> None:
    samples = registered_samples()

    assert len(samples["FACTS"]) == 18
    assert len(samples["NEUTRAL"]) == 8
    assert len(samples["WHITEN_TEXT"]) == 18
    assert len(samples["NOVEL"]) == 20
    assert len(samples["REAL"]) == 20
    assert len(samples["GENERIC"]) == 20
    assert samples["NOVEL"][0] == samples["ORIG_FACT"]


def test_target_pool_rule_is_sorted_and_filtered() -> None:
    pool = target_pool(_FakeTokenizer(), require_registered=False)

    assert pool == [2, 7, 9]
    assert len(pool) <= TARGET_POOL_COUNT
    assert stable_json_sha256(pool) == stable_json_sha256([2, 7, 9])


def test_output_paths_fail_closed_outside_staging(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must stay under"):
        _require_staging_path(tmp_path / "result.json")


def test_unit_value_keeps_registered_direction() -> None:
    value = torch.tensor([3.0, 4.0, 0.0, 0.0])
    assert torch.allclose(unit(value), torch.tensor([0.6, 0.8, 0.0, 0.0]))


def test_scientific_verdict_bars_are_fail_closed() -> None:
    accepted_exact = [
        {
            "kind": "exact",
            "fact_index": index,
            "gate_open": True,
            "target_success": True,
        }
        for index in (0, 1)
    ]
    refused_exact = [
        {
            "kind": "exact",
            "fact_index": 2,
            "gate_open": False,
        }
    ]
    paraphrases = [
        {"kind": "paraphrase", "fact_index": index // 3, "gate_open": False}
        for index in range(54)
    ]
    neutrals = [
        {
            "kind": "neutral",
            "fact_index": None,
            "gate_open": False,
            "logits_bit_identical": True,
            "baseline_logits_sha256": "a" * 64,
            "adapted_logits_sha256": "a" * 64,
        }
        for _ in range(8)
    ]
    result = {
        "direct_certified_injection": {
            "accepted_count": 3,
            "ordinary_head_success_count": 3,
            "rows": [{"ordinary_head_success": True} for _ in range(3)],
        },
        "faithful": {
            "accepted_count": 2,
            "refused_count": 1,
            "facts": [
                {"fact_index": 0, "admission": "ADMIT"},
                {"fact_index": 1, "admission": "ADMIT"},
                {"fact_index": 2, "admission": "REFUSE"},
            ],
            "gate_rows": accepted_exact + refused_exact + paraphrases + neutrals,
            "summary": {
                "accepted_exact_checked": 2,
                "accepted_exact_gate_successes": 2,
                "accepted_exact_full_successes": 2,
                "refused_exact_checked": 1,
                "paraphrases_checked": 54,
                "neutral_checked": 8,
                "false_gate_applications": 0,
                "neutral_bit_identical": 8,
            },
        },
    }
    assert scientific_failures(result) == []

    result["faithful"]["gate_rows"][3]["gate_open"] = True
    failures = scientific_failures(result)
    assert failures == ["an off-support query opened the gate"]


def test_vector_envelope_matches_independent_scalar_reduction() -> None:
    head = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.4, 0.2, 0.8],
            [0.3, -0.7, 0.5],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    _directions, slopes = build_slope_matrix(head, targets)
    base = head @ torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    summary, arrays = envelope_for_key(
        base,
        slopes,
        targets,
        keep_arrays=True,
    )

    assert arrays is not None
    scalar = scalar_original_check(base, slopes, targets, arrays)
    assert scalar["all_match"] is True
    assert scalar["targets_checked"] == 2
    assert sum(summary["risk_counts"].values()) == 2


def test_registered_grid_dose_is_strictly_inside_and_positive() -> None:
    intercept = torch.tensor([float("inf"), -1.0, 1.0], dtype=torch.float64)
    slope = torch.tensor([0.0, 2.0, -0.1], dtype=torch.float64)
    beta, margin = grid_dose(intercept, slope, lower=0.5, upper=10.0)

    assert 0.5 < beta < 10.0
    assert margin > 0.0
