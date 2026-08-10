"""CPU contracts for the registered HLM5 E5-v3 harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))

from hlm5_memory import EditableHLM5Memory  # noqa: E402
from run_address_conditioning_v1 import (  # noqa: E402
    CANONICAL_DOMAIN,
    HarnessInvalidError,
    collision_registry,
    evaluate_operator,
    export_lifecycle_bundle,
    memory_state_equal,
    pair_equivalence,
    parse_args,
    require_harness_invariant,
    restore_lifecycle_bundle,
    scientific_sha256,
    select_formal_verdict,
    shake_address,
    support_route_diagnostics,
    unique_support_signature,
)


def basis(index: int, dim: int = 4) -> torch.Tensor:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value


class CanonicalAddressTests(unittest.TestCase):
    def test_shake_address_is_normalized_deterministic_and_domain_bound(self) -> None:
        first = shake_address("capital_of|strasse", 64, CANONICAL_DOMAIN)
        second = shake_address("capital_of|strasse", 64, CANONICAL_DOMAIN)
        other = shake_address("capital_of|strasse", 64, "other-domain")
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertAlmostEqual(float(first.norm()), 1.0, places=6)
        self.assertEqual(set(first.tolist()), {-0.125, 0.125})

    def test_collision_plan_is_nested_and_exposes_rejected_false_route(self) -> None:
        keys = torch.stack(
            (
                basis(0),
                torch.tensor([0.999, 0.045, 0.0, 0.0]),
                basis(1),
                basis(2),
            )
        )
        order = torch.arange(4)
        short = collision_registry(keys, order, 2, tau_cos=0.95)
        full = collision_registry(keys, order, 4, tau_cos=0.95)
        self.assertEqual(short.accepted_flags, full.accepted_flags[:2])
        self.assertEqual(short.accepted_global, [0])
        self.assertEqual(short.rejected_global, [1])

        memory = EditableHLM5Memory(
            dim=4,
            memory_size=4,
            degree=5,
            read_mode="support_masked",
            support_threshold=0.95**5,
            learnable_memory=False,
        )
        memory.inject(keys[0], value=basis(3))
        hidden = torch.zeros(1, 4)
        head = torch.eye(4)
        rejected = evaluate_operator(
            memory,
            keys[1:2],
            hidden,
            head,
            "d5",
            0.95,
            1,
        )
        self.assertTrue(bool(rejected["routed"].item()))
        self.assertTrue(bool(rejected["hidden_changed"].item()))


class LifecycleBundleTests(unittest.TestCase):
    def test_pending_deactivation_round_trips_and_reactivates(self) -> None:
        memory = EditableHLM5Memory(
            dim=4,
            memory_size=2,
            degree=5,
            read_mode="support_masked",
            support_threshold=0.95**5,
            learnable_memory=False,
        )
        slot = memory.inject(
            basis(0),
            value=basis(1),
            value_scale=3.0,
            strength=0.7,
            label="fact:0",
        )
        snapshot = memory.deactivate(slot)
        blob = export_lifecycle_bundle(
            memory,
            {0: snapshot},
            address_mode="canonical",
            address_domain=CANONICAL_DOMAIN,
        )
        restored, snapshots, metadata = restore_lifecycle_bundle(
            blob,
            torch.device("cpu"),
        )
        self.assertTrue(memory_state_equal(memory, restored))
        self.assertEqual(metadata["address_domain"], CANONICAL_DOMAIN)
        self.assertEqual(snapshots[0], snapshot)
        restored.reactivate(snapshots[0])
        self.assertTrue(bool(restored.active[slot]))
        self.assertAlmostEqual(float(restored.alphas[slot]), 0.7, places=6)
        torch.testing.assert_close(restored.values[slot], basis(1) * 3.0)


class OperatorTests(unittest.TestCase):
    def make_memory(self, read_mode: str) -> EditableHLM5Memory:
        memory = EditableHLM5Memory(
            dim=4,
            memory_size=4,
            degree=5,
            temperature=0.1,
            read_mode=read_mode,
            support_threshold=0.95**5 if read_mode == "support_masked" else None,
            learnable_memory=False,
        )
        memory.inject(basis(0), value=basis(2), value_scale=2.0, label="a")
        memory.inject(basis(1), value=basis(3), value_scale=2.0, label="b")
        return memory

    def test_operator_has_live_positive_control_and_real_gate_off(self) -> None:
        memory = self.make_memory("support_masked")
        hidden = torch.zeros(2, 4)
        head = torch.eye(4)
        result = evaluate_operator(
            memory,
            torch.stack((basis(0), -basis(0))),
            hidden,
            head,
            "d5",
            0.95,
            2,
        )
        self.assertTrue(bool(result["hidden_changed"][0]))
        self.assertFalse(bool(result["hidden_changed"][1]))
        self.assertTrue(torch.equal(result["output"][1], hidden[1]))
        self.assertTrue(bool(result["routed"][0]))
        self.assertFalse(bool(result["routed"][1]))

    def test_degree5_and_matched_cache_are_exact_under_unique_support(self) -> None:
        d5 = self.make_memory("support_masked")
        cache = self.make_memory("hard_top1")
        queries = torch.stack((basis(0), basis(1), -basis(0)))
        hidden = torch.zeros(3, 4)
        head = torch.eye(4)
        d5_eval = evaluate_operator(d5, queries, hidden, head, "d5", 0.95, 3)
        cache_eval = evaluate_operator(
            cache,
            queries,
            hidden,
            head,
            "cache",
            0.95,
            3,
        )
        self.assertEqual(d5_eval["signature"], cache_eval["signature"])
        comparison = pair_equivalence(
            {"phase_signatures": {"create": d5_eval["signature"]}},
            {"phase_signatures": {"create": cache_eval["signature"]}},
        )
        self.assertTrue(comparison["exact"])

    def test_multi_support_rejection_does_not_require_output_equivalence(self) -> None:
        common = {
            "routed": torch.tensor([True, True]),
            "selected_slot": torch.tensor([0, 1]),
            "support_size": torch.tensor([2, 1]),
            "prediction": torch.tensor([3, 2]),
        }
        degree5 = {**common, "output": torch.tensor([[0.5, 0.5], [0.0, 1.0]])}
        cache = {
            **common,
            "prediction": torch.tensor([1, 2]),
            "output": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        self.assertNotEqual(
            degree5["output"][0].tolist(),
            cache["output"][0].tolist(),
        )
        self.assertEqual(
            unique_support_signature(degree5),
            unique_support_signature(cache),
        )

    def test_hard_delete_zeroes_route_and_restores_base_operator(self) -> None:
        memory = self.make_memory("support_masked")
        memory.hard_delete(0, rerandomize=False)
        hidden = torch.randn(1, 4)
        head = torch.eye(4)
        result = evaluate_operator(
            memory,
            basis(0)[None, :],
            hidden,
            head,
            "d5",
            0.95,
            1,
        )
        self.assertEqual(int(result["support_size"].item()), 0)
        self.assertFalse(bool(result["routed"].item()))
        self.assertTrue(torch.equal(result["output"], hidden))

    def test_negative_assay_reports_support_histogram_and_false_route_slots(self) -> None:
        memory = self.make_memory("support_masked")
        hidden = torch.zeros(3, 4)
        head = torch.eye(4)
        result = evaluate_operator(
            memory,
            torch.stack((basis(0), -basis(0), basis(1))),
            hidden,
            head,
            "d5",
            0.95,
            3,
        )
        diagnostics = support_route_diagnostics(result, [7, 8, 9])
        self.assertEqual(diagnostics["support_size_histogram"], {"0": 1, "1": 2})
        self.assertEqual(diagnostics["false_route_global_indices"], [7, 9])
        self.assertEqual(diagnostics["false_route_slots"], [0, 1])


class ReportTests(unittest.TestCase):
    def test_registered_cli_rejects_operator_parameter_changes(self) -> None:
        alterations = (
            ("--tau-cos", "0.94"),
            ("--calibration-count", "128"),
            ("--floor-fraction", "0.02"),
            ("--dose-margin", "1.10"),
            ("--dose-epsilon", "0.002"),
        )
        for smoke in (False, True):
            for option, value in alterations:
                with self.subTest(smoke=smoke, option=option):
                    argv = ["runner", option, value]
                    if smoke:
                        argv.append("--smoke")
                    with mock.patch.object(sys, "argv", argv):
                        with mock.patch.object(sys, "stderr"):
                            with self.assertRaises(SystemExit):
                                parse_args()

    def test_harness_invalid_precedes_smoke_verdict(self) -> None:
        progress = {
            "hidden": {"max_passing_k": 4096},
            "canonical": {"max_passing_k": 4096},
        }
        self.assertEqual(
            select_formal_verdict(
                smoke=True,
                harness_invalid=True,
                track_progress=progress,
            ),
            "HARNESS_INVALID",
        )

    def test_failed_bundle_invariant_is_fatal(self) -> None:
        require_harness_invariant(True, "must not raise")
        with self.assertRaisesRegex(HarnessInvalidError, "restore mismatch"):
            require_harness_invariant(False, "restore mismatch")

    def test_scientific_hash_excludes_only_runtime_fields(self) -> None:
        first = {
            "metric": 1.0,
            "environment": {"gpu": "a"},
            "timing_ms": {"score": 1.0},
        }
        second = {
            "metric": 1.0,
            "environment": {"gpu": "b"},
            "timing_ms": {"score": 99.0},
        }
        self.assertEqual(scientific_sha256(first), scientific_sha256(second))
        second["metric"] = 0.0
        self.assertNotEqual(scientific_sha256(first), scientific_sha256(second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
