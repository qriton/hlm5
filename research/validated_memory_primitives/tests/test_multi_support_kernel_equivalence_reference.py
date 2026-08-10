"""CPU contract tests for the HLM5 multi-support equivalence reference.

These tests use a non-registered synthetic seed and never call the scored
``run_reference`` package.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

import multi_support_kernel_equivalence_reference as reference  # noqa: E402


SYNTHETIC_SEED = 123456


class KernelEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = reference.build_seed(SYNTHETIC_SEED)
        self.memory = reference.make_memory(
            self.state["keys"],
            self.state["values"],
        )

    def test_independent_degree5_cache_is_bit_exact(self) -> None:
        queries = self.state["queries"][: reference.PAIRS]
        hlm = reference.read_hlm5(self.memory, queries, self.state["head"])
        cache = reference.read_cache(
            "matched_degree5_cache",
            queries,
            self.memory.keys,
            self.memory.values,
            self.memory.alphas,
            self.memory.active,
            self.state["head"],
        )
        equivalence = reference.exact_equivalence(hlm, cache)
        self.assertTrue(equivalence["all_exact"])
        self.assertTrue(all(equivalence["fields"].values()))

    def test_balanced_decoder_requires_two_live_shares(self) -> None:
        queries = self.state["queries"][: reference.PAIRS]
        labels = self.state["labels"][: reference.PAIRS]
        hlm = reference.read_hlm5(self.memory, queries, self.state["head"])
        hard = reference.read_cache(
            "hard_top1_cache",
            queries,
            self.memory.keys,
            self.memory.values,
            self.memory.alphas,
            self.memory.active,
            self.state["head"],
        )
        self.assertTrue(bool(hlm["prediction"].eq(labels).all()))
        self.assertFalse(bool(hard["prediction"].eq(labels).any()))

        for pair in range(reference.PAIRS):
            self.memory.deactivate(2 * pair + 1)
        one_share = reference.read_hlm5(
            self.memory,
            queries,
            self.state["head"],
        )
        self.assertTrue(bool(one_share["support_size"].eq(1).all()))
        self.assertFalse(bool(one_share["prediction"].eq(labels).any()))

    def test_locked_skew_ladder_keeps_exactly_two_supports(self) -> None:
        hlm = reference.read_hlm5(
            self.memory,
            self.state["queries"],
            self.state["head"],
        )
        self.assertTrue(bool(hlm["support_size"].eq(2).all()))

    def test_unused_basis_queries_are_bit_exact_gate_off(self) -> None:
        for kind in (
            "matched_degree5_cache",
            "cosine_soft_cache",
            "hard_top1_cache",
        ):
            result = reference.read_cache(
                kind,
                self.state["unused_keys"],
                self.memory.keys,
                self.memory.values,
                self.memory.alphas,
                self.memory.active,
                self.state["head"],
            )
            self.assertFalse(bool(result["routed"].any()))
            self.assertTrue(torch.equal(result["output"], torch.zeros_like(result["output"])))
        hlm = reference.read_hlm5(
            self.memory,
            self.state["unused_keys"],
            self.state["head"],
        )
        self.assertTrue(torch.equal(hlm["output"], torch.zeros_like(hlm["output"])))

    def test_pair_rotation_is_a_bijection_over_pairs(self) -> None:
        rotated = reference.rotate_pairs(self.state["values"])
        expected = self.state["values"].reshape(reference.PAIRS, 2, reference.DIM).roll(
            shifts=-1,
            dims=0,
        )
        self.assertTrue(torch.equal(rotated.reshape_as(expected), expected))
        self.assertEqual(int(torch.unique(rotated, dim=0).shape[0]), reference.ACTIVE_SLOTS)


class HarnessContractTests(unittest.TestCase):
    def test_cache_reader_does_not_call_hlm_reader_methods(self) -> None:
        source = inspect.getsource(reference.read_cache)
        self.assertNotIn(".score(", source)
        self.assertNotIn("._selected_scores(", source)
        self.assertNotIn(".forward(", source)

    def test_pinned_lineage_and_initial_registration_match(self) -> None:
        self.assertEqual(
            reference.sha256_file(reference.DEFAULT_PROTOCOL),
            reference.EXPECTED_PROTOCOL_SHA256,
        )
        self.assertEqual(
            reference.sha256_file(reference.DEFAULT_REGISTRATION),
            reference.EXPECTED_REGISTRATION_SHA256,
        )
        for relative, expected in reference.PINNED_LINEAGE.items():
            self.assertEqual(reference.sha256_file(ROOT / relative), expected)
        pinned = json.loads(
            (ROOT / "baselines" / "results" / "exact_address_lifecycle_reference.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(pinned["status"], reference.PINNED_REFERENCE_STATUS)
        self.assertEqual(
            pinned["scientific_sha256"],
            reference.PINNED_REFERENCE_SCIENTIFIC_SHA256,
        )

    def test_execution_receipt_binds_tool_test_and_command(self) -> None:
        hashes = reference.verify_contract()
        self.assertEqual(hashes["tool"], reference.sha256_file(Path(reference.__file__)))
        self.assertEqual(hashes["test"], reference.sha256_file(Path(__file__)))

    def test_cli_rejects_alternate_output_before_assay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            alternate = Path(temporary) / "result.json"
            with mock.patch.object(
                sys,
                "argv",
                ["reference", "--out", str(alternate)],
            ):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        reference.parse_args()
            self.assertFalse(alternate.exists())

    def test_classification_precedence(self) -> None:
        def phase(accuracy: float, balanced: float | None = None) -> dict:
            return {
                "accuracy": accuracy,
                "per_ratio": {
                    f"{ratio:.2f}": {
                        "accuracy": accuracy if ratio != 1.0 else (
                            accuracy if balanced is None else balanced
                        )
                    }
                    for ratio in reference.MIX_RATIOS
                },
            }

        def aggregate(accuracy: float, balanced: float, control: float = 0.0) -> dict:
            return {
                "phases": {
                    "create": {
                        "hlm5_degree5": phase(accuracy, balanced),
                        "cosine_soft_cache": phase(accuracy),
                        "hard_top1_cache": phase(control),
                    },
                    "reactivate": {"hlm5_degree5": phase(accuracy, balanced)},
                    "pair_edit": {"hlm5_degree5": phase(accuracy, balanced)},
                    "deactivate_one_share": {"hlm5_degree5": phase(control)},
                    "shuffled_value_null": {"accuracy": control},
                }
            }

        valid = {"contract": True}
        invalid = {"contract": False}
        self.assertEqual(reference.classify(aggregate(1.0, 1.0), invalid)[0], "INVALID")
        self.assertEqual(
            reference.classify(aggregate(0.4, 1.0), valid)[0],
            "NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT",
        )
        self.assertEqual(
            reference.classify(aggregate(1.0, 1.0), valid)[0],
            "ROBUST_MULTI_SUPPORT_CACHE_EQUIVALENT",
        )
        self.assertEqual(
            reference.classify(aggregate(0.4, 0.0), valid)[0],
            "NO_MULTI_SUPPORT_FUNCTION",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
