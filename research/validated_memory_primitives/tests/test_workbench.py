"""CPU checks for the curated HLM5 validated-primitives workbench."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_workbench as workbench  # noqa: E402


class ValidatedMemoryPrimitivesWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = workbench.verify_snapshot()
        cls.frozen_exact, cls.frozen_multi = workbench.verify_frozen_evidence()
        cls.replayed_exact, cls.replayed_multi = workbench.replay_references()
        cls.report = workbench.build_report(
            cls.manifest,
            cls.replayed_exact,
            cls.replayed_multi,
        )

    def test_snapshot_has_only_the_pinned_minimal_reference_tree(self) -> None:
        self.assertEqual(len(self.manifest["files"]), 18)
        self.assertEqual(
            self.manifest["schema"],
            "hlm5-validated-memory-primitives-snapshot-v1",
        )

    def test_replays_are_identical_to_frozen_results(self) -> None:
        self.assertEqual(self.replayed_exact, self.frozen_exact)
        self.assertEqual(self.replayed_multi, self.frozen_multi)

    def test_exact_address_lifecycle_is_complete_and_cache_equivalent(self) -> None:
        exact = self.report["working"]["exact_address_lifecycle"]
        self.assertTrue(all(exact["operations"].values()))
        self.assertTrue(exact["matched_degree5_cache_bit_exact"])

    def test_multi_support_result_is_narrow_and_cache_equivalent(self) -> None:
        multi = self.report["working"]["bounded_multi_support"]
        self.assertEqual(multi["hlm5_degree5_accuracy"], 0.40)
        self.assertEqual(multi["matched_degree5_cache_accuracy"], 0.40)
        self.assertEqual(multi["cosine_soft_cache_accuracy"], 0.60)
        self.assertEqual(multi["hard_top1_accuracy"], 0.0)
        self.assertTrue(multi["matched_degree5_cache_bit_exact"])

    def test_controls_and_off_support_gate_are_binding(self) -> None:
        multi = self.report["working"]["bounded_multi_support"]
        self.assertEqual(multi["shuffled_value_null_accuracy"], 0.0)
        self.assertEqual(multi["one_share_deactivated_accuracy"], 0.0)
        self.assertTrue(multi["off_support_outputs_bit_zero"])

    def test_claim_boundaries_remain_explicit(self) -> None:
        nonclaims = " ".join(self.report["do_not_claim"])
        self.assertIn("HLM-specific retrieval advantage", nonclaims)
        self.assertIn("semantic or paraphrase routing", nonclaims)
        self.assertIn("training or scaling authorization", nonclaims)

    def test_snapshot_hash_mismatch_fails_closed(self) -> None:
        real_sha = workbench.sha256_file

        def changed_hash(path: Path) -> str:
            if path.name == "hlm5_memory.py":
                return "0" * 64
            return real_sha(path)

        with mock.patch.object(workbench, "sha256_file", side_effect=changed_hash):
            with self.assertRaises(workbench.WorkbenchError):
                workbench.verify_snapshot()

    def test_generated_output_is_outside_the_pinned_snapshot(self) -> None:
        relative = workbench.OUTPUT.relative_to(workbench.ROOT).as_posix()
        self.assertTrue(relative.startswith("outputs/"))
        self.assertNotIn(relative, self.manifest["files"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
