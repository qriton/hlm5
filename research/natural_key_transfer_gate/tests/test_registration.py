"""Contract tests for NKT-1 registration gating."""

from __future__ import annotations

import unittest

from research.natural_key_transfer_gate.run_natural_key_transfer_gate import (
    scientific_constants,
    snapshot_reader_parity,
)
import torch


class RegistrationTests(unittest.TestCase):
    def test_scientific_constants_are_complete(self) -> None:
        constants = scientific_constants()
        self.assertEqual(constants["candidate"], "source_fisher")
        self.assertEqual(constants["degree"], 5)
        self.assertEqual(constants["shrinkage"], 0.05)
        self.assertEqual(constants["support_per_intent"], 3)
        self.assertTrue(constants["no_test_encoding"])
        self.assertTrue(constants["no_target_metric_fit"])
        self.assertTrue(constants["no_recurrence"])
        self.assertEqual(constants["bootstrap_resamples"], 10_000)

    def test_registered_paths_cover_every_implementation_file(self) -> None:
        from research.natural_key_transfer_gate.contract import IMPLEMENTATION_PATHS

        self.assertEqual(len(IMPLEMENTATION_PATHS), 8)
        self.assertTrue(all(path.is_file() for path in IMPLEMENTATION_PATHS))

    def test_snapshot_reader_matches_positive_and_abstaining_rows(self) -> None:
        supports = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        queries = torch.tensor([[0.8, 0.2], [-1.0, -1.0]], dtype=torch.float64)
        parity = snapshot_reader_parity(queries, supports)
        self.assertTrue(parity["prediction_match"])
        self.assertEqual(parity["prediction_or_abstention_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
