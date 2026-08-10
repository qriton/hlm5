"""Tests for locked selection, estimators, and verdict precedence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract import DEV_REGISTERED_COMMAND, TEST_REGISTERED_COMMAND  # noqa: E402
from metrics import (  # noqa: E402
    classify_development_verdict,
    paired_intent_bootstrap,
    select_highest_macro,
    select_shrinkage,
)


class RegistrationTests(unittest.TestCase):
    def test_selection_ties_are_locked(self) -> None:
        self.assertEqual(
            select_highest_macro({"raw": 0.8, "centered": 0.8}), "centered"
        )
        self.assertEqual(select_shrinkage({0.25: 0.9, 0.05: 0.9}), 0.05)

    def test_bootstrap_point_uses_same_paired_mean(self) -> None:
        candidate = {"a": 1.0, "b": 0.5, "c": 0.0}
        baseline = {"a": 0.5, "b": 0.5, "c": 0.5}
        interval = paired_intent_bootstrap(candidate, baseline, resamples=1000, seed=7)
        self.assertAlmostEqual(interval["point"], 0.0)
        self.assertLessEqual(interval["low"], interval["point"])
        self.assertGreaterEqual(interval["high"], interval["point"])

    def test_validity_has_precedence_and_claims_are_separable(self) -> None:
        valid = {"a": True}
        alignment = {"a": True}
        dynamics = {"a": False}
        self.assertEqual(
            classify_development_verdict(valid, alignment, dynamics),
            "PASS_ALIGNMENT_TO_TEST",
        )
        self.assertEqual(
            classify_development_verdict(valid, alignment, {"a": True}),
            "PASS_BOTH_TO_TEST",
        )
        self.assertEqual(
            classify_development_verdict({"a": False}, alignment, {"a": True}),
            "HARNESS_INVALID",
        )
        self.assertEqual(
            classify_development_verdict(
                valid, alignment, {"a": True}, interrupted=True
            ),
            "INCONCLUSIVE",
        )

    def test_commands_are_exact_and_test_is_distinct(self) -> None:
        self.assertEqual(
            DEV_REGISTERED_COMMAND,
            "python -B run_task_aligned_energy_gate.py --run-development",
        )
        self.assertEqual(
            TEST_REGISTERED_COMMAND,
            "python -B run_task_aligned_energy_gate.py --run-test",
        )
        self.assertNotEqual(DEV_REGISTERED_COMMAND, TEST_REGISTERED_COMMAND)


if __name__ == "__main__":
    unittest.main()
