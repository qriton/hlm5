"""Synthetic contract tests for the certified semantic attractor gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attractor import (  # noqa: E402
    certified_settle,
    descent_direction,
    independent_matched_settle,
    polynomial_energy,
    static_cosine_predictions,
    static_degree_predictions,
    unit,
)
from metrics import (  # noqa: E402
    average_rank_auc,
    classify_verdict,
    macro_accuracy,
    maximum_relation_prediction_share,
    paired_relation_bootstrap,
    relation_accuracies,
)


class AttractorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.keys = torch.eye(4, dtype=torch.float64)
        raw = torch.tensor(
            [
                [0.90, 0.30, 0.10, 0.00],
                [0.10, 0.85, 0.25, 0.05],
                [0.05, 0.20, 0.80, 0.15],
            ],
            dtype=torch.float64,
        )
        self.queries = unit(raw)

    def test_direction_is_tangent_and_matches_energy_descent(self) -> None:
        direction = descent_direction(self.queries, self.keys, degree=5)
        radial = torch.sum(direction * self.queries, dim=1)
        self.assertLess(float(torch.abs(radial).max()), 1e-12)
        before = polynomial_energy(self.queries, self.keys, degree=5)
        after = polynomial_energy(unit(self.queries + 1e-4 * direction), self.keys, degree=5)
        self.assertTrue(bool(torch.all(after < before)))

    def test_registered_settle_descends_every_transition(self) -> None:
        trace = certified_settle(self.queries, self.keys)
        deltas = trace.energies[:, 1:] - trace.energies[:, :-1]
        self.assertTrue(bool(torch.all(deltas <= 1e-12)))
        self.assertTrue(bool(torch.all(trace.armijo_residuals <= 1e-12)))
        self.assertTrue(bool(torch.isfinite(trace.final_state).all()))
        self.assertLess(
            float(torch.abs(trace.final_state.norm(dim=1) - 1.0).max()),
            1e-12,
        )

    def test_zero_steps_is_bit_identical(self) -> None:
        trace = certified_settle(self.queries, self.keys, steps=0)
        self.assertTrue(torch.equal(trace.final_state, self.queries))
        self.assertEqual(tuple(trace.energies.shape), (3, 1))
        self.assertEqual(tuple(trace.step_sizes.shape), (3, 0))

    def test_independent_cache_optimizer_matches(self) -> None:
        candidate = certified_settle(self.queries, self.keys).final_state
        matched = independent_matched_settle(self.queries, self.keys)
        self.assertLess(float(torch.abs(candidate - matched).max()), 1e-12)
        candidate_slots, _ = static_cosine_predictions(candidate, self.keys)
        matched_slots, _ = static_cosine_predictions(matched, self.keys)
        self.assertTrue(torch.equal(candidate_slots, matched_slots))

    def test_static_degree_and_cosine_select_same_positive_maximum(self) -> None:
        cosine_slots, _ = static_cosine_predictions(self.queries, self.keys)
        degree_slots, _ = static_degree_predictions(self.queries, self.keys, degree=5)
        self.assertTrue(torch.equal(cosine_slots, degree_slots))


class MetricContractTests(unittest.TestCase):
    def test_relation_macro_and_bootstrap_share_one_estimator(self) -> None:
        relations = ["P1", "P1", "P2", "P2"]
        targets = [0, 1, 2, 3]
        baseline = relation_accuracies([0, 9, 9, 9], targets, relations)
        candidate = relation_accuracies([0, 1, 2, 9], targets, relations)
        self.assertEqual(baseline, {"P1": 0.5, "P2": 0.0})
        self.assertEqual(candidate, {"P1": 1.0, "P2": 0.5})
        self.assertEqual(macro_accuracy(candidate), 0.75)
        interval = paired_relation_bootstrap(
            candidate,
            baseline,
            resamples=100,
            seed=7,
        )
        self.assertEqual(interval["point"], 0.5)
        self.assertEqual(interval, paired_relation_bootstrap(
            candidate,
            baseline,
            resamples=100,
            seed=7,
        ))

    def test_auc_uses_average_ranks_for_ties(self) -> None:
        self.assertEqual(average_rank_auc([1.0, 1.0], [1.0, 1.0]), 0.5)
        self.assertEqual(average_rank_auc([2.0, 3.0], [0.0, 1.0]), 1.0)

    def test_prediction_share_uses_slot_relation(self) -> None:
        share = maximum_relation_prediction_share(
            [0, 1, 2, 2],
            ["P1", "P1", "P2"],
        )
        self.assertEqual(share, 0.5)

    def test_invalidity_precedes_interruption_and_scientific_bars(self) -> None:
        self.assertEqual(
            classify_verdict({"hashes": False}, {"gain": True}, interrupted=True),
            "HARNESS_INVALID",
        )
        self.assertEqual(
            classify_verdict({"hashes": True}, {"gain": True}, interrupted=True),
            "INCONCLUSIVE",
        )
        self.assertEqual(
            classify_verdict({"hashes": True}, {"gain": False}),
            "FAIL",
        )
        self.assertEqual(
            classify_verdict({"hashes": True}, {"gain": True}),
            "PASS",
        )

    def test_bootstrap_quantiles_use_numpy_linear_convention(self) -> None:
        candidate = {"a": 1.0, "b": 0.25, "c": 0.75, "d": 0.0}
        baseline = {"a": 0.0, "b": 0.0, "c": 0.25, "d": 0.0}
        got = paired_relation_bootstrap(candidate, baseline, resamples=128, seed=3)
        self.assertTrue(np.isfinite([got["point"], got["low"], got["high"]]).all())
        self.assertLessEqual(got["low"], got["point"])
        self.assertGreaterEqual(got["high"], got["point"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
