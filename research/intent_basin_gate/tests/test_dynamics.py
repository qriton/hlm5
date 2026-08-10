"""Analytic and synthetic tests for the I1 dynamics and estimators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics import (  # noqa: E402
    angular_distance,
    class_degree_scores,
    constrained_settle,
    independent_matched_settle,
    project_to_cap,
    unit,
)
from metrics import (  # noqa: E402
    average_rank_auc,
    classify_development_verdict,
    maximum_intent_prediction_share,
    paired_intent_bootstrap,
    select_static_control,
)


def settle_kwargs(steps: int = 8) -> dict[str, int | float]:
    return {
        "degree": 5,
        "steps": steps,
        "initial_step_radius_fraction": 0.5,
        "armijo_c": 1e-4,
        "max_halvings": 24,
        "stationary_tolerance": 1e-12,
        "energy_tolerance": 1e-12,
        "cap_tolerance": 1e-10,
    }


class DynamicsContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prototypes = unit(
            torch.tensor(
                [
                    [1.0, 0.2, 0.0, 0.0],
                    [0.9, 0.3, 0.1, 0.0],
                    [0.0, 0.1, 1.0, 0.2],
                    [0.1, 0.0, 0.9, 0.3],
                ],
                dtype=torch.float64,
            )
        )
        self.queries = unit(
            torch.tensor(
                [[0.75, 0.45, 0.15, 0.0], [0.15, 0.0, 0.75, 0.45]],
                dtype=torch.float64,
            )
        )

    def test_registered_settle_descends_stays_in_cap_and_matches(self) -> None:
        radius = 0.40
        candidate, trace = constrained_settle(
            self.queries,
            self.prototypes,
            radius,
            **settle_kwargs(),
        )
        matched = independent_matched_settle(
            self.queries,
            self.prototypes,
            radius,
            **settle_kwargs(),
        )
        self.assertLessEqual(
            float((trace.energies[:, 1:] - trace.energies[:, :-1]).max()),
            1e-12,
        )
        self.assertLessEqual(float(trace.armijo_residuals.max()), 1e-12)
        self.assertLessEqual(
            float((angular_distance(self.queries, candidate) - radius).max()),
            1e-10,
        )
        self.assertLessEqual(float(torch.abs(candidate - matched).max()), 1e-10)
        self.assertLessEqual(
            float(torch.abs(candidate.norm(dim=1) - 1.0).max()), 1e-12
        )

    def test_zero_steps_is_bit_exact(self) -> None:
        candidate, trace = constrained_settle(
            self.queries,
            self.prototypes,
            0.40,
            **settle_kwargs(steps=0),
        )
        self.assertTrue(torch.equal(candidate, self.queries))
        self.assertEqual(tuple(trace.step_angles.shape), (2, 0))
        self.assertEqual(tuple(trace.energies.shape), (2, 1))

    def test_cap_projection_hits_registered_boundary(self) -> None:
        origins = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        proposals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        projected, outside = project_to_cap(origins, proposals, 0.25)
        self.assertTrue(bool(outside.item()))
        self.assertAlmostEqual(
            float(angular_distance(origins, projected).item()), 0.25, places=12
        )

    def test_class_degree_scores_average_equal_supports(self) -> None:
        prototypes = self.prototypes.reshape(2, 2, 4)
        scores = class_degree_scores(self.queries, prototypes, degree=5)
        manual = torch.relu(self.queries @ self.prototypes.T).pow(5).reshape(2, 2, 2)
        self.assertTrue(torch.allclose(scores, manual.mean(dim=-1), atol=0, rtol=0))

    def test_outward_cap_direction_becomes_exact_stationary_copy(self) -> None:
        prototypes = unit(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            )
        )
        query = unit(torch.tensor([[0.6, 0.8, 0.0]], dtype=torch.float64))
        candidate, trace = constrained_settle(
            query,
            prototypes,
            0.10,
            **settle_kwargs(steps=4),
        )
        self.assertEqual(trace.constrained_stationary.tolist(), [[False, False, True, True]])
        self.assertEqual(trace.step_angles[0, 2:].tolist(), [0.0, 0.0])
        self.assertLessEqual(
            float(angular_distance(query, candidate).item()), 0.10 + 1e-12
        )


class MetricContractTests(unittest.TestCase):
    def test_bootstrap_point_uses_same_arithmetic_estimator(self) -> None:
        candidate = {"a": 1.0, "b": 0.0, "c": 0.75, "d": 0.25}
        baseline = {"a": 0.0, "b": 0.0, "c": 0.25, "d": 0.25}
        result = paired_intent_bootstrap(candidate, baseline, resamples=100, seed=7)
        self.assertEqual(result["point"], 0.375)

    def test_auc_uses_average_ranks_for_ties(self) -> None:
        self.assertEqual(average_rank_auc([1.0, 0.5], [0.5, 0.0]), 0.875)

    def test_static_selection_is_highest_then_lexicographic(self) -> None:
        selected = select_static_control({"z": 0.8, "a": 0.8, "b": 0.7})
        self.assertEqual(selected, "a")

    def test_invalidity_precedes_interruption_and_scientific_bars(self) -> None:
        self.assertEqual(
            classify_development_verdict(
                {"hashes": False}, {"gain": True}, interrupted=True
            ),
            "HARNESS_INVALID",
        )
        self.assertEqual(
            classify_development_verdict(
                {"hashes": True}, {"gain": False}, interrupted=False
            ),
            "STOP_BEFORE_TEST",
        )

    def test_prediction_share_detects_collapse(self) -> None:
        self.assertEqual(maximum_intent_prediction_share([1, 1, 1, 2]), 0.75)


if __name__ == "__main__":
    unittest.main()
