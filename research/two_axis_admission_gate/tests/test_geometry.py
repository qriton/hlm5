"""CPU tests for the SCA-2 class-margin and gate estimators."""

from __future__ import annotations

import unittest

import torch

from research.two_axis_admission_gate.geometry import (
    calibrate_arm,
    calibrate_thresholds,
    class_route_scores,
    fit_key_operators,
    gate_masks,
    higher_quantile,
    matched_proximity_mask,
)


class GeometryTests(unittest.TestCase):
    def test_class_margin_uses_runner_up_class_not_runner_up_slot(self) -> None:
        support = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.99, 0.01],
                [0.0, 0.98, 0.02],
            ],
            dtype=torch.float64,
        )
        support = support / torch.linalg.vector_norm(support, dim=1, keepdim=True)
        query = torch.tensor([[0.8, 0.6, 0.0]], dtype=torch.float64)
        labels = ["A", "A", "A", "B", "B", "B"]
        route = class_route_scores(query, support, labels)
        self.assertEqual(route["predicted"], ["A"])
        expected = route["class_scores"][0, 0] - route["class_scores"][0, 1]
        self.assertEqual(route["class_margins"][0], expected)
        slot_values = torch.sort(route["scores"][0], descending=True).values
        slot_margin = slot_values[0] - slot_values[1]
        self.assertGreater(route["class_margins"][0], slot_margin)

    def test_zero_score_tie_has_zero_margin(self) -> None:
        support = torch.eye(3, dtype=torch.float64)
        query = -torch.ones((1, 3), dtype=torch.float64)
        route = class_route_scores(query, support, ["A", "B", "C"])
        self.assertEqual(route["predicted"], ["A"])
        self.assertEqual(float(route["maximum_scores"][0]), 0.0)
        self.assertEqual(float(route["class_margins"][0]), 0.0)

    def test_higher_quantile_and_strict_gates_bound_calibration(self) -> None:
        negative = {
            "maximum_scores": torch.arange(20, dtype=torch.float64),
        }
        positive = {
            "predicted": ["wrong"] * 10,
            "class_margins": torch.arange(10, dtype=torch.float64),
        }
        thresholds = calibrate_thresholds(negative, positive, ["truth"] * 10)
        self.assertEqual(thresholds["tau_absolute"], 19.0)
        self.assertEqual(thresholds["tau_margin"], 9.0)
        self.assertEqual(thresholds["calibration_negative_absolute_rate"], 0.0)
        self.assertEqual(
            thresholds["calibration_error_margin_survival_rate"], 0.0
        )

    def test_insufficient_errors_stops_margin_calibration(self) -> None:
        negative = {"maximum_scores": torch.arange(20, dtype=torch.float64)}
        positive = {
            "predicted": ["truth"] * 9 + ["wrong"] * 7,
            "class_margins": torch.arange(16, dtype=torch.float64),
        }
        labels = ["truth"] * 16
        thresholds = calibrate_thresholds(negative, positive, labels)
        self.assertFalse(thresholds["sufficient_margin_errors"])
        self.assertIsNone(thresholds["tau_margin"])

    def test_calibration_records_independent_reader_errors(self) -> None:
        operators = fit_key_operators(torch.eye(4, dtype=torch.float64))
        support = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.9, -0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [-0.1, 0.9, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        labels = ["a", "a", "a", "b", "b", "b"]
        metrics, _ = calibrate_arm(
            operators,
            "raw",
            support,
            labels,
            support,
            labels,
            -support,
        )
        self.assertLessEqual(metrics["positive_independent_max_abs_error"], 1e-10)
        self.assertLessEqual(metrics["negative_independent_max_abs_error"], 1e-10)

    def test_matched_proximity_is_exact_and_ties_use_row_id(self) -> None:
        route = {
            "maximum_scores": torch.tensor([0.9, 0.8, 0.8, 0.7]),
        }
        absolute = torch.tensor([True, True, True, True])
        selected = matched_proximity_mask(
            route, absolute, 2, ["d", "c", "a", "b"]
        )
        self.assertEqual(selected.tolist(), [True, False, True, False])

    def test_gate_requires_both_axes_strictly(self) -> None:
        route = {
            "maximum_scores": torch.tensor([0.5, 0.6, 0.6]),
            "class_margins": torch.tensor([0.4, 0.3, 0.4]),
        }
        masks = gate_masks(route, {"tau_absolute": 0.5, "tau_margin": 0.3})
        self.assertEqual(masks["absolute_only"].tolist(), [False, True, True])
        self.assertEqual(masks["dual"].tolist(), [False, False, True])

    def test_higher_quantile_rejects_empty_vector(self) -> None:
        with self.assertRaises(ValueError):
            higher_quantile(torch.empty(0), 0.95)


if __name__ == "__main__":
    unittest.main()
