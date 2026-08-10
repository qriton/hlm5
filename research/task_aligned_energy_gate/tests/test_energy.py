"""Fast mathematical tests for the task-aligned energy implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from energy import (  # noqa: E402
    aligned_scores,
    constrained_settle,
    decision_boundary_radius,
    fit_aligned_model,
    independent_matched_settle,
    mixture_energy,
    negative_energy_gradient,
    transform,
    unit,
)


class EnergyTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(1234)
        features = []
        targets = []
        for class_index in range(4):
            center = torch.zeros(8, dtype=torch.float64)
            center[class_index] = 1.0
            noise = 0.04 * torch.randn(
                (16, 8), generator=generator, dtype=torch.float64
            )
            features.append(unit(center + noise))
            targets.extend([class_index] * 16)
        self.features = torch.cat(features)
        self.targets = torch.tensor(targets, dtype=torch.int64)
        self.model = fit_aligned_model(self.features, self.targets, 4, 0.25)
        self.states = transform(self.features, self.model)
        self.radius = decision_boundary_radius(
            self.states, self.targets, self.model.centroids
        )

    def kwargs(self, steps: int) -> dict[str, int | float]:
        return {
            "temperature": 1.0,
            "steps": steps,
            "initial_step_radius_fraction": 0.5,
            "armijo_c": 1e-4,
            "max_halvings": 24,
            "stationary_tolerance": 1e-12,
            "energy_tolerance": 1e-12,
            "cap_tolerance": 1e-10,
        }

    def test_fit_shapes_and_positive_covariance(self) -> None:
        self.assertEqual(tuple(self.model.projection.shape), (3, 8))
        self.assertEqual(tuple(self.model.centroids.shape), (4, 3))
        self.assertGreater(float(self.model.within_eigenvalues.min()), 0.0)
        self.assertTrue(torch.isfinite(self.model.projection).all())

    def test_lda_scores_equal_negative_squared_distance_up_to_common_term(self) -> None:
        scores = aligned_scores(self.states, self.model.centroids)
        distances = -0.5 * torch.sum(
            (self.states.unsqueeze(1) - self.model.centroids.unsqueeze(0)) ** 2,
            dim=2,
        )
        self.assertTrue(
            torch.equal(torch.argmax(scores, dim=1), torch.argmax(distances, dim=1))
        )

    def test_analytic_negative_gradient_matches_autograd(self) -> None:
        states = self.states[:5].clone().requires_grad_(True)
        energy = mixture_energy(states, self.model.centroids, 1.0).sum()
        gradient = torch.autograd.grad(energy, states)[0]
        analytic = negative_energy_gradient(states.detach(), self.model.centroids, 1.0)
        self.assertLess(float(torch.max(torch.abs(gradient + analytic))), 1e-12)

    def test_settle_descends_respects_cap_and_matches_independent(self) -> None:
        initial = self.states[:12]
        candidate, trace = constrained_settle(
            initial, self.model.centroids, self.radius, **self.kwargs(8)
        )
        matched = independent_matched_settle(
            initial, self.model.centroids, self.radius, **self.kwargs(8)
        )
        self.assertLess(float(torch.max(torch.abs(candidate - matched))), 1e-10)
        self.assertLessEqual(
            float(torch.max(trace.energies[:, 1:] - trace.energies[:, :-1])), 1e-12
        )
        self.assertLessEqual(
            float(
                torch.max(
                    torch.linalg.vector_norm(candidate - initial, dim=1) - self.radius
                )
            ),
            1e-10,
        )
        self.assertLessEqual(float(torch.max(trace.armijo_residuals)), 1e-12)

    def test_zero_steps_is_bit_exact(self) -> None:
        initial = self.states[:4]
        identity, trace = constrained_settle(
            initial, self.model.centroids, self.radius, **self.kwargs(0)
        )
        self.assertTrue(torch.equal(identity, initial))
        self.assertEqual(tuple(trace.step_lengths.shape), (4, 0))

    def test_high_dimensional_cap_path_is_bit_exact_between_optimizers(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(20_260_810)
        states = torch.randn((32, 76), generator=generator, dtype=torch.float64) * 2.0
        centroids = (
            torch.randn((77, 76), generator=generator, dtype=torch.float64) * 1.5
        )
        radius = 4.720266704071971
        candidate, _ = constrained_settle(states, centroids, radius, **self.kwargs(8))
        matched = independent_matched_settle(
            states, centroids, radius, **self.kwargs(8)
        )
        self.assertTrue(torch.equal(candidate, matched))


if __name__ == "__main__":
    unittest.main()
