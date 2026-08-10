"""CPU tests for SCA-1 operators, calibration, and binding bars."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from research.source_calibrated_admission_gate.geometry import (
    admission_mask,
    calibration_threshold,
    development_verdict,
    fit_key_operators,
    paired_intent_bootstrap,
    registered_random_basis,
)


class GeometryTests(unittest.TestCase):
    def test_random_basis_is_exactly_repeatable_and_orthogonal(self) -> None:
        first, first_metadata = registered_random_basis(12)
        second, second_metadata = registered_random_basis(12)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_metadata, second_metadata)
        self.assertLess(first_metadata["orthogonality_max_abs_error"], 1e-10)

    def test_blind_and_random_basis_have_matched_spectra(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(7)
        values = torch.randn(96, 12, generator=generator, dtype=torch.float64)
        values[:, 0] *= 5.0
        values[:, 1] *= 0.2

        first = fit_key_operators(values)
        second = fit_key_operators(values)
        metadata = first.metadata

        self.assertLess(metadata["blind_zca"]["inverse_identity_max_abs_error"], 1e-10)
        self.assertLess(metadata["matched_spectrum_max_abs_error"], 1e-10)
        self.assertLess(metadata["matched_spectrum_max_rel_error"], 1e-10)
        self.assertTrue(metadata["basis_hashes_differ"])
        self.assertEqual(
            metadata["random_basis_sha256"], second.metadata["random_basis_sha256"]
        )
        self.assertTrue(
            torch.equal(
                first.transforms["random_basis_zca"],
                second.transforms["random_basis_zca"],
            )
        )

    def test_higher_quantile_with_strict_admission_controls_rate(self) -> None:
        scores = torch.arange(180, dtype=torch.float64)
        threshold = calibration_threshold(scores)
        admitted = admission_mask(scores, threshold)
        self.assertLessEqual(float(admitted.to(torch.float64).mean().item()), 0.05)
        self.assertEqual(
            threshold,
            float(np.quantile(scores.numpy(), 0.95, method="higher")),
        )
        tied = torch.full((180,), 0.5, dtype=torch.float64)
        self.assertEqual(
            int(admission_mask(tied, calibration_threshold(tied)).sum()), 0
        )

    def test_bootstrap_point_matches_arithmetic_intent_mean(self) -> None:
        candidate = {"target": {"per_intent": {}}}
        comparator = {"target": {"per_intent": {}}}
        differences = []
        for index in range(9):
            label = f"intent-{index}"
            candidate_value = index / 10.0
            comparator_value = (index % 3) / 20.0
            differences.append(candidate_value - comparator_value)
            candidate["target"]["per_intent"][label] = {
                "correct_admission_rate": candidate_value
            }
            comparator["target"]["per_intent"][label] = {
                "correct_admission_rate": comparator_value
            }
        result = paired_intent_bootstrap(candidate, comparator)
        self.assertEqual(result["point"], float(np.mean(differences)))
        self.assertEqual(result["resamples"], 10_000)

    def test_verdict_requires_every_binding_bar(self) -> None:
        per_intent = {
            f"intent-{index}": {"correct_admission_rate": 0.4} for index in range(9)
        }
        candidate = {
            "target": {
                "macro_correct_admission_rate": 0.4,
                "macro_coverage": 0.5,
                "selective_accuracy": 0.8,
                "maximum_predicted_intent_share": 0.2,
                "macro_top1_accuracy": 0.6,
                "per_intent": per_intent,
            },
            "off_support_audit": {"false_admission_rate": 0.04},
        }
        control = copy.deepcopy(candidate)
        control["target"]["macro_top1_accuracy"] = 0.59
        arms = {
            "blind_zca": candidate,
            "raw": control,
            "diagonal_std": copy.deepcopy(control),
            "random_basis_zca": copy.deepcopy(control),
        }
        comparisons = {
            name: {"point": 0.05, "interval": [0.01, 0.09]}
            for name in ("raw", "diagonal_std", "random_basis_zca")
        }
        verdict, bars = development_verdict(arms, comparisons)
        self.assertEqual(verdict, "PASS_TO_TEST")
        self.assertTrue(all(bars.values()))

        failed = copy.deepcopy(arms)
        failed["blind_zca"]["off_support_audit"]["false_admission_rate"] = 0.051
        verdict, bars = development_verdict(failed, comparisons)
        self.assertEqual(verdict, "STOP_BEFORE_TEST")
        self.assertFalse(bars["candidate_off_support_false_admission_rate"])


if __name__ == "__main__":
    unittest.main()
