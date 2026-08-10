"""CPU tests for the registered NKT-1 geometry."""

from __future__ import annotations

import unittest

import torch

from research.natural_key_transfer_gate.geometry import (
    binary_auc,
    development_verdict,
    fit_key_operators,
    paired_intent_bootstrap,
    route_scores,
)


class GeometryTests(unittest.TestCase):
    def test_inverse_roots_and_label_null_are_well_formed(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(17)
        values = torch.randn(24, 6, generator=generator, dtype=torch.float64)
        labels = [f"c{index // 6}" for index in range(24)]
        shuffled = labels[::2] + labels[1::2]
        operators = fit_key_operators(values, labels, shuffled)
        self.assertEqual(tuple(operators.mean.shape), (6,))
        for arm in ("blind_zca", "shuffled_within", "source_fisher"):
            self.assertLess(
                operators.metadata[arm]["inverse_identity_max_abs_error"],
                1e-8,
            )

    def test_degree_five_matches_positive_cosine_and_numpy(self) -> None:
        supports = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64
        )
        queries = torch.tensor(
            [[0.8, 0.2], [0.1, 0.9], [-0.7, 0.1]], dtype=torch.float64
        )
        output = route_scores(queries, supports)
        self.assertEqual(output["slots"].tolist(), [0, 1, 2])
        self.assertLessEqual(output["independent_max_abs_error"], 1e-10)

    def test_binary_auc_handles_ties(self) -> None:
        positive = torch.tensor([1.0, 2.0], dtype=torch.float64)
        negative = torch.tensor([0.0, 1.0], dtype=torch.float64)
        self.assertAlmostEqual(binary_auc(positive, negative), 0.875)

    def test_bootstrap_uses_the_point_mean(self) -> None:
        candidate = {"a": 1.0, "b": 0.5, "c": 0.0}
        comparator = {"a": 0.0, "b": 0.5, "c": 1.0}
        result = paired_intent_bootstrap(candidate, comparator)
        self.assertEqual(result["point"], 0.0)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["tied"], 1)
        self.assertEqual(result["worsened"], 1)

    def test_verdict_requires_every_bar(self) -> None:
        arms = {
            "source_fisher": {
                "maximum_predicted_intent_share": 0.08,
                "supported_off_support_auroc": 0.80,
            },
            "raw": {"supported_off_support_auroc": 0.80},
        }
        comparisons = {
            "raw": {"point": 0.03, "interval": [0.01, 0.05]},
            "blind_zca": {"point": 0.02, "interval": [0.005, 0.04]},
            "shuffled_within": {"point": 0.02, "interval": [0.005, 0.04]},
        }
        verdict, bars = development_verdict(arms, comparisons)
        self.assertEqual(verdict, "PASS_TO_TEST")
        self.assertTrue(all(bars.values()))
        comparisons["blind_zca"]["interval"][0] = 0.0
        verdict, _ = development_verdict(arms, comparisons)
        self.assertEqual(verdict, "STOP_BEFORE_TEST")


if __name__ == "__main__":
    unittest.main()
