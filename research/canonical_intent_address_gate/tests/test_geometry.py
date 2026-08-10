"""CPU tests for canonical-address routing and admission."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from research.canonical_intent_address_gate.contract import (
    EXPECTED_PERMUTATION,
    MINILM_ARM,
    PRIMARY_ARM,
    SHUFFLED_ARM,
)
from research.canonical_intent_address_gate.geometry import (
    canonical_text,
    evaluate_arm,
    hlm_route,
    minilm_route,
    registered_permutation,
    route_from_scores,
    source_verdict,
)


class GeometryTests(unittest.TestCase):
    def test_canonical_text_is_exact_and_rejects_variants(self) -> None:
        self.assertEqual(canonical_text("card_payment_fee_charged"),
                         "intent: card payment fee charged")
        for invalid in ("", " Upper", "UPPER", "has space", "trailing_"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                canonical_text(invalid)

    def test_registered_permutation_is_exact_and_nonidentity(self) -> None:
        permutation = registered_permutation()
        self.assertEqual(tuple(permutation.tolist()), EXPECTED_PERMUTATION)
        self.assertNotEqual(tuple(permutation.tolist()), tuple(range(12)))
        self.assertEqual(sorted(permutation.tolist()), list(range(12)))

    def test_hlm_reader_matches_independent_degree_five_numpy(self) -> None:
        query = torch.tensor(
            [[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]], dtype=torch.float64
        )
        keys = torch.tensor([[4.0, 0.0], [0.0, 5.0]], dtype=torch.float64)
        routed = hlm_route(query, keys, ["a", "b"])
        self.assertEqual(routed["predicted"], ["a", "b", "a"])
        self.assertEqual(routed["maximum_scores"].tolist(), [1.0, 1.0, 0.0])
        self.assertLessEqual(routed["independent_max_abs_error"], 1e-10)

    def test_minilm_reader_matches_independent_cosine_numpy(self) -> None:
        query = torch.tensor([[3.0, 4.0], [1.0, -1.0]], dtype=torch.float64)
        keys = torch.eye(2, dtype=torch.float64)
        routed = minilm_route(query, keys, ["a", "b"])
        self.assertEqual(routed["predicted"], ["b", "a"])
        self.assertLessEqual(routed["independent_max_abs_error"], 1e-10)

    def test_independent_score_disagreement_fails_closed(self) -> None:
        scores = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
        with self.assertRaises(AssertionError):
            route_from_scores(scores, ["a", "b"], np.array([[0.8, 0.2]]))

    def test_higher_threshold_and_strict_admission_bound_calibration(self) -> None:
        labels = ["a", "b"]
        keys = torch.eye(2, dtype=torch.float64)
        negative_queries = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
            dtype=torch.float64,
        )
        positive_queries = torch.eye(2, dtype=torch.float64)
        positive = hlm_route(positive_queries, keys, labels)
        negative = hlm_route(negative_queries, keys, labels)
        metrics, arrays = evaluate_arm(
            positive,
            labels,
            negative,
            positive,
            labels,
            negative,
        )
        self.assertEqual(metrics["threshold"]["negative_admitted_count"], 0)
        self.assertEqual(metrics["threshold"]["comparison"], "strict_greater_than")
        self.assertFalse(arrays["calibration_off_support"]["admitted"].any())

    def test_source_verdict_requires_every_primary_bar(self) -> None:
        supported = {
            "macro_top1_accuracy": 0.50,
            "macro_coverage": 0.25,
            "macro_correct_admission_rate": 0.20,
            "selective_accuracy": 0.80,
            "maximum_predicted_intent_share": 0.20,
        }
        arms = {
            PRIMARY_ARM: {
                "audit_supported": supported,
                "audit_off_support": {"false_admission_rate": 0.04},
            },
            SHUFFLED_ARM: {
                "audit_supported": {
                    "macro_top1_accuracy": 0.20,
                    "macro_correct_admission_rate": 0.05,
                }
            },
            MINILM_ARM: {"audit_supported": {"macro_top1_accuracy": 0.55}},
        }
        verdict, bars, _ = source_verdict(arms)
        self.assertEqual(verdict, "SOURCE_SCREEN_LIVE")
        self.assertTrue(all(bars.values()))
        arms[PRIMARY_ARM]["audit_off_support"]["false_admission_rate"] = 0.06
        verdict, bars, _ = source_verdict(arms)
        self.assertEqual(verdict, "CLOSED_SOURCE_ONLY")
        self.assertFalse(bars["off_support_false_admission_rate"])


if __name__ == "__main__":
    unittest.main()
