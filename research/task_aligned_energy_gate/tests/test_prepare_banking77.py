"""Fast tests for the outcome-blind Banking77 manifest construction."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contract import (  # noqa: E402
    DEVELOPMENT_MANIFEST_PATH,
    SCIENTIFIC_CONSTANTS,
    TEST_MANIFEST_PATH,
)
from prepare_banking77 import (  # noqa: E402
    assign_training_roles,
    deduplicate_training,
    normalize_text,
)


class PrepareBanking77Tests(unittest.TestCase):
    def test_normalization_is_registered_and_non_semantic(self) -> None:
        self.assertEqual(normalize_text("  CAFÉ\tCard  "), "café card")
        self.assertEqual(normalize_text("ＡＢＣ"), "abc")

    def test_deduplication_keeps_lowest_source_index_and_rejects_no_label(self) -> None:
        rows = [
            {
                "row_id": "train|00002",
                "source_index": 2,
                "text": " Hello  ",
                "intent": "a",
            },
            {
                "row_id": "train|00001",
                "source_index": 1,
                "text": "hello",
                "intent": "a",
            },
            {
                "row_id": "train|00003",
                "source_index": 3,
                "text": "world",
                "intent": "b",
            },
        ]
        kept, excluded = deduplicate_training(rows)
        self.assertEqual(
            [row["row_id"] for row in kept], ["train|00001", "train|00003"]
        )
        self.assertEqual(excluded[0]["kept_row_id"], "train|00001")

    def test_assignment_is_deterministic_and_disjoint(self) -> None:
        categories = ["a", "b"]
        rows = []
        for category_index, category in enumerate(categories):
            for index in range(25):
                rows.append(
                    {
                        "row_id": f"train|{category_index * 100 + index:05d}",
                        "source_index": category_index * 100 + index,
                        "text": f"{category} row {index}",
                        "intent": category,
                        "target_index": category_index,
                    }
                )
        roles = assign_training_roles(rows, categories)
        self.assertEqual(len(roles["calibration"]), 20)
        self.assertEqual(len(roles["development"]), 20)
        self.assertEqual(len(roles["fit"]), 10)
        ids = [{row["row_id"] for row in roles[role]} for role in roles]
        self.assertFalse(ids[0] & ids[1])
        self.assertFalse(ids[0] & ids[2])
        self.assertFalse(ids[1] & ids[2])
        self.assertEqual(roles, assign_training_roles(rows, categories))

    def test_checked_in_manifests_match_blind_census(self) -> None:
        development = json.loads(DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
        test = json.loads(TEST_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            development["counts"]["fit"], SCIENTIFIC_CONSTANTS["fit_count"]
        )
        self.assertEqual(
            development["counts"]["development"],
            SCIENTIFIC_CONSTANTS["development_count"],
        )
        self.assertEqual(
            test["counts"]["primary"], SCIENTIFIC_CONSTANTS["primary_test_count"]
        )
        self.assertEqual(test["counts"]["normalized_train_overlap_exclusions"], 7)
        self.assertFalse(development["construction"]["encoder_loaded"])
        self.assertFalse(development["construction"]["model_outcomes_computed"])


if __name__ == "__main__":
    unittest.main()
