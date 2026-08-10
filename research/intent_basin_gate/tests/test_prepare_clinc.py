"""Outcome-blind manifest tests for the pinned CLINC population."""

from __future__ import annotations

import ast
import unittest
from collections import Counter

from contract import (
    DEVELOPMENT_MANIFEST_PATH,
    SOURCE_COMMIT,
    SOURCE_DATA_SHA256,
    TEST_MANIFEST_PATH,
)
from prepare_clinc import exclude_train_overlaps, load_source
from run_intent_basin_gate import load_json, manifest_errors


class ClincManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.development = load_json(DEVELOPMENT_MANIFEST_PATH)
        cls.test = load_json(TEST_MANIFEST_PATH)

    def test_manifests_match_frozen_population(self) -> None:
        self.assertEqual(manifest_errors(self.development, "development"), [])
        self.assertEqual(manifest_errors(self.test, "test"), [])
        self.assertEqual(self.development["counts"]["in_scope"], 2997)
        self.assertEqual(self.test["counts"]["in_scope"], 4498)

    def test_exact_overlap_exclusions_are_recorded(self) -> None:
        development_indices = [row["source_index"] for row in self.development["exclusions"]]
        test_indices = [row["source_index"] for row in self.test["exclusions"]]
        self.assertEqual(development_indices, [1011, 1794, 2369])
        self.assertEqual(test_indices, [599, 938])
        train_texts = {row["text"] for row in self.development["train"]}
        self.assertFalse(train_texts.intersection(row["text"] for row in self.development["in_scope"]))
        self.assertFalse(train_texts.intersection(row["text"] for row in self.test["in_scope"]))

    def test_training_support_is_exactly_balanced(self) -> None:
        counts = Counter(row["target_index"] for row in self.development["train"])
        self.assertEqual(len(counts), 150)
        self.assertEqual(set(counts.values()), {100})

    def test_manifest_source_provenance_is_pinned(self) -> None:
        for manifest in (self.development, self.test):
            self.assertEqual(manifest["source"]["commit"], SOURCE_COMMIT)
            self.assertEqual(manifest["source"]["data_sha256"], SOURCE_DATA_SHA256)
            self.assertFalse(manifest["construction"]["encoder_loaded"])
            self.assertFalse(manifest["construction"]["routing_outcomes_computed"])

    def test_builder_has_no_encoder_import(self) -> None:
        source = (DEVELOPMENT_MANIFEST_PATH.parent.parent / "prepare_clinc.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(
            imported.intersection({"sentence_transformers", "transformers", "torch"})
        )

    def test_overlap_filter_preserves_official_indices(self) -> None:
        rows = [["keep", "a"], ["duplicate", "b"], ["also keep", "a"]]
        kept, excluded = exclude_train_overlaps(rows, "val", {"duplicate"})
        self.assertEqual([index for index, _ in kept], [0, 2])
        self.assertEqual(excluded[0]["source_index"], 1)

    def test_load_source_rejects_wrong_split_contract(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_source(DEVELOPMENT_MANIFEST_PATH.parent / "missing.json")


if __name__ == "__main__":
    unittest.main()
