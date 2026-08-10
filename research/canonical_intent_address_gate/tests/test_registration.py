"""CPU contract tests for CA-1 registration and source isolation."""

from __future__ import annotations

import unittest

from research.canonical_intent_address_gate.contract import (
    ALLOWED_GROUPS,
    EXPECTED_MINILM_TREE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    HLM_ADDRESS_CACHE_PATH,
    IMPLEMENTATION_PATHS,
    MINILM_CACHE_PATH,
    MINILM_SNAPSHOT_PATH,
    PROTOCOL_PATH,
    SOURCE_QUERY_POPULATIONS,
    directory_tree_record,
    sha256_file,
)
from research.canonical_intent_address_gate.run_source_screen import (
    protocol_commit_is_ancestor,
    scientific_constants,
    source_labels,
)
from research.two_axis_admission_gate.contract import DEFAULT_DATA_ROOT
from research.two_axis_admission_gate.prepare_hwu64 import build_populations


class RegistrationTests(unittest.TestCase):
    def test_protocol_is_immutable_and_ancestral(self) -> None:
        self.assertEqual(sha256_file(PROTOCOL_PATH), EXPECTED_PROTOCOL_SHA256)
        self.assertTrue(protocol_commit_is_ancestor())

    def test_registered_implementation_is_complete(self) -> None:
        self.assertTrue(all(path.is_file() for path in IMPLEMENTATION_PATHS))
        constants = scientific_constants()
        self.assertEqual(constants["allowed_groups"], list(ALLOWED_GROUPS))
        self.assertEqual(
            constants["source_query_populations"], list(SOURCE_QUERY_POPULATIONS)
        )
        self.assertFalse(constants["target_names_encoded"])
        self.assertFalse(constants["target_utterances_encoded"])

    def test_canonical_caches_are_separate_from_each_other(self) -> None:
        self.assertNotEqual(HLM_ADDRESS_CACHE_PATH, MINILM_CACHE_PATH)

    def test_source_labels_exclude_target_group(self) -> None:
        built = build_populations(DEFAULT_DATA_ROOT)
        labels = source_labels(built)
        source = set(labels["calibration"]) | set(labels["audit"])
        target = set(built["metadata"]["groups"]["target_supported"])
        self.assertEqual(len(source), 24)
        self.assertFalse(source & target)

    def test_minilm_snapshot_is_exactly_pinned(self) -> None:
        record = directory_tree_record(MINILM_SNAPSHOT_PATH)
        self.assertEqual(record["tree_sha256"], EXPECTED_MINILM_TREE_SHA256)


if __name__ == "__main__":
    unittest.main()
