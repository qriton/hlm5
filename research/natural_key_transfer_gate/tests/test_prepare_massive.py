"""Outcome-blind population-construction tests for NKT-1."""

from __future__ import annotations

import collections
import unittest

from research.natural_key_transfer_gate.prepare_massive import (
    clean_evaluation,
    clean_train,
    normalize_text,
    rank_domains,
    shuffled_label_assignment,
    support_sort_key,
)


def row(
    index: int,
    text: str,
    intent: str,
    partition: str = "train",
    scenario: str = "domain",
) -> dict:
    return {
        "id": str(index),
        "partition": partition,
        "scenario": scenario,
        "intent": intent,
        "utterance": text,
        "normalized_text": normalize_text(text),
        "source_index": index,
    }


class PrepareMassiveTests(unittest.TestCase):
    def test_normalization_is_registered(self) -> None:
        self.assertEqual(normalize_text("  A\u00a0  B  "), "a b")
        self.assertEqual(normalize_text("\uff21"), "a")

    def test_train_drops_conflicts_and_later_duplicates(self) -> None:
        rows = [
            row(0, "same", "a"),
            row(1, " SAME ", "a"),
            row(2, "conflict", "a"),
            row(3, "CONFLICT", "b"),
            row(4, "unique", "b"),
        ]
        retained, conflicts, duplicates = clean_train(rows)
        self.assertEqual([item["source_index"] for item in retained], [0, 4])
        self.assertEqual([item["source_index"] for item in conflicts], [2, 3])
        self.assertEqual([item["source_index"] for item in duplicates], [1])

    def test_evaluation_blocks_train_and_deduplicates(self) -> None:
        rows = [
            row(0, "train text", "a", "train"),
            row(1, "train text", "a", "dev"),
            row(2, "new", "b", "dev"),
            row(3, " NEW ", "b", "dev"),
        ]
        retained, overlaps, duplicates = clean_evaluation(
            rows, "dev", {normalize_text("train text")}
        )
        self.assertEqual([item["source_index"] for item in retained], [2])
        self.assertEqual([item["source_index"] for item in overlaps], [1])
        self.assertEqual([item["source_index"] for item in duplicates], [3])

    def test_domain_and_support_order_are_content_deterministic(self) -> None:
        domains = {"z", "a", "m"}
        self.assertEqual(
            rank_domains(domains), rank_domains(set(reversed(list(domains))))
        )
        rows = [row(3, "third", "a"), row(1, "first", "a"), row(2, "second", "a")]
        first = sorted(rows, key=support_sort_key)
        second = sorted(reversed(rows), key=support_sort_key)
        self.assertEqual(
            [item["source_index"] for item in first],
            [item["source_index"] for item in second],
        )

    def test_shuffle_preserves_label_multiset(self) -> None:
        rows = [row(index, f"text {index}", f"c{index % 3}") for index in range(12)]
        shuffled = shuffled_label_assignment(rows)
        self.assertEqual(
            collections.Counter(shuffled),
            collections.Counter(item["intent"] for item in rows),
        )
        self.assertEqual(shuffled, shuffled_label_assignment(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
