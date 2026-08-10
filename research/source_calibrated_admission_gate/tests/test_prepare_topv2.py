"""CPU tests for deterministic TOPv2 population preparation."""

from __future__ import annotations

import unittest

from research.source_calibrated_admission_gate.prepare_topv2 import (
    assign_source_intent_groups,
    clean_partition,
    normalize_text,
    parse_root_intent,
    select_rows,
)


def row(
    row_id: str,
    normalized: str,
    intent: str,
    partition: str = "train",
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "normalized": normalized,
        "intent": intent,
        "partition": partition,
    }


class PrepareTopv2Tests(unittest.TestCase):
    def test_normalization_and_root_intent_are_explicit(self) -> None:
        self.assertEqual(normalize_text("  Café\tALARM  "), "café alarm")
        self.assertEqual(
            parse_root_intent("[IN:CREATE_TIMER [SL:DATE_TIME ten minutes ] ]"),
            "CREATE_TIMER",
        )
        with self.assertRaises(ValueError):
            parse_root_intent("not a TOP parse")

    def test_cleanup_drops_conflicts_duplicates_and_overlap(self) -> None:
        rows = [
            row("b", "same", "alarm::A"),
            row("a", "same", "alarm::A"),
            row("c", "conflict", "alarm::A"),
            row("d", "conflict", "alarm::B"),
            row("e", "blocked", "alarm::A"),
            row("f", "keep", "alarm::B"),
        ]
        clean, diagnostics = clean_partition(rows, {"blocked"})
        self.assertEqual([item["row_id"] for item in clean], ["a", "f"])
        self.assertEqual(diagnostics["duplicate_rows_dropped"], 1)
        self.assertEqual(diagnostics["conflict_rows_dropped"], 2)
        self.assertEqual(diagnostics["overlap_rows_dropped"], 1)

    def test_source_groups_are_disjoint_and_content_deterministic(self) -> None:
        intents = [f"domain::INTENT_{index:02d}" for index in range(42)]
        forward = assign_source_intent_groups(intents)
        reverse = assign_source_intent_groups(reversed(intents))
        self.assertEqual(forward, reverse)
        selected = (
            forward["calibration_supported"]
            + forward["calibration_negative"]
            + forward["off_support_audit"]
        )
        self.assertEqual(len(selected), 27)
        self.assertEqual(len(set(selected)), 27)
        self.assertEqual(len(forward["unused"]), 15)

    def test_row_selection_is_salted_not_input_ordered(self) -> None:
        rows = [row(str(index), f"u{index}", "alarm::A") for index in range(10)]
        first = select_rows(rows, 3, "fixed-salt")
        second = select_rows(reversed(rows), 3, "fixed-salt")
        self.assertEqual(
            [item["row_id"] for item in first],
            [item["row_id"] for item in second],
        )
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
