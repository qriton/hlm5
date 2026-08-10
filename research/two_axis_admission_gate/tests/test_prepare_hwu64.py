"""CPU tests for deterministic HWU64 population preparation."""

from __future__ import annotations

import unittest

from research.two_axis_admission_gate.contract import (
    EXPECTED_MANIFEST_SHA256,
    GROUP_INTENT_COUNT,
    GROUP_NAMES,
    MANIFEST_PATH,
    read_json,
    sha256_file,
)
from research.two_axis_admission_gate.prepare_hwu64 import (
    assign_intent_groups,
    clean_partition,
    normalize_text,
    parse_args,
    select_rows,
)


def row(row_id: str, normalized: str, intent: str) -> dict[str, object]:
    return {"row_id": row_id, "normalized": normalized, "intent": intent}


class PrepareHwu64Tests(unittest.TestCase):
    def test_registered_manifest_hashes_are_pinned(self) -> None:
        manifest = read_json(MANIFEST_PATH)
        self.assertEqual(
            sha256_file(MANIFEST_PATH), EXPECTED_MANIFEST_SHA256["file"]
        )
        self.assertEqual(
            manifest["scientific_sha256"],
            EXPECTED_MANIFEST_SHA256["scientific"],
        )

    def test_normalization_is_explicit(self) -> None:
        self.assertEqual(normalize_text("  Café\tALARM  "), "café alarm")
        with self.assertRaises(ValueError):
            normalize_text(" \t ")

    def test_cleanup_drops_conflicts_duplicates_and_overlap(self) -> None:
        rows = [
            row("b", "same", "A"),
            row("a", "same", "A"),
            row("c", "conflict", "A"),
            row("d", "conflict", "B"),
            row("e", "blocked", "A"),
            row("f", "keep", "B"),
        ]
        clean, diagnostics = clean_partition(rows, {"blocked"})
        self.assertEqual([item["row_id"] for item in clean], ["a", "f"])
        self.assertEqual(diagnostics["duplicate_rows_dropped"], 1)
        self.assertEqual(diagnostics["conflict_rows_dropped"], 2)
        self.assertEqual(diagnostics["overlap_rows_dropped"], 1)

    def test_five_groups_are_disjoint_and_deterministic(self) -> None:
        intents = [f"intent_{index:02d}" for index in range(64)]
        forward = assign_intent_groups(intents)
        reverse = assign_intent_groups(reversed(intents))
        self.assertEqual(forward, reverse)
        assigned = [intent for name in GROUP_NAMES for intent in forward[name]]
        self.assertEqual(len(assigned), len(GROUP_NAMES) * GROUP_INTENT_COUNT)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(len(forward["unused_eligible"]), 4)

    def test_row_selection_is_salted_not_input_ordered(self) -> None:
        rows = [row(str(index), f"u{index}", "A") for index in range(10)]
        first = select_rows(rows, 3, "fixed-salt")
        second = select_rows(reversed(rows), 3, "fixed-salt")
        self.assertEqual(
            [item["row_id"] for item in first],
            [item["row_id"] for item in second],
        )

    def test_cli_rejects_unregistered_input_or_output_paths(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--data-root", "D:/not-the-registered-dataset"])
        with self.assertRaises(SystemExit):
            parse_args(["--out", "research/two_axis_admission_gate/README.md"])


if __name__ == "__main__":
    unittest.main()
