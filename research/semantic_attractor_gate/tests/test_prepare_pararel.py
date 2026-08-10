"""Tests for the outcome-blind ParaRel manifest builder."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_pararel import build_manifest, render, selected_unique_facts  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


class ParaRelManifestTests(unittest.TestCase):
    def test_render_requires_exact_placeholders(self) -> None:
        self.assertEqual(
            render("[X] was born in [Y].", "Ada", "something"),
            "Ada was born in something.",
        )
        with self.assertRaises(ValueError):
            render("[X] has no object.", "Ada", "something")

    def test_fact_selection_is_sorted_and_subject_unique(self) -> None:
        rows = [
            {"uuid": "b", "sub_label": "same", "obj_label": "two"},
            {"uuid": "a", "sub_label": "same", "obj_label": "one"},
            {"uuid": "c", "sub_label": "other", "obj_label": "three"},
        ]
        selected = selected_unique_facts(rows, 2)
        self.assertEqual([row["uuid"] for row in selected], ["a", "c"])

    def test_builder_never_loads_a_model_and_constructs_disjoint_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(source), "config", "user.name", "Test"],
                check=True,
            )
            pattern_dir = source / "data" / "pattern_data" / "graphs_json"
            fact_dir = source / "data" / "trex_lms_vocab"
            for relation, subjects in (
                ("P1", ("Ada", "Grace")),
                ("P2", ("Paris", "Berlin")),
            ):
                write_jsonl(
                    pattern_dir / f"{relation}.jsonl",
                    [
                        {"pattern": "[X] links to [Y]."},
                        {"pattern": "[Y] is linked by [X]."},
                    ],
                )
                write_jsonl(
                    fact_dir / f"{relation}.jsonl",
                    [
                        {"uuid": f"{relation}-0", "sub_label": subjects[0], "obj_label": "A"},
                        {"uuid": f"{relation}-1", "sub_label": subjects[1], "obj_label": "B"},
                    ],
                )
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(source), "commit", "-q", "-m", "fixture"],
                check=True,
            )

            manifest = build_manifest(
                source,
                facts_per_relation=2,
                mask_literal="something",
            )
            self.assertEqual(manifest["counts"]["eligible_relations"], 2)
            self.assertEqual(manifest["counts"]["stored_addresses"], 4)
            self.assertEqual(manifest["counts"]["supported_queries"], 4)
            stored_pairs = {
                (slot["relation"], slot["subject"])
                for slot in manifest["slots"]
            }
            for query in manifest["queries"]:
                self.assertNotIn(
                    (query["relation"], query["off_support_subject"]),
                    stored_pairs,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
