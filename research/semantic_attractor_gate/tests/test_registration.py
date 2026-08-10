"""Fail-closed registration and CLI tests for HLM5 S1."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contract import REGISTERED_COMMAND, SCIENTIFIC_CONSTANTS  # noqa: E402
from run_semantic_attractor_gate import (  # noqa: E402
    canonical_json_sha256,
    evaluate_rows,
    load_json,
    manifest_errors,
    registration_digest_matches,
)
from attractor import unit  # noqa: E402


class RegistrationContractTests(unittest.TestCase):
    def test_manifest_matches_frozen_population(self) -> None:
        manifest = load_json(ROOT / "data" / "pararel_manifest.json")
        self.assertEqual(manifest_errors(manifest), [])
        mutated = copy.deepcopy(manifest)
        mutated["counts"]["stored_addresses"] -= 1
        self.assertTrue(manifest_errors(mutated))

    def test_scientific_digest_detects_any_mutation(self) -> None:
        registration = {
            "schema_version": 1,
            "registered_command": REGISTERED_COMMAND,
            "scientific_constants": SCIENTIFIC_CONSTANTS,
        }
        registration["scientific_digest"] = canonical_json_sha256(registration)
        self.assertTrue(registration_digest_matches(registration))
        mutated = copy.deepcopy(registration)
        mutated["scientific_constants"]["degree"] = 3
        self.assertFalse(registration_digest_matches(mutated))

    def test_cli_rejects_scientific_overrides_before_execution(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run_semantic_attractor_gate.py"),
                "--run-registered",
                "--degree",
                "3",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unrecognized arguments", completed.stderr)

    def test_row_evaluator_writes_raw_trajectories_before_aggregation(self) -> None:
        keys = torch.eye(4, dtype=torch.float64)
        queries = unit(
            torch.tensor(
                [[0.9, 0.3, 0.1, 0.0], [0.1, 0.8, 0.2, 0.1]],
                dtype=torch.float64,
            )
        )
        metadata = [
            {"query_id": "q0", "relation": "P1", "target_slot": 0, "subject": "A"},
            {"query_id": "q1", "relation": "P2", "target_slot": 1, "subject": "B"},
        ]
        raw = io.StringIO()
        evaluated = evaluate_rows(
            kind="supported",
            queries=queries,
            metadata=metadata,
            keys=keys,
            raw_handle=raw,
        )
        rows = [json.loads(line) for line in raw.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]["energies"]), 9)
        self.assertEqual(rows[0]["candidate_slot"], rows[0]["matched_slot"])
        self.assertLessEqual(evaluated["maximum_energy_increase"], 1e-12)
        self.assertLessEqual(evaluated["maximum_matched_error"], 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
