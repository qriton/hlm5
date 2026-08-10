"""Registration and phase-boundary tests for the I1 gate."""

from __future__ import annotations

import copy
import subprocess
import sys
import unittest

from contract import (
    DEVELOPMENT_MANIFEST_PATH,
    HASHED_IMPLEMENTATION_PATHS,
    ROOT,
    SCIENTIFIC_CONSTANTS,
    TEST_MANIFEST_PATH,
)
from metrics import classify_development_verdict
from run_intent_basin_gate import (
    canonical_json_sha256,
    cross_manifest_errors,
    load_json,
    manifest_errors,
    registration_digest_matches,
)


class RegistrationContractTests(unittest.TestCase):
    def test_all_registered_implementation_paths_exist(self) -> None:
        missing = [str(path) for path in HASHED_IMPLEMENTATION_PATHS if not path.is_file()]
        self.assertEqual(missing, [])

    def test_manifests_are_valid_before_registration(self) -> None:
        development = load_json(DEVELOPMENT_MANIFEST_PATH)
        test = load_json(TEST_MANIFEST_PATH)
        self.assertEqual(manifest_errors(development, "development"), [])
        self.assertEqual(manifest_errors(test, "test"), [])
        self.assertEqual(cross_manifest_errors(development, test), [])

    def test_development_manifest_contains_no_test_split_rows(self) -> None:
        manifest = load_json(DEVELOPMENT_MANIFEST_PATH)
        encoded_splits = {row["split"] for row in manifest["train"]}
        encoded_splits.update(row["split"] for row in manifest["in_scope"])
        encoded_splits.update(row["split"] for row in manifest["oos"])
        self.assertEqual(encoded_splits, {"train", "val", "oos_val"})

    def test_cli_rejects_scientific_overrides_before_execution(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "run_intent_basin_gate.py"),
                "--run-development",
                "--degree",
                "3",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("unrecognized arguments", completed.stderr)

    def test_scientific_digest_detects_any_mutation(self) -> None:
        receipt = {
            "phase": "development",
            "scientific_constants": SCIENTIFIC_CONSTANTS,
        }
        receipt["scientific_digest"] = canonical_json_sha256(receipt)
        self.assertTrue(registration_digest_matches(receipt))
        changed = copy.deepcopy(receipt)
        changed["scientific_constants"]["steps"] = 9
        self.assertFalse(registration_digest_matches(changed))

    def test_failed_development_science_never_authorizes_test(self) -> None:
        verdict = classify_development_verdict(
            {"all_valid": True},
            {"static_gain": True, "one_step_gain": False},
        )
        self.assertEqual(verdict, "STOP_BEFORE_TEST")


if __name__ == "__main__":
    unittest.main()
