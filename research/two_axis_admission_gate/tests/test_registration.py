"""CPU contract tests for SCA-2 registration and stage isolation."""

from __future__ import annotations

import unittest

import torch

from research.natural_key_transfer_gate.run_natural_key_transfer_gate import (
    snapshot_reader_parity,
)
from research.two_axis_admission_gate.contract import (
    ARM_NAMES,
    EXPECTED_PREREG_SHA256,
    EXPECTED_RANDOM_BASIS_SHA256,
    IMPLEMENTATION_PATHS,
    PREREG_PATH,
    TEST_FOLD,
    sha256_file,
)
from research.two_axis_admission_gate.geometry import registered_random_basis
from research.two_axis_admission_gate.run_two_axis_admission_gate import (
    SOURCE_HIDDEN_POPULATION_NAMES,
    TARGET_HIDDEN_POPULATION_NAMES,
    HarnessInvalid,
    _calibration_row_records,
    hidden_population_rows,
    prereg_commit_is_ancestor,
    registered_execution_contract,
    scientific_constants,
)


def synthetic_row(row_id: str, fold: int = 3) -> dict[str, object]:
    return {
        "row_id": row_id,
        "fold": fold,
        "utterance": f"utterance {row_id}",
    }


class RegistrationTests(unittest.TestCase):
    def test_preregistration_is_immutable_and_ancestral(self) -> None:
        self.assertEqual(sha256_file(PREREG_PATH), EXPECTED_PREREG_SHA256)
        self.assertTrue(prereg_commit_is_ancestor())

    def test_registered_implementation_is_complete(self) -> None:
        self.assertTrue(all(path.is_file() for path in IMPLEMENTATION_PATHS))
        constants = scientific_constants()
        self.assertEqual(constants["arms"], list(ARM_NAMES))
        self.assertTrue(constants["source_audit_precedes_target_encoding"])
        self.assertTrue(constants["no_test_encoding"])

    def test_execution_contract_pins_two_separate_caches(self) -> None:
        contract = registered_execution_contract()
        self.assertEqual(contract["mode"], "development")
        self.assertEqual(contract["device"], "cuda")
        self.assertNotEqual(contract["source_cache"], contract["target_cache"])
        self.assertEqual(
            set(contract),
            {
                "mode",
                "data_root",
                "checkpoint",
                "tokenizer",
                "source_cache",
                "target_cache",
                "device",
            },
        )

    def test_hidden_population_rejects_test_fold(self) -> None:
        names = SOURCE_HIDDEN_POPULATION_NAMES
        populations = {
            name: [synthetic_row(f"{name}-row")] for name in names
        }
        rows = hidden_population_rows(populations, names)
        self.assertEqual(len(rows), len(names))
        populations["audit_negative"] = [synthetic_row("test-row", TEST_FOLD)]
        with self.assertRaises(HarnessInvalid):
            hidden_population_rows(populations, names)

    def test_source_and_target_population_names_are_disjoint(self) -> None:
        self.assertFalse(
            set(SOURCE_HIDDEN_POPULATION_NAMES) & set(TARGET_HIDDEN_POPULATION_NAMES)
        )

    def test_random_basis_is_exactly_registered(self) -> None:
        _, metadata = registered_random_basis(768)
        self.assertEqual(metadata["basis_sha256"], EXPECTED_RANDOM_BASIS_SHA256)

    def test_snapshot_reader_matches_degree_five_reader(self) -> None:
        support = torch.eye(4, dtype=torch.float64)
        query = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]],
            dtype=torch.float64,
        )
        parity = snapshot_reader_parity(query, support)
        self.assertTrue(parity["prediction_match"])
        self.assertEqual(parity["prediction_or_abstention_mismatch_count"], 0)

    def test_insufficient_calibration_preserves_rows_with_null_dual(self) -> None:
        populations = {
            "calibration_positive": [
                {
                    **synthetic_row("positive", 1),
                    "source_index": 1,
                    "utterance_sha256": "positive-hash",
                    "intent": "truth",
                }
            ],
            "calibration_negative": [
                {
                    **synthetic_row("negative", 1),
                    "source_index": 2,
                    "utterance_sha256": "negative-hash",
                    "intent": "other",
                }
            ],
        }
        science = {
            "calibration": {
                arm: {
                    "thresholds": {"tau_absolute": 0.4, "tau_margin": None}
                }
                for arm in ARM_NAMES
            }
        }
        arrays = {
            "calibration": {
                arm: {
                    "positive_route": {
                        "predicted": ["truth"],
                        "maximum_scores": torch.tensor([0.5]),
                        "class_margins": torch.tensor([0.2]),
                    },
                    "negative_route": {
                        "predicted": ["truth"],
                        "maximum_scores": torch.tensor([0.3]),
                        "class_margins": torch.tensor([0.1]),
                    },
                }
                for arm in ARM_NAMES
            }
        }
        records = _calibration_row_records(populations, science, arrays)
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0]["arms"]["blind_zca"]["admission"]["absolute_only"])
        self.assertIsNone(records[0]["arms"]["blind_zca"]["admission"]["dual"])
        self.assertFalse(records[1]["arms"]["blind_zca"]["admission"]["absolute_only"])


if __name__ == "__main__":
    unittest.main()
