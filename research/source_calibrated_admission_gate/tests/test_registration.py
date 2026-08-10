"""CPU contract tests for SCA-1 registration and reader parity."""

from __future__ import annotations

import unittest

import torch

from research.source_calibrated_admission_gate.contract import (
    ARM_NAMES,
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_QUANTILE_METHOD,
    IMPLEMENTATION_PATHS,
)
from research.source_calibrated_admission_gate.run_source_calibrated_admission_gate import (
    HIDDEN_POPULATION_NAMES,
    HarnessInvalid,
    hidden_population_rows,
    registered_execution_contract,
    scientific_constants,
    snapshot_reader_parity,
)


def synthetic_row(row_id: str, partition: str = "train") -> dict[str, object]:
    return {
        "row_id": row_id,
        "partition": partition,
        "utterance": f"utterance {row_id}",
    }


class RegistrationTests(unittest.TestCase):
    def test_registered_implementation_is_complete(self) -> None:
        self.assertTrue(all(path.is_file() for path in IMPLEMENTATION_PATHS))
        constants = scientific_constants()
        self.assertEqual(constants["arms"], list(ARM_NAMES))
        self.assertEqual(constants["bootstrap_resamples"], BOOTSTRAP_RESAMPLES)
        self.assertEqual(
            constants["calibration_quantile_method"],
            CALIBRATION_QUANTILE_METHOD,
        )
        self.assertTrue(constants["no_test_encoding"])

    def test_execution_contract_is_closed_to_registered_paths(self) -> None:
        contract = registered_execution_contract()
        self.assertEqual(contract["mode"], "development")
        self.assertEqual(contract["device"], "cuda")
        self.assertEqual(
            set(contract),
            {"mode", "data_root", "checkpoint", "tokenizer", "cache", "device"},
        )

    def test_hidden_population_rejects_test_rows(self) -> None:
        populations = {
            name: [synthetic_row(f"{name}-row")] for name in HIDDEN_POPULATION_NAMES
        }
        rows = hidden_population_rows(populations)
        self.assertEqual(len(rows), len(HIDDEN_POPULATION_NAMES))
        populations["target_development"] = [synthetic_row("test-row", "test")]
        with self.assertRaises(HarnessInvalid):
            hidden_population_rows(populations)

    def test_snapshot_reader_matches_degree_five_reader(self) -> None:
        support = torch.eye(4, dtype=torch.float64)
        query = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]],
            dtype=torch.float64,
        )
        parity = snapshot_reader_parity(query, support)
        self.assertTrue(parity["prediction_match"])
        self.assertEqual(parity["prediction_or_abstention_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
