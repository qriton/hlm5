"""CPU tests for the HLM5 exact-address lifecycle runnable reference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demos"))

from exact_address_lifecycle_reference import (  # noqa: E402
    PINNED_E5V3_SCIENTIFIC_SHA256,
    parse_args,
    run_reference,
    verify_pinned_e5v3,
)


class ExactAddressLifecycleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_reference()

    def test_pinned_e5v3_is_the_cache_equivalent_verdict(self) -> None:
        pinned = verify_pinned_e5v3()
        self.assertEqual(pinned["formal_verdict"], "CANONICAL_CACHE_EQUIVALENT")
        self.assertEqual(
            pinned["scientific_sha256"],
            PINNED_E5V3_SCIENTIFIC_SHA256,
        )

    def test_full_lifecycle_passes(self) -> None:
        self.assertEqual(self.report["status"], "CACHE_EQUIVALENT_REFERENCE")
        self.assertTrue(all(self.report["lifecycle"].values()))
        self.assertTrue(self.report["bundle"]["restored_state_exact"])

    def test_every_phase_is_exactly_matched_by_cache(self) -> None:
        self.assertTrue(self.report["all_phases_degree5_cache_bit_exact"])
        self.assertTrue(
            all(phase["degree5_cache_bit_exact"] for phase in self.report["phases"])
        )

    def test_deactivate_delete_and_off_support_are_gate_off(self) -> None:
        phases = {phase["phase"]: phase for phase in self.report["phases"]}
        self.assertFalse(phases["deactivate"]["routed"][0])
        self.assertTrue(phases["reactivate"]["routed"][0])
        self.assertFalse(phases["hard_delete"]["routed"][2])
        self.assertFalse(any(phases["off_support"]["routed"]))

    def test_claim_boundary_is_explicit(self) -> None:
        boundary = self.report["claim_boundary"]
        self.assertIn("No semantic routing", boundary)
        self.assertIn("HLM-specific retrieval advantage", boundary)
        self.assertIn("scaling authorization", boundary)

    def test_output_cannot_collide_with_pinned_e5v3_evidence(self) -> None:
        pinned_result = ROOT / "baselines" / "results" / "address_conditioning_v1.json"
        with mock.patch.object(
            sys,
            "argv",
            ["reference", "--out", str(pinned_result)],
        ):
            with mock.patch.object(sys, "stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main(verbosity=2)
