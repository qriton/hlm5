# B77-E1 Development Verdict — Task-Aligned Energy Decomposition

**Date:** 2026-08-10

**Formal verdict:** `HARNESS_INVALID`

**Validity:** FAIL

## Result

The calibration-selected shrinkage was `0.05` and the
selected unaligned control was `raw_centroid`.

| Arm | Development macro accuracy |
|---|---:|
| Selected unaligned | 0.844155844156 |
| Aligned static | 0.888311688312 |
| One step | 0.888311688312 |
| Eight-step candidate | 0.888311688312 |
| Shuffled-label static | 0.024675324675 |

Alignment gain: `0.044155844155844164` with 95%
intent-cluster interval `[0.014285714285714292,
0.07792207792207792]`.

Candidate minus aligned static: `0.0`
with interval `[0.0,
0.0]`.

Candidate minus one step: `0.0`
with interval `[0.0,
0.0]`.

## Gates

```json
{
  "alignment_bars": {
    "interval_lower": true,
    "macro_gain": true,
    "maximum_intent_share": true
  },
  "dynamics_bars": {
    "maximum_intent_share": true,
    "one_step_interval_lower": false,
    "one_step_macro_gain": false,
    "static_interval_lower": false,
    "static_macro_gain": false
  },
  "validity": {
    "calibration_selection": true,
    "cap": true,
    "energy_and_armijo": true,
    "finite_and_positive_covariance": true,
    "identity": true,
    "independent_matched_optimizer": false,
    "population": true,
    "registration_and_hashes": true,
    "shuffled_label_control": true,
    "test_sealed": true,
    "trust_radius": true
  }
}
```

The official Banking77 test split was not encoded during this development run.
Only a claim named by the formal verdict may be registered for test. This
result does not establish an HLM-specific advantage, explainability,
compliance, deployment readiness, or a reason to scale.
