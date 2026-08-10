# E7c spent-pool geometry screen

Date: 2026-08-10
Status: development protocol; no fresh-pool evidence is authorized here

## Question

E7b was implementation-valid and replay-exact, but failed its total coverage bar:
893 of 1,200 targets were geometrically reachable, below the frozen minimum of
960. Every reachable target passed the ordinary BF16 head check (893/893).

This screen asks one bounded development question: can a fixed readout-aligned
preconditioner recover the missing geometry without reintroducing a native-head
failure?

## Frozen inputs and scope

The screen may use only the already-spent E7b pool and its pinned model/prompt.
It must bind these artifacts before loading the model:

- E7b admission SHA-256:
  `60249139741d79c3995205bc09c04e0f68067155098abff7d4c1d9d08e828b39`;
- E7b result SHA-256:
  `9a5232b33f8828195794cb137ee2ec83e10ec8af524418abe3608cf0ac2d4d11`;
- E7b verdict SHA-256:
  `5f5e35f71d432d2df0287c7dec1350e8b53c3dae7bea0f025634536a3a6fd016`.

The E7b source tree, SmolLM3-3B-Base revision, prompt, target order, native
runtime switches, exact affine geometry, dose grid, and strict native-head rule
remain unchanged. This is post-outcome mechanism development, not independent
confirmation and not evidence on a new population.

## Development/validation split

For target id `t`, compute SHA-256 of the UTF-8 string
`hlm5-e7c-spent-split-v1:{t}`. An even low bit in the first digest byte assigns
the target to development; an odd low bit assigns it to validation. This gives
583 development and 617 validation targets. Candidate selection reads only the
development geometry: non-selected candidates are never evaluated on validation
targets. Validation geometry and native outcomes are read only after one
candidate is selected.

## Frozen candidate family

Let `E` be the untouched BF16 unembedding converted to float32 for covariance
construction, `mu` its vocabulary-row mean, and
`C = E^T E / V - mu mu^T`, symmetrized before `torch.linalg.eigh`. Let `m` be
the median strictly positive eigenvalue and `r = 0.01 m`. Candidate directions
are normalized in float64 with denominator `norm + 1e-12`.

In fixed tie-break order:

1. `raw_unit`: `e_t`;
2. `centered_unit`: `e_t - mu`;
3. `diag_whiten_centered_r1e2`: `(diag(C)+r)^(-1/2)(e_t-mu)`;
4. `zca_half_raw_r1e2`: `(C+rI)^(-1/2)e_t`;
5. `zca_half_centered_r1e2`: `(C+rI)^(-1/2)(e_t-mu)`;
6. `mahalanobis_raw_r1e2`: `(C+rI)^(-1)e_t`;
7. `mahalanobis_centered_r1e2`: `(C+rI)^(-1)(e_t-mu)`.

No ridge, power, centering choice, prompt, dose cap, or native threshold may be
changed after candidate outcomes are computed.

## Validity and selection

The raw candidate must reproduce every E7b stored geometry field exactly:
target id, interval, dose, float64 margin, class, and blocker id. Any mismatch
makes the screen invalid. The protocol, screen CLI, and targeted test must be
committed and their hashes recorded before execution.

Select the candidate with the largest development reachable count. Ties choose
the earliest candidate in the frozen list. Only that candidate receives the
ordinary BF16 native-head evaluation.

The selected candidate is ready for a separately preregistered fresh-pool E7c
gate only if, on the 617 held-out spent-pool validation targets:

- `5 * geometrically_reachable >= 4 * 617` (at least 80%); and
- `100 * native_admitted >= 99 * geometrically_reachable`.

Report gained and lost reachable targets against raw E7b, all refusal counts,
the covariance/eigenbasis hashes, exact native-logit hashes for the selected
candidate, and the complete selected rows. A passing screen authorizes writing
a fresh-pool preregistration; it does not itself authorize promotion.
