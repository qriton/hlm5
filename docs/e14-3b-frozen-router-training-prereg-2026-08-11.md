# HLM5 E14: frozen-3B router-training gate

Date: 2026-08-11

Status: preregistration; no E14 3B hidden, router-training, route, head,
locality, rollback, or matched-control outcome has been computed

Scope: first bounded training rung after E13. HLM-KB, HLM-Flow, Ingesto,
full-trunk fine-tuning, from-scratch 3B pretraining, attention replacement,
semantic truth, and commercial claims are out of scope.

## Decision question

Can a small trainable router attached to the pinned frozen public 3B trunk learn
paraphrase-to-exact-key alignment on one fact set, transfer that alignment to a
disjoint fact set, and preserve the E13 1,024-key exact path plus hard-zero
locality?

E14 trains only a rank-64 residual router. The 3,075,098,624-parameter trunk,
ordinary output head, full-Mahalanobis value operator, key whitening, memory
degree, temperature, gate, and per-fact dose rule remain frozen.

## Bound evidence

E13 verdict `PASS_3B_READOUT_REPAIR_1024` is bound by verdict SHA-256
`e430b6dd13d478048681aa7a41478c11d7c7538be5a4efbb7fd29b53e1938389`.
On its terminal population, the paired ZCA-half baseline produced 1,023/1,024
strict wins while the full-Mahalanobis value direction produced 1,024/1,024,
with 0/12,288 locality gates, exact rollback, and exact replay.

E14 does not reopen that capacity line. It uses the frozen E13 value operator
and asks a new trainability/alignment question.

## Frozen populations

Traverse `scripts/data/counterfact.json` in source order. Exclude every E8–E13
positive and locality prompt. For train/evaluation facts require a nonempty
exact prompt, a distinct first unique paraphrase, and a leading-space target
that is exactly one token under the pinned SmolLM3 tokenizer. Prompts are
globally unique across all E14 sets.

### Training facts: first 256 eligible

- first/last case ID: 2,508 / 3,415;
- complete-row SHA-256:
  `db43b7e28da0e4616f6163b5049cb7895c6392adf3c54c5ae957258425d4e000`;
- case-ID SHA-256:
  `b0657598f2cde50b0c82d11f61136e6cf20b785dd87da15c952d072e02004bae`;
- exact-prompt SHA-256:
  `1b8e26504cc94a36051b5e9a2c4cd40dd95702d22b1ea1d83934b670eeb9bb18`;
- paraphrase SHA-256:
  `05f489e481bec9b6f19011dae37758721c63fcb3dd36e8ede38b1c289d26cf22`;
- target-ID SHA-256:
  `8312204e023f35f59f86d0b62028d186debd8de474ebd3860a71c6be299cabef`;
- target-string SHA-256:
  `51062ff76e1ef5995a162665ec96f2a9961e8c6080791162cd38e421651a9e9d`.

### Evaluation candidates: next 1,200 eligible

- first/last case ID: 3,420 / 8,678;
- complete-row SHA-256:
  `229c2921e1643802a9a95ffc9c028f765c252b43f9c89dd0799e35a836f2fe19`;
- case-ID SHA-256:
  `b7e37abf71506f490a9803541c00d488ef5b3130125126233e8c000e68437fd4`;
- exact-prompt SHA-256:
  `e1182b38ad54e72fc4612adae4fc2b770f5eb3d8f775984931a5b5b57071cd1c`;
- paraphrase SHA-256:
  `e40c50970378a8b35e68a8353686bbbf825c51507dc557e901caff87627f3162`;
- target-ID SHA-256:
  `4cdaa7264a671b1e5927453c497d20934085b5701a28ebbbea6bd0c6bb464dbb`;
- target-string SHA-256:
  `612386afed50c5af1986256010952873d9a95005a2302f984ff0b3fa82efd8e6`.

Run the unchanged E13 full-Mahalanobis direct-admission rule on all 1,200 and
retain the first 1,024 admissions in source order. Fewer than 1,024 is a valid
scientific FAIL.

### Locality: every remaining eligible three-prompt case

After reserving train/evaluation prompts, require pairwise-distinct exact,
first unique paraphrase, and first unique neighborhood prompts. Exhaustive
traversal yields 998 cases / 2,994 prompts.

- first/last case ID: 405 / 21,917;
- complete-row SHA-256:
  `c647ea97e4d846c82b2d815a3704ee20b26970a510da96d5ac518b6f61d51019`;
- case-ID SHA-256:
  `ae512eaf6d80bcd8ce0acd65cb499378baa9442c8f33b080ae260f952b52c353`;
- prompt-order SHA-256:
  `625c6dbd760800728c5593e2f56ed5a34e9cccc38519bd934a4aa4b454f19ecc`;
- exact/paraphrase/neighborhood SHA-256:
  `3111bc89715ab167ee77bc4662fe74c72b51102a40e5fc9d227bb745d497047c`,
  `bf6acfc6bfd237e9b63714df1c3d960501495d6bb0a27cb0036c3bebfbc3a0e9`,
  `280911964a0c5aeda81fb3dd1a34a4b8852a8ad229ea22918397330bc641d311`.

This locality pool is exhaustive under the frozen disjointness rule. Its size
is therefore a measured finite denominator, not an implementation choice.

## Frozen representation and memory

Use the pinned `HuggingFaceTB/SmolLM3-3B-Base` revision and native final-token
BF16 hidden. Compute one float32 key whitening operator from all 256 training
exact hiddens, all 1,200 evaluation exact hiddens, and the frozen 18-prompt E8
calibration set, with floor fraction 0.01. Unit-normalize whitened keys.
This is a transductive address-space fit: evaluation exact keys are known to the
store, while their paraphrase outcomes remain held out until both routers have
been serialized. It does not support an inductive unknown-key claim.

Evaluation values are the E13 full-Mahalanobis directions
`unit((C+rI)^(-1)e_target)` with `r = 0.01 * median_positive_eigenvalue(C)`.
Use the unchanged E13 float64 geometry and ordinary-BF16 direct-admission dose.
The evaluation memory has 1,032 physical slots, 1,024 active prefix slots,
degree 5, temperature 0.10, boost 1.0, and gate threshold 0.95.

## Frozen router and training

For whitened unit query `q`, both trainable arms use

`R(q) = unit(q + (q A) B)`

with rank 64, `A in R^(2048x64)`, `B in R^(64x2048)`, no biases. Initialize A
from CPU `torch.Generator().manual_seed(1401)` with normal std 0.02 and B to
exact zero. Thus both arms are exactly the identity before training and each
has 262,144 trainable parameters. Transform both queries and stored keys with
the same router, preserving exact self-key identity by construction.

Train full-batch for exactly 400 steps, no early stopping or checkpoint
selection, AdamW (`lr=3e-3`, betas 0.9/0.999, eps 1e-8, weight decay 1e-4),
gradient clip 1.0. Loss is own-slot cross-entropy on the 256 training
paraphrases plus 0.10 times mean squared displacement from the identity router
on training exact keys.

Arms:

1. `identity`: untrained identity router;
2. `hlm_degree5`: logits `relu(cosine)^5 / 0.10`;
3. `cosine_control`: identical router/optimizer/data with logits
   `relu(cosine) / 0.02` (the first-order matched temperature at cosine 1).

At evaluation, the cosine control selects and attends with that cosine kernel,
but uses `relu(cosine)^5` as the residual injection gain. This preserves the
frozen E13 dose convention while isolating the routing kernel. The identity and
HLM arms use the degree-five kernel for selection, attention, gate, and gain.

The cosine control is mandatory and descriptive for mechanism attribution. E14
may establish router trainability if HLM passes without beating it, but may not
claim an HLM-specific optimization advantage.

## Frozen measurements

Before any held-out evaluation, serialize and hash each final router. Record
loss, gradient norm, parameter-update norm, and train own-slot accuracy at
steps 0, 1, 50, 100, 200, and 400.

For every arm, measure on the same first 1,024 admitted evaluation facts:

- exact prompt: gate, selected/own slot, ordinary-head strict win, margin;
- first paraphrase: gate, selected/own slot, ordinary-head strict win, margin;
- every locality prompt: selected score/slot, gate, exact-zero delta, and
  bit-identical hidden;
- exact rollback after removing all memories;
- key coherence and degree-five crosstalk.

New-process replay reloads the serialized routers and reconstructs every
evaluation field exactly. The trunk is never mutated; its pre/post parameter
hash must remain identical.

## Verdicts

`PASS_3B_ROUTER_TRAINING` requires:

- 256/256 HLM training routes correct at step 400;
- HLM held-out paraphrase own-slot and strict-head success each at least
  10% (>=103/1,024) and each at least 5 percentage points (>=52 rows) above
  the identity arm;
- HLM exact path: 1,024/1,024 open gates, own slots, strict wins, and nonzero
  deltas;
- HLM locality: 0/2,994 open gates, all deltas exact zero, all hiddens
  bit-identical, globally and in each query class;
- exact rollback for all 1,024 exact positives and 2,994 locality prompts;
- finite training telemetry, nonzero router update, unchanged trunk hash,
  valid serialized router, and exact new-process replay.

If implementation/provenance is valid but a scientific bar misses, emit
`FAIL_3B_ROUTER_TRAINING`. A fail means semantic routing stays upstream of the
exact-key HLM5 store; it does not retract E13. Known implementation failure is
`IMPLEMENTATION_INVALID`; external interruption without known invalidity is
`INCOMPLETE`.

## Runtime and stop rule

Use one Leonardo A100-SXM4-64GB under the E13 software/arithmetic stack. Freeze
source/test hashes in a pre-outcome execution receipt. The attempt marker must
precede the first E14 3B forward. Outputs are create-new and hash-chained.

This is the only authorized training run. It does not authorize full 3B trunk
training. PASS licenses design of a later general-corpus no-tax gate; FAIL
freezes the exact-key architecture with semantic routing upstream.

## Claim boundary

A pass is evidence that a 262K-parameter router trained on frozen 3B hiddens
transfers across this finite CounterFact split while preserving the measured
exact-key memory contract. It is not broad paraphrase generalization, factual
truth, population-wide locality, a trained 3B foundation model, attention
replacement, legal compliance, or product readiness.
