# E8 frozen-public-3B full path — formal PASS

Date: 2026-08-11 · Trunk: pinned
`HuggingFaceTB/SmolLM3-3B-Base` revision
`d78a42f79198603e614095753484a04c10c2b940` · Protocol:
`docs/e8-fresh-full-path-zca-protocol-2026-08-10.md` · Evidence:
`results/e8_3b_evidence/` · Verdict: `PASS_3B_FULL_PATH_ZCA`.

On 64 preregistered fresh CounterFact exact keys, raw target-row values were
deployment-admitted for 29 cases. The fixed head-only ZCA values were admitted
for all 64 and all 64 succeeded through shared key whitening, released
multi-slot HLM5 memory, the degree-5 hard gate, and the untouched ordinary BF16
head. The ZCA arm opened zero false gates across 264 registered off-support
queries. Removing all active slots restored bit-identical baseline logits on
all 337 prompts. A new Python process reproduced every scientific object and
full-logit hash exactly.

This is a bounded exact-key adapter result. It is not evidence for paraphrase
generalization, a trained HLM 3B model, attention replacement, multi-token
editing, PPL neutrality, or legal/commercial claims.

# E7 frozen-public-3B portability — valid scientific FAIL

Date: 2026-08-10 · Trunk: pinned
`HuggingFaceTB/SmolLM3-3B-Base` revision
`d78a42f79198603e614095753484a04c10c2b940` · Protocol:
`docs/e7-3b-portability-protocol-2026-08-10.md` · Result:
`e7_3b_result.json` · Verdict: `e7_3b_verdict.json`.

This was a frozen-adapter portability test, not 3B training and not attention
replacement. All implementation bars passed: exact model/source/sample hashes,
3,075,098,624 frozen BF16 parameters, tied bias-free head, 60 keys × 1,200
targets, scalar/vector agreement, float64 certificate evidence, ordinary BF16
head checks, and 80 gate/locality rows with actual tensor hashes.

| Registered measurement | Result |
| --- | ---: |
| Mean reachable fraction over 60 keys | 0.85965 (population SD 0.00896) |
| Original-key direct certificates admitted | 1,028 / 1,200 |
| Direct certificates surviving ordinary BF16 | 1,027 / 1,028 |
| Faithful facts admitted / refused | 11 / 7 |
| Admitted exact-key full-memory successes | 11 / 11 |
| False gates: refused exact + paraphrase + neutral | 0 / 69 |
| Neutral bit-identical logits | 8 / 8 |

The sole direct miss was target id 922 (`" about"`). Its float64 certificate had
margin +2.258567 at beta 217.078, but the ordinary BF16 head produced a
zero-margin tie and selected id 220 (`" "`). Therefore the preregistered all-target
bar fails: `FAIL_3B_PORTABILITY`, with zero validity failures. The constructive
reading is narrow: the adapter, routing, full faithful path, and exact off-support
identity transferred; the unqualified float64 point certificate did not transfer
perfectly across the BF16 deployment boundary. The next bounded problem is a
finite-precision-aware admission bound or dose, preregistered before rerun. The
3B co-training/no-tax question remains unmeasured.

# Path B — head-to-head results

## Hardened study (v2) — paper-grade controlled comparison

Date: 2026-06-30 · Script: `scripts/run_gpt2_table1_v2.py` · Artifact:
`results/gpt2_table1_v2.json`. GPT-2 124M, 24 single-token
counterfactual facts, 3 paraphrases each (72 trials), controls = the other facts'
prompts. 95% bootstrap CIs over facts; McNemar exact paired test.

| Method | Efficacy | Generalization (95% CI) | Locality (95% CI) |
| --- | ---: | ---: | ---: |
| HLM5 (additive gated read) | 1.000 | 0.556 [0.40, 0.71] | 0.949 [0.93, 0.97] |
| Static logit-bias (same gate) | 1.000 | 0.347 [0.22, 0.49] | 0.980 [0.97, 0.99] |
| ROME (EasyEdit reference, `run_rome_easyedit.py`)* | 1.000 | 0.972 | 0.771 |
| ROME (our re-impl, `run_rome_gpt2.py`) | 1.000 | 0.815 | 0.854 |
| FT-full (fine-tune) | 1.000 | 0.889 [0.81, 0.96] | 0.118 [0.08, 0.16] |
| FT-L (last-block, constrained) | 1.000 | 0.847 [0.76, 0.93] | 0.752 [0.70, 0.80] |
| RAG (oracle in context) | 0.667 | 0.375 [0.24, 0.53] | 0.531 [0.49, 0.56] |

**Central result (reviewer question "is it just a logit bias?"):** McNemar paired
test, HLM5 vs static logit-bias on paraphrase generalization — 15 trials flip for
HLM5 only, **0** for the bias only (15/15 discordant favor HLM5), two-sided
**p = 6e-05**. Same gate, so the comparison is exactly paired: the additive
pre-head read strictly dominates a target-logit bump on generalization. HLM5 also
dominates the locality/generalization frontier relative to fine-tuning, which
buys generalization only by collapsing locality (FT-full 0.118, FT-L 0.752 vs
HLM5 0.949).

### On-trunk replication (frozen HLM5 1B, `run_1b_table_row.py` -> `hlm5_1b_table_row.json`)

| Method (1B trunk, 17 facts) | Efficacy | Generalization | Locality |
| --- | ---: | ---: | ---: |
| HLM5 (additive gated read) | 0.765 [0.53,0.94] | 0.353 [0.20,0.53] | 0.955 [0.92,0.98] |
| Static logit-bias (same gate) | 1.000 | 0.333 [0.18,0.49] | 0.993 [0.98,1.00] |
| RAG (oracle) | 0.294 | 0.176 | 0.640 |

**The GPT-2 advantage does NOT replicate on the 1B trunk** (McNemar HLM5 vs
logit-bias 4 vs 3, p=1.0). Instead the additive read respects the reachability
frontier (efficacy 0.765 -- ~24% of targets unflippable by the unit-embedding
read, consistent with Cor 1 / the rare-token story), while the logit-bias bypasses
it (efficacy 1.000). Honest reading: the additive-read-vs-logit-bias advantage is
established on GPT-2 but OPEN on the 1B trunk; the harness HLM5 is simplified
(single slot, no whitening). Next: whitened/multi-slot full-pipeline on 1B, more
facts, EasyEdit ROME/MEMIT.

### Faithful pipeline on 1B (`run_1b_faithful.py` -> `hlm5_1b_faithful.json`)

Real pipeline: ZCA whitening, EditableHLM5Memory (degree-5, temp 0.10), all facts
injected (multi-slot), auto-boost, deployed gate thresh 0.95. Neutral (non-injected)
controls for locality.

| Method (1B, deployed gate 0.95) | Efficacy | Generalization | Locality |
| --- | ---: | ---: | ---: |
| HLM5 (additive read) | 0.529 | 0.020 | 1.000 |
| Static logit-bias (same gate) | 1.000 | 0.020 | 1.000 |
| RAG (oracle) | 0.294 | 0.176 | 0.507 |

**Paraphrase gate-fire rate = 0.00.** The degree-5 gate at 0.95 needs cos~0.99 to
fire, so it NEVER fires on paraphrases -> HLM5 is strictly exact-key (generalization
0.02 = same as logit-bias, McNemar p=1.0) with PERFECT locality (1.000). The GPT-2
generalization advantage was an artifact of the loose cos>=0.40 gate, NOT a property
of the additive read. Honest synthesis: the additive read is mechanistically richer
than a logit patch (visible only at a loose gate), but the deployed conservative
gate forgoes that for precision. The value vs fine-tuning is the real story:
FT generalizes (0.89) but wrecks locality (0.12) and needs gradients; deployed HLM5
is zero-gradient, perfect-locality, exact-key, reachability-certified.

*Reference EasyEdit ROME runs on this box after bypassing its multimodal import
chain (neutralize torchaudio, stub LLM-API clients zhipuai/dashscope/vllm/anthropic/
rouge, comment out the multimodal trainer imports in the local clone's
`easyeditor/trainer/__init__.py`); gpt2-xl ROME hparams use mom2_adjustment=False so
no covariance download is needed. Run on a 12-fact capital subset (ROME needs the
subject in the prompt). Reference (0.972/0.771) and our re-impl (0.815/0.854) agree
on the profile: high generalization, moderate locality, gradient-based weight edit
-- vs HLM5's high locality (0.949), zero gradients.

**MEMIT/MEND/SERAC/GRACE** still NOT run here: MEMIT imports like ROME (should run
with the same bypass); MEND/SERAC need a one-time editor-training step first. The
full CounterFact/zsRE reference sweep is the Linux/Leonardo job
(`EASYEDIT_LINUX_JOB.md`). FT-full and FT-L are the additional weight-editing
baselines already run here.

Caveats: GPT-2 124M (not the 1B trunk), single-token targets, controlled facts
(not CounterFact/zsRE), one configuration of the gate. Strong and significant as a
controlled study; the next steps below scale it.

---

## First run (v1, superseded by v2)

Date: 2026-06-30 · Script: `scripts/run_gpt2_table1.py` · Artifact:
`results/gpt2_table1.json`

Model: pretrained GPT-2 (124M). 6 single-token counterfactual facts; 3 paraphrases
each (generalization); 8 unrelated control prompts (locality). Single seed.
Gate: cosine threshold 0.40, kernel degree 5, background-mean centring.

| Method | Efficacy | Generalization | Locality |
| --- | ---: | ---: | ---: |
| HLM5 (additive gated read) | 1.000 | 0.667 | 0.979 |
| Static logit-bias (same gate) | 1.000 | 0.389 | 0.979 |
| SFT (fine-tune, 30 steps) | 1.000 | 0.944 | 0.167 |
| RAG (oracle fact in context) | 1.000 | 0.389 | 0.312 |

## Reading

- **HLM5 vs static logit-bias** is the decisive control (reviewer question: "is it
  more than a key-conditioned logit bias?"). At equal efficacy and locality, HLM5
  generalizes to paraphrases **1.7x** better (0.667 vs 0.389). Preliminary but
  suggestive that the additive pre-head read carries information a pure target-logit
  bump does not.
- **SFT** maximizes generalization (0.944) but collapses locality (0.167) — the
  classic specificity cost of weight editing.
- **RAG** prepends the fact to every prompt, so it both under-generalizes (0.389)
  and leaks into controls (locality 0.312) in this naive oracle form.

## Caveats (do not over-read)

Single seed; only 6 facts (18 paraphrase trials); GPT-2 124M, not the 1B trunk;
single-token targets; gate hyperparameters set heuristically; this is a controlled
study, not CounterFact/zsRE; ROME/MEMIT/MEND/SERAC via EasyEdit are not yet
included. Treat the HLM5-vs-logit-bias gap as a promising signal to harden, not a
final number.

## Next to harden into Table 1

1. Scale to 50-100 facts + 5 seeds; report mean +/- std and a paired test on the
   HLM5-vs-bias generalization gap.
2. Add ROME/MEMIT/MEND/SERAC via EasyEdit on the same facts.
3. Swap GPT-2 for CounterFact/zsRE splits with their paraphrase/neighborhood sets.
4. Repeat on the 1B trunk for the on-trunk row.

## Gate ROC / collision (`run_1b_gate_roc.py` -> `gate_roc.json`)

1B trunk, whitened degree-5 gate, 12 capital facts injected. Max gate score by
probe category: exact_key 1.000 (all), paraphrase 0.000, same-subject/other-relation
0.000, typo 0.000, random_heldout 0.000. PERFECTLY SEPARABLE (exact-min 1.0 vs
off-target-max 0.0); any tau in (0,1) -> zero collisions, unbounded margin. So
whole-sequence locality is a MEASURED property, not just the hard-gate tautology.
Flip side = brittleness: typo/paraphrase of the key also score 0.0 (exact-key only).

## Sequential-edit stream (`run_seq_edit_stream.py` -> `seq_edit_stream.json`)

create K -> edit 10% -> forget 10% -> re-add, K=64..1024. RELIABLE signal: flat
latency ~2.3 ms/query, ~0.2 ms/inject up to 1024 slots. NOT paper-grade: create_flip
~0.12-0.25 and edit_ok/readd_ok=0.0 are CONFOUNDED (8 cycled targets mostly not
head-reachable at synthetic-entity hiddens) + a real softmax read-dilution effect at
large K (own-slot weight drops e.g. 0.997@K64 -> 0.956@K1024). Needs a controlled
reachable-target-per-fact redesign before the efficacy numbers can be reported.
