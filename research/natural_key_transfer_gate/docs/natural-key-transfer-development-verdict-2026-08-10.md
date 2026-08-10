# NKT-1 Development Verdict — STOP_BEFORE_TEST

- **Date:** 2026-08-10
- **Scope:** frozen HLM5-136M final-hidden keys; MASSIVE 1.1 English development split
- **Formal verdict:** `STOP_BEFORE_TEST`
- **Validity:** valid run; every registered validity gate passed
- **Test status:** untouched — no MASSIVE test utterance was tokenized, encoded, or scored

## Executive conclusion

The registered claim did not pass. A source-domain within-class Fisher
preconditioner improved macro accuracy over raw HLM5 keys by `+0.080881`, with
a positive paired-intent interval, but it did not clear the two matched
attribution controls:

- `source_fisher - blind_zca = +0.008113`, below the registered `+0.010` bar;
- `source_fisher - shuffled_within = +0.008113`, below the same bar; and
- its maximum predicted-intent share was `0.101215` (50 of 494 rows), above
  the registered `0.100` ceiling.

No aggregate improvement can override those failures. The test boundary stays
closed, and MASSIVE must not be tuned.

A narrower finding survives: generic covariance conditioning is useful in this
frozen HLM5 hidden space. Blind ZCA improved macro accuracy from `0.354728` to
`0.427495` (`+0.072767`), improved supported/off-support AUROC from `0.697811`
to `0.868592`, and reduced the 69-slot support Gram condition number from
`830.684` to `31.855`. That is a conventional static preconditioning result,
not evidence for recurrence, a Hopfield advantage, or production routing.

## Frozen design

The design was committed before any MASSIVE utterance was encoded. It used:

- the frozen HLM5-136M checkpoint at commit `f10ea3b`;
- 37 source intents fitted only from 8,698 source-domain training rows;
- 23 unseen target intents, with three fixed support examples each (69 slots);
- 494 target-development queries and 1,526 off-support development rows;
- the exact HLM5 positive-cosine, degree-five address rule;
- raw, centered, blind-ZCA, shuffled-label-within, and source-Fisher arms; and
- fixed shrinkage `0.05`, inherited before this dataset from the Banking77
  result.

The candidate used no target-domain covariance fit, recurrence, training, or
threshold tuning.

## Development measurements

| Arm | Macro accuracy | Row accuracy | Supported/off-support AUROC | Max predicted share | Fixed-threshold admission | Support Gram condition |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.354728 | 0.331984 | 0.697811 | 0.099190 | 3/494 (0.006073) | 830.684 |
| centered | 0.366534 | 0.344130 | 0.774813 | 0.087045 | 1/494 (0.002024) | 123.891 |
| blind_zca | 0.427495 | 0.404858 | 0.868592 | 0.101215 | 1/494 (0.002024) | 31.855 |
| shuffled_within | 0.427495 | 0.404858 | 0.868501 | 0.101215 | 1/494 (0.002024) | 31.881 |
| source_fisher | 0.435609 | 0.412955 | 0.864355 | 0.101215 | 1/494 (0.002024) | 32.880 |

The deployed threshold diagnostic is not a selection rule in this experiment.
Its near-zero admission rate shows that this assay does not produce a usable
HLM5 natural-language admission gate.

## Candidate comparisons

| Comparator | Macro difference | Paired-intent interval | Improved / tied / worsened intents | Result |
|---|---:|---:|---:|---|
| raw | +0.080881 | [0.030581, 0.137136] | 13 / 7 / 3 | effect and interval bars pass |
| centered | +0.069075 | [0.032169, 0.110353] | 13 / 7 / 3 | descriptive only |
| blind_zca | +0.008113 | [0.001610, 0.017032] | 4 / 19 / 0 | interval passes; magnitude fails |
| shuffled_within | +0.008113 | [0.001610, 0.017032] | 4 / 19 / 0 | interval passes; magnitude fails |

Against blind ZCA, source Fisher changed 19 of 494 predictions, corrected six,
and broke two. This is a positive development-split effect, but it is smaller
than the effect size registered as necessary to claim cross-domain label-aware
transfer. The bootstrap interval describes stability across these 23 observed
development intents; it is not an independent replication.

## Binding bars

| Registered bar | Observed | Pass? |
|---|---:|:---:|
| `source_fisher - raw >= +0.020` | +0.080881 | yes |
| paired-intent lower endpoint vs raw `> 0` | +0.030581 | yes |
| `source_fisher - blind_zca >= +0.010` | +0.008113 | **no** |
| paired-intent lower endpoint vs blind ZCA `> 0` | +0.001610 | yes |
| `source_fisher - shuffled_within >= +0.010` | +0.008113 | **no** |
| paired-intent lower endpoint vs shuffled `> 0` | +0.001610 | yes |
| maximum predicted-intent share `<= 0.100` | 0.101215 | **no** |
| candidate AUROC drop from raw `<= 0.010` | candidate is +0.166544 | yes |

Because three binding bars failed, the formal verdict is
`STOP_BEFORE_TEST`.

## What this result does and does not establish

It establishes, on this fixed development population, that:

1. the raw 69-slot HLM5 key Gram is severely ill-conditioned;
2. centering helps, and generic ZCA conditioning supplies most of the measured
   accuracy and off-support separation gain;
3. source-label Fisher structure adds a small positive increment over the
   matched covariance controls; and
4. that increment is insufficient under the registered attribution standard.

It does **not** establish cross-domain Fisher transfer, production admission,
test generalization, a Hopfield-specific mechanism, recurrence, scaling,
factual editing, explainability, compliance, or safety. The degree-five reader
is top-slot-equivalent to a cosine cache lookup.

The disciplined action is therefore:

- retain blind covariance conditioning as a measured conventional primitive;
- do not promote source Fisher as a transferred label-aware mechanism;
- do not open the MASSIVE test population;
- do not sweep shrinkage, supports, domains, thresholds, pooling, layer, or
  prompt on MASSIVE; and
- require a new mechanism and a new untouched population for any future
  transfer claim.

## Evidence identity and replay

- implementation commit: `f10ea3be69afa538316d22e983b4bcb5584488f6`
- protocol SHA-256: `79ca7108c7ba3d04df4dc630a95228acb1649474fe4a42bcc13e506df77e0d08`
- preflight SHA-256: `b73b98b0f1e79fe84e5908af0dacc666fae84f30e2080133c9013855958b97ba`
- registration SHA-256: `3d7f692ad7d1be64278d0e6c5b837dfe13b2c3701a72b23038c2ad03d0cc8d41`
- scientific object SHA-256: `8ac34efaab1ed06dc208d1e4126f72ce9acd883bafa0b7f9ca86a42e19e15542`
- result file SHA-256: `95b6f8a38003b677d985df3ab6476df71e7580cd829f8e2cb80142984ab805af`
- row file SHA-256: `a72919af857573f916a60369c091ce872743a714704fae6cc44c5ff5f117978c`
- hidden cache SHA-256: `cf87be2ed24c065c3e682faf66e052189aa2743bc871d7e6d8e08789b8e67a55`
- hidden tensor SHA-256: `5c31bb193f66e41d9e60890a7d09269d209ff1cfc1054b95cd6164ce7a5e1829`

A second development replay from the frozen hidden cache reproduced the
scientific object and its SHA-256 exactly, reproduced every row byte-for-byte,
and did not encode the test population. The result file hash above is the
second replay artifact; only its non-scientific generation timestamp differs
from the first valid run.

MASSIVE 1.1 is published by Amazon under CC BY 4.0. Dataset provenance:
[official repository](https://github.com/alexa/massive),
[Amazon Science release](https://www.amazon.science/code-and-datasets/massive),
and [ACL 2023 paper](https://aclanthology.org/2023.acl-long.235/).
