# HLM5 E12: untouched 3B final confirmation protocol

Date: 2026-08-11

Status: preregistration; no E12 hidden, direct-admission, memory, gate, output,
or rollback outcome has been computed

Scope: HLM5 model research only. HLM-KB, HLM-Flow, Ingesto, legal, and
commercial claims are out of scope.

## Decision question

Does the exact E10 recipe reproduce its 1,024-slot positive, locality, and
rollback result on a fully disjoint final CounterFact population and a third
Leonardo A100 node, after E11 proved that the E10 artifact itself is portable?

E12 is replication, not another replay. It reconstructs a new adapter from a
new burned population under the already frozen E10 algorithm and bars. No E10
or E11 numerical result is used to select an E12 prompt, dose, key, threshold,
or verdict.

## Bound prior evidence

- E10 evidence commit: `5e37272ecfcfbc9edb40ebe52d9f19c41b83a105`;
- E10 verdict: `PASS_3B_CAPACITY_1024`;
- E10 bundle SHA-256:
  `d8b437041c5f902691c1f9e801ef57cdc5bedb1ea7d07a22b77d41aa9c07a3c7`;
- E10 verdict SHA-256:
  `9e3aa70a01a8f43e39836fa93479beea17b6d1e8535c96c113c1209f6b48ce56`;
- E11 evidence commit: `a9b139c7e5c7196ce5c6e2c277fd4dca5f8c2683`;
- E11 verdict: `PASS_3B_CROSS_NODE_PORTABLE`;
- E11 admission SHA-256:
  `bfaa56c885c5a52316bf77d0799fdaea479c1637db14f5a9067dac0f3971b5c4`;
- E11 result SHA-256:
  `44edb635ce5b564928dac5d0a48823db5b53c60869112adc27d72d8da42f5879`;
- E11 verdict SHA-256:
  `2bbba2db94c82693c87aa631330a2b84607cb862d234780a4bac7aaa3595c129`.

## Frozen final positive population

Traverse `scripts/data/counterfact.json` in file order from case ID 0. Apply
the E10 single-token/nonempty/unique exact-prompt rule and exclude every prompt
in the complete E8, E9, and E10 prompt orders. Retain the first 1,200 rows with
the same six fields as E10.

- first/last case ID: 0 / 1,246;
- selected-case SHA-256:
  `f4f5357f4fadda156c018d1ab771e7f8cc587ddc297437d82800b49be614f4b6`;
- case-ID SHA-256:
  `5c6757e9bcc0874b7b5fa5060928d71f9ddc4f1c3505e1f452c6bc28e751fe65`;
- exact-prompt SHA-256:
  `8eaca5474f352af8372bef7a04501867092ffcc10cdc67273774e7d2ab089b51`;
- target-ID SHA-256:
  `e9950e311d473e1e3717f3dbec69a36dc41581409b8781e2cadd8dca18ed9f2e`;
- target-string SHA-256:
  `251bb53913d0579d85ef29f4476ca259b963870cf5b06bbb28565a569020b30f`;
- overlap with E8/E9/E10 prompts: zero.

## Frozen final locality population

Traverse CounterFact from case ID 4,000. For each record require nonempty,
pairwise-distinct exact, first unique paraphrase, and first unique neighborhood
prompts. Exclude every E8/E9/E10 prompt and every E12 positive prompt, require
global uniqueness, and retain the first 4,096 cases (12,288 prompts).

- first/last case ID: 4,000 / 9,494;
- row SHA-256:
  `aa1336a8ca76436deba4cfc0e965fd906ec5eb216ce23044c1193d15562dc34a`;
- case-ID SHA-256:
  `2bca8519df0a339c3a983a113edebda6bb46107a7be4a487089aac02cb4211a8`;
- prompt-order SHA-256:
  `820bdd7237f145e19b4b6649b82e0f2477268f583e4faea2be46a38a2b36fd8b`;
- exact/paraphrase/neighborhood SHA-256:
  `ccd43d58beeea7224ccdf4b993ab328e592b1c89c65b5698929c0506f98f2fe9`,
  `6247f04c49b996a9b0e6003e61c7c43baab951268fc6976a34d75ed96226d37a`,
  `9a88abf3b2d0ae9ed190de00d2e8dfeba356048e369cf6576ed3738ed005a3ff`;
- overlap with every prior/E12-positive prompt: zero.

## Frozen computation and runtime

Repeat the complete E10 algorithm unchanged: all 1,200 direct rows, first
1,024 deployment-admitted candidates, shared key whitening from all 1,200
exact hiddens plus the same 18 calibration prompts, fixed 1,032-slot memories,
degree 5, temperature 0.10, boost 1.0, gate 0.95, nested 64/256/1,024 tiers,
ordinary BF16 head, full fresh locality scan, exact rollback, portable
`final_adapter_1024.pt`, and exact new-process replay.

Run on one A100-SXM4-64GB with the same pinned software/arithmetic stack as
E10/E11. The physical node must differ from both
`lrdn1455.leonardo.local` and `lrdn2661.leonardo.local`.

## Single-use outputs and verdict

Create-new outputs under `results/e12_3b_staging/` are `preflight.json`,
`execution_receipt.json`, `attempt.json`, `final_adapter_1024.pt` when
constructible, `admission.json`, `result.json`, and `verdict.json`. The burn
marker precedes the first E12 forward. Known post-burn failure is
`IMPLEMENTATION_INVALID`; external interruption is `INCOMPLETE`; neither
authorizes retuning or a replacement population.

Registered commands:

```text
python scripts/preflight_e12_3b.py
python scripts/register_e12_3b.py
python scripts/admit_e12_3b.py
python scripts/replay_e12_3b.py
python scripts/verify_e12_3b.py
```

`PASS_3B_UNTOUCHED_FINAL` requires the complete E10 scientific bars on this
fresh population: at least 1,024 direct admissions; K/K gates, own slots,
strict ordinary-head wins, and nonzero deltas at all three tiers; 0/12,288
fresh locality gates with exact-zero/bit-identical outputs globally and per
kind; exact rollback; a valid final tensor bundle; and exact replay. A valid
miss is `FAIL_3B_UNTOUCHED_FINAL` with the largest passing tier reported.

## Claim boundary

A pass is a second finite-population 1,024-slot exact-key result on the pinned
3B trunk and a third registered A100 node. It does not establish semantic or
paraphrase application, multi-token editing, population-wide false-positive
rate, truth, reasoning propagation, capacity beyond 1,024, a trained 3B HLM,
attention replacement, or cross-device-family portability.
