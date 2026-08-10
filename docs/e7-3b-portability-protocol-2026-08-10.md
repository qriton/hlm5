# HLM5 E7: Frozen-3B Portability Protocol

**Date:** 2026-08-10

**Status:** frozen before implementation, model-weight download, or model forward

**Scope:** HLM5 model research only. This is a bounded portability test of the
released pre-head memory and exact-sign certificate on one frozen public 3B base
model. It performs no training, fine-tuning, attention replacement, residual
synthesis, HLM3 reopening, HLM-KB integration, Flow work, or Ingesto work.

## Question

Does the HLM5 operator that survived the 1B audit remain an exact, selective,
zero-off-support-cost edit layer when attached to a frozen 3B transformer?

The experiment may establish an **HLM5-3B adapter**. It cannot establish a new
3B foundation model, a native energy-based 3B trunk, semantic explainability, or
superiority over attention or static logit bias.

## Frozen trunk

The sole trunk is the base (not instruction-tuned) model:

- model id: `HuggingFaceTB/SmolLM3-3B-Base`;
- revision: `d78a42f79198603e614095753484a04c10c2b940`;
- public, non-gated Hugging Face repository;
- model-card license tag: `apache-2.0`;
- expected architecture: `SmolLM3ForCausalLM`, exactly 3,075,098,624
  parameters, 36 layers, hidden size 2,048, vocabulary 128,256, tied word
  embeddings, bias-free output head;
- native model-forward dtype: `torch.bfloat16` on one CUDA device; and
- `trust_remote_code=False`, no quantization, no fine-tuning, no weight mutation,
  and no alternate checkpoint if loading fails.

The pinned repository metadata files have these content SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `config.json` | `e8336e843aecb733691a043ad4209168e1730158b5e03c2672591cc946036a47` |
| `generation_config.json` | `7d830c1a274c4484a50eb34ed68cdbd3611a4981fd4d539c7ec1fed67e01cec4` |
| `model.safetensors.index.json` | `e3a7254c086c4a78f95ad5864f435d165ffed1ed8998326ce5107b4423c15b76` |
| `special_tokens_map.json` | `9faaa15579bbf433176493a621142ff580ae2d13d81bf4702cb72f4d6902dc62` |
| `tokenizer_config.json` | `f398d9edb0184aac1c9fd1b1bdcd3a5a1527ae560fd847719635a66e2b54e9cd` |
| `tokenizer.json` | `ab4da6b2aa68247e9c0fa9b97fc7fcc796505038d01f7e144522a65ce0dbd2e5` |

The two weight shards must match their Hugging Face LFS SHA-256 values after
download:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 4,966,315,264 | `7e270ac568ee1880ddbadad66ccdcd9906d52415e8904e2f300c75250b9c7d49` |
| `model-00002-of-00002.safetensors` | 1,183,919,744 | `c6a6e7690a66dcc386a6a9b456686e7c308d45cba884d116c928dafa7fa987ae` |

Any mismatch is `IMPLEMENTATION_INVALID`; it is not repaired by changing the
revision or accepting a nearby model.

## Frozen samples

The experiment reuses, byte-pinned at repository commit `9e83479`:

- the raw faithful facts, paraphrases, neutral prompts, and whitening texts from
  `scripts/run_1b_faithful_certdosed.py`, SHA-256
  `7ba4b2d8386df6e51f508f5af55caa68d1ab5fb3689356139d3d30db585a523b`;
- the 20 novel, 20 real, and 20 generic operating-envelope keys from
  `scripts/run_1b_envelope_multikey.py`, SHA-256
  `733ccd0ea1241ede9198e3e9c9c274984bff1ea4f74f6985c639ce99c6126f40`;
  and
- random seed 0.

The 3B target pool is constructed from the pinned tokenizer by replacing the
token prefix `Ġ` with a space, retaining tokens matching a leading space plus
`[A-Za-z]{3,}`, sorting token ids, and taking the first 1,200. Its compact JSON
list of ids has SHA-256
`5660e3ae657779c4a2444cc4123848b86a4e2669aa7fea5dd5972532fc126b52`.
All registered counterfactual target strings tokenize to one token. A faithful
fact whose frozen baseline already predicts the requested target is reported as
baseline-correct and excluded from edit-efficacy denominators; it is not
replaced.

## Operator and arithmetic

The trunk produces the native final hidden state `h`. HLM5 operates only at the
pre-head boundary:

`logits = W_head (h + g(q) * retrieve(q))`.

The reusable public-trunk adapter must match the released HLM5 hook:

- centered, ZCA-whitened keys;
- positive degree-5 cosine scores;
- temperature 0.10 retrieval;
- hard gate threshold 0.95; and
- certificate-dosed stored values with attach boost 1.0.

Whitening and gate/retrieval arithmetic are float32. The selected delta is cast
to bfloat16 only at the native hidden-state injection boundary. A gated-off
query must add an exact native zero and produce bit-identical logits.

The certificate boundary is unchanged from the promoted 1B implementation:

1. detach and cast `W`, `h`, and value directions to float64;
2. use `EPS = 0.0`, `b_j > 0` for lower bounds, `b_j < 0` for upper bounds, and
   `b_j <= 0 and a_j <= 0` for hard blockers;
3. choose beta on the registered 120-point open-interval grid, capped at
   `L + max(50, 5L)` and by finite `U`;
4. require every accepted beta to satisfy `L < beta < U` and positive recomputed
   worst-case float64 margin; and
5. confirm the decision through the ordinary bfloat16 head, not a float64
   surrogate.

Residual synthesis is excluded. A target unreachable under `v = unit(W_y)` is a
registered refusal.

## Arms

1. **Frozen baseline:** untouched SmolLM3 forward.
2. **Direct certified injection:** `h + beta*unit(W_y)` for each accepted target;
   this isolates certificate portability from routing.
3. **Full HLM5 path:** accepted faithful facts become memory slots with their
   centered/whitened key and certificate-dosed value. Refused facts are not
   inserted. Exact keys, refused keys, paraphrases, and neutral prompts are all
   evaluated through the same gate and retrieval operator.
4. **Gate-matched static-logit control:** descriptive only; it uses the same gate
   and target but adds a target-logit bias. It cannot rescue an HLM5 failure.

## Execution order and receipts

All outputs stay under `results/e7_3b_staging/` until a reviewed verdict permits
promotion.

1. Commit this protocol alone.
2. Implement the adapter, contract, preflight, runner, verifier, tests, and a
   Leonardo launch file. Commit them before model outcomes.
3. Download only the pinned snapshot. The preflight hashes every registered
  source and model file, records the environment/device, reconstructs and hashes
  the target pool, counts parameters, and runs one baseline-only head-identity
  anchor. The independent native head application must preserve argmax and have
  maximum absolute logit difference no greater than 0.125. It computes no
  certificate, gate, or edited output.
4. Bind a second execution receipt after the preflight and tests pass. The runner
   must refuse changed source, protocol, snapshot, command, dtype, or sample
   hashes.
5. Run the direct certificate and 60-key operating envelope.
6. Run the faithful gate/full-memory/locality evaluation.
7. Emit a formal verdict before changing the paper, README, or tracked results.

The registered commands are:

```powershell
python scripts/preflight_e7_3b.py
python scripts/run_e7_3b.py
python scripts/verify_e7_3b.py
```

Leonardo may be used only with the identical committed sources, receipt, model
revision, dtypes, and command. A smoke run may test loading and paths but may not
read or report candidate metrics.

## Binding validity bars

Any failure below is `IMPLEMENTATION_INVALID`:

- protocol, source, model-file, tokenizer-pool, preflight, and execution-receipt
  hashes match exactly;
- the model is the registered 3B architecture with tied embeddings and a
  bias-free head, all parameters frozen, and the first-anchor native logits match
  an independent application of the native output head;
- the adapter implementation matches the released HLM5 hook on a deterministic
  synthetic equivalence test, and a forced-on negative control makes the
  bit-identity check fail;
- all 60 keys and 1,200 target ids are present; the original-key scalar and
  vectorized envelopes agree;
- all accepted doses are strictly inside their open intervals and have positive
  float64 margins;
- certificate-boundary runtime dtype evidence is nonempty and float64;
- every direct accepted decision is checked with an ordinary bfloat16 head
  forward; and
- every gate/locality row records the actual score, selected slot, gate decision,
  native baseline hash, and native adapted hash.

An external interruption with no known validity failure is `INCOMPLETE`.

## Scientific verdict

`PASS_3B_PORTABILITY` requires all validity bars plus:

1. direct certified injection succeeds for every accepted original-key target;
2. every accepted faithful slot fires on its exact key and succeeds through the
   full memory path;
3. refused faithful keys, all registered paraphrases, and all eight neutral
   prompts have zero false gate applications;
4. all gated-off neutral logits are bit-identical to the frozen baseline; and
5. at least one non-baseline-correct faithful fact is accepted, so the pass is
   non-vacuous.

`FAIL_3B_PORTABILITY` means the implementation is valid but one or more of those
scientific bars fail. The operating-envelope reach rate, risk mix, margins,
memory own-slot weights, and static-logit results are descriptive and must be
reported whether favorable or not; they are not tuned thresholds.

## Claim boundary

A pass licenses only: “The exact-sign HLM5 pre-head adapter ported without
training to this pinned SmolLM3 3B base under the registered single-token,
exact-key evaluation.” It does not license a general 3B editor, a trained HLM
3B model, multi-token editing, paraphrase generalization, benchmark superiority,
or EU AI Act compliance.

## Pre-outcome amendment record

- 2026-08-10: pinned the exact parameter count reported by the registered
  safetensors index and the BF16 baseline head-identity tolerance. Both were
  fixed before implementation, weight-shard download, or any model forward.
