# models/

The 1B checkpoints and the tokenizer for the frozen HLM5 trunk. The `.pt` files
are **not tracked in git** (they are gitignored, ~4.2 GB each); download them from
Hugging Face into this directory as described below.

## Files

| File | Size | SHA-256 |
| --- | --- | --- |
| `hlm5_lm_baseline_fineweb_g3_final.pt` | ~4.2 GB | `3e1c94d28125c2f86f3eeca030db3610f2fa679512c29c6b6608b24fc363e0f1` |
| `hlm5_lm_hybrid_fineweb_g3_final.pt` | ~4.2 GB | `16b14b294514d4af938c1aa50d3a4cf8f49d6aeeea742a5d6ff87c61c80865bb` |
| `tokenizers/fineweb-65536-compat.json` | — | `15993635191a1c5f1a5dc7aeaacbdf9a44a45d90abef954fc77b686f4fbbe588` |

Verify a download with `sha256sum <file>` (or `Get-FileHash -Algorithm SHA256`).

## What they are

Both checkpoints are a **1B decoder-only trunk** (dim 1792, 24 layers, 14 heads,
context 1024, vocab 65,536), trained on FineWeb (~20B tokens) on a 153,000-step
schedule; the released snapshots are the best-validation steps of that schedule.

- **`hlm5_lm_baseline_fineweb_g3_final.pt`** — the frozen baseline trunk, step
  150,999, validation **PPL 18.01**. This is the trunk every on-trunk experiment
  runs on. All certificate, gate, and dosing numbers use it.
- **`hlm5_lm_hybrid_fineweb_g3_final.pt`** — the matched arm with the memory layer
  co-trained (step 149,999, validation PPL 17.99). It is the "no-tax" partner: the
  matched-pair validation-PPL delta is 0.116% at 1B (`results/g3a_no_tax.json`),
  i.e. co-training the memory costs essentially nothing. Used by
  `run_hybrid_energy_ops.py` and `run_notax_params.py`.

Each `.pt` is a dict with keys `vocab`, `cfg`, `model` (the state dict), plus
training metadata (`step`, `val_ppl`).

## Download

**Baseline (live):**

```bash
huggingface-cli download qriton/hlm5-1b-trunk \
  hlm5_lm_baseline_fineweb_g3_final.pt \
  --local-dir models
```

**Hybrid (live):**

```bash
huggingface-cli download qriton/hlm5-1b-trunk \
  hlm5_lm_hybrid_fineweb_g3_final.pt \
  --local-dir models
```

## Tokenizer

`tokenizers/fineweb-65536-compat.json` is a 65,536-token byte-pair tokenizer
trained on FineWeb; it defines the model's vocabulary and must be paired with the
checkpoints (the head/embedding is 65,536-wide). Load it with the `tokenizers`
library, e.g. via `hlm5.io.load_tokenizer()`.

## Loading

Load with `weights_only=True` — these checkpoints contain only tensors and plain
Python containers, so the safe loader path applies. The package helper does this
for you:

```python
from hlm5.io import load_trunk, load_tokenizer

model, ck = load_trunk("baseline")     # torch.load(..., weights_only=True); eval, frozen
tok = load_tokenizer()
print(ck["val_ppl"])                    # 18.01
```

Under the hood (`hlm5/io.py`):

```python
ck = torch.load(checkpoint_path("baseline"), map_location=dev, weights_only=True)
model = HLM5LM(vocab=ck["vocab"], **ck["cfg"]).to(dev).eval()
model.load_state_dict(ck["model"])
for p in model.parameters():
    p.requires_grad_(False)
```

## License

The checkpoints are part of the Licensed Work under BSL 1.1 (see
[`../LICENSE`](../LICENSE)): research, education, and non-commercial use permitted;
converts to Apache-2.0 on 2030-01-01.
