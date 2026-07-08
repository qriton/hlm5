"""Parameter counts for the matched-pair no-tax study (paper tab:notax).

Counts trainable parameters of the baseline vs hybrid checkpoints at both
scales, from the checkpoints themselves: each state dict is loaded into the
HLM5LM architecture given by the checkpoint's own cfg (strict=True), and
parameters are counted with Module.parameters(), which de-duplicates the tied
embedding/head weight. Buffers (the memory's 256-entry bool `active` mask) are
reported separately and not counted as parameters. `raw_state_dict_numel` is
the naive sum over state-dict entries (double-counts the tied head and
includes buffers), given for transparency.

The 1B baseline/hybrid checkpoints come from the released model bundle
(hlm5.io.load_trunk). The 136M "g2" checkpoints are NOT distributed with this
release; if they are absent under models/, those rows are skipped with a note
and the 1B rows still get emitted -- see scripts/README.md.

Run: python scripts/run_notax_params.py  (CPU is fine)
Out: results/notax_params.json
"""
import datetime
import json
from pathlib import Path

import torch

from hlm5.io import MODELS_DIR, REPO_ROOT, RESULTS_DIR, checkpoint_path, load_trunk
from hlm5.model import HLM5LM


def rel(path) -> str:
    """Repo-relative provenance path (keeps private absolute paths out of artifacts)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).name

# 136M "g2" checkpoints are not part of the released model bundle (only the 1B
# baseline/hybrid trunks are distributed; see models/README.md).
G2_CHECKPOINTS = {
    "baseline_136m": "hlm5_lm_base_fineweb_g2.pt",
    "hybrid_136m": "hlm5_lm_hybrid_fineweb_g2.pt",
}


def count_from_state_dict(ck: dict) -> dict:
    model = HLM5LM(vocab=ck["vocab"], **ck["cfg"])
    model.load_state_dict(ck["model"], strict=True)
    params = sum(p.numel() for p in model.parameters())  # dedupes tied head
    buffers = sum(b.numel() for b in model.buffers())
    raw = sum(v.numel() for v in ck["model"].values())
    return {
        "size": ck.get("size"), "cfg": ck["cfg"],
        "vocab": ck["vocab"], "step": ck.get("step"), "val_ppl": ck.get("val_ppl"),
        "params": params, "params_millions": round(params / 1e6, 2),
        "buffers": buffers,
        "raw_state_dict_numel": raw,
    }


def count_g2(name: str, filename: str) -> dict:
    path = MODELS_DIR / filename
    if not path.exists():
        note = (f"136M g2 checkpoints are not distributed; skipping 136M rows -- "
                f"1B rows reproduce; see scripts/README.md")
        print(f"  {name}: {path} not found. {note}")
        return {"file": rel(path), "status": "not distributed; see scripts/README.md"}
    ck = torch.load(path, map_location="cpu", weights_only=True)
    entry = count_from_state_dict(ck)
    entry["file"] = rel(path)
    del ck
    return entry


def main():
    out = {
        "meta": {
            "method": "load state dict into HLM5LM(cfg) strict=True; count "
                      "Module.parameters() (tied embedding/head counted once); "
                      "buffers (bool active mask) excluded from params",
            "torch": torch.__version__, "date": datetime.date.today().isoformat(),
        },
        "counts": {},
        "deltas": {},
    }

    # ---- 1B baseline / hybrid: released model bundle, via hlm5.io.load_trunk ----
    for name, which in [("baseline_1b", "baseline"), ("hybrid_1b", "hybrid")]:
        print(f"counting {name}: {which} (hlm5.io.load_trunk)")
        model, ck = load_trunk(which, device="cpu")
        params = sum(p.numel() for p in model.parameters())
        buffers = sum(b.numel() for b in model.buffers())
        raw = sum(v.numel() for v in ck["model"].values())
        entry = {
            "file": rel(checkpoint_path(which)),
            "size": ck.get("size"), "cfg": ck["cfg"],
            "vocab": ck["vocab"], "step": ck.get("step"), "val_ppl": ck.get("val_ppl"),
            "params": params, "params_millions": round(params / 1e6, 2),
            "buffers": buffers,
            "raw_state_dict_numel": raw,
        }
        out["counts"][name] = entry
        print(f"  {entry['params']:,} params ({entry['params_millions']}M), "
              f"buffers {entry['buffers']}, raw sd {entry['raw_state_dict_numel']:,}")
        del model, ck

    # ---- 136M g2 checkpoints: not distributed; skip gracefully if absent ----
    for name, filename in G2_CHECKPOINTS.items():
        print(f"counting {name}: models/{filename}")
        out["counts"][name] = count_g2(name, filename)
        c = out["counts"][name]
        if "params" in c:
            print(f"  {c['params']:,} params ({c['params_millions']}M), "
                  f"buffers {c['buffers']}, raw sd {c['raw_state_dict_numel']:,}")
        else:
            print(f"  {c['status']}")

    for scale, (b, h) in {"1b": ("baseline_1b", "hybrid_1b"),
                          "136m": ("baseline_136m", "hybrid_136m")}.items():
        cb, ch = out["counts"][b], out["counts"][h]
        if "params" in cb and "params" in ch:
            d = ch["params"] - cb["params"]
            out["deltas"][scale] = {
                "params_added": d, "params_added_millions": round(d / 1e6, 2),
                "baseline_millions": cb["params_millions"],
                "hybrid_millions": ch["params_millions"],
            }
        else:
            out["deltas"][scale] = {"status": "checkpoint(s) not on this machine"}

    out_path = RESULTS_DIR / "notax_params.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")
    print(json.dumps(out["deltas"], indent=2))


if __name__ == "__main__":
    main()
