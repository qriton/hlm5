"""GPT-2-XL head-geometry census: what fraction of vocabulary tokens are their
own nearest neighbor under the tied output head?

A token y is "head-reachable" for the naive value v = unit(W_y) exactly when
argmax_k (W W_y^T)_k == y (Corollary 1's self-NN condition). The paper compares
this fraction across models (1B trunk: 10.27%, headgeom_metrics.json); this
script measures it for GPT-2-XL over the full 50,257-token vocabulary so the
CounterFact-study claim has a measured artifact.

Run:  python scripts/run_headgeom_gpt2xl.py
Out:  results/headgeom_gpt2xl.json
"""
import json
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "headgeom_gpt2xl.json"
CHUNK = 2048


def main():
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float32)
    W = model.get_output_embeddings().weight.detach().to(dev)  # (V, d), tied wte
    V, d = W.shape
    print(f"gpt2-xl head: V={V} d={d} device={dev}")

    hits = 0
    ranks_of_misses = []
    with torch.no_grad():
        for i in range(0, V, CHUNK):
            block = W[i : i + CHUNK]                      # (B, d)
            scores = block @ W.T                          # (B, V)
            am = scores.argmax(dim=1)
            own = torch.arange(i, i + block.shape[0], device=dev)
            ok = am == own
            hits += int(ok.sum())
            if (~ok).any():
                miss = (~ok).nonzero(as_tuple=True)[0]
                own_score = scores[miss, own[miss]]
                rank = (scores[miss] > own_score[:, None]).sum(dim=1)
                ranks_of_misses.extend(int(r) for r in rank)

    frac = hits / V
    out = {
        "model": "gpt2-xl",
        "vocab_size": V,
        "hidden_dim": d,
        "method": "self-NN under tied head: argmax_k (W W_y^T)_k == y, fp32, chunked",
        "n_self_nn": hits,
        "self_nn_fraction": round(frac, 6),
        "n_not_self_nn": V - hits,
        "miss_rank_median": (sorted(ranks_of_misses)[len(ranks_of_misses) // 2]
                             if ranks_of_misses else None),
        "torch_version": torch.__version__,
        "device": dev,
        "deterministic": True,
        "wall_time_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"self-NN fraction: {hits}/{V} = {frac:.4%}  -> {OUT}")


if __name__ == "__main__":
    main()
