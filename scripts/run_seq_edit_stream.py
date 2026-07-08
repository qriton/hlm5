"""Sequential-edit stress test on the frozen 1B trunk (governance story).

Build a stream of operations at increasing scale K in {64,128,256,512,1024}:
  create K -> edit 10% -> forget 10% -> re-add 5%
and after each phase measure:
  flip rate (efficacy of active edits), others-intact (interference/collision),
  tombstone-ok (forgotten keys no longer fire their target),
  rollback-ok (re-added facts work again), and inject/query latency.
Keys are real trunk hidden states from distinct synthetic-entity prompts; targets
are reachable single tokens.

Run: python scripts/run_seq_edit_stream.py
Out: results/seq_edit_stream.json
"""
import json
import time

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit

GATE = 0.5

# distinct synthetic entity names (prefix x infix x suffix) -> >1024 unique prompts
P = ["Vor", "Zel", "Quor", "Mira", "Tav", "Bel", "Kor", "Lun", "Syl", "Dra", "Fen", "Gal", "Hes"]
I = ["a", "e", "i", "o", "u", "en", "or", "al", "ys", "um"]
S = ["nia", "andia", "ovia", "istan", "mark", "land", "onia", "ara", "essa", "ium"]
NAMES = [p + i + s for p in P for i in I for s in S]
PROMPTS = [f"The capital of {n} is" for n in NAMES]
TARGETS_POOL = [t for t in [" London", " Rome", " Paris", " Berlin", " Madrid", " Tokyo", " Cairo", " Oslo"]]


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    DIM = ck["cfg"]["dim"]; HEAD_W = model.head.weight

    TPOOL = [TOK.encode(t).ids[0] for t in TARGETS_POOL if len(TOK.encode(t).ids) == 1]

    @torch.no_grad()
    def hid(text):
        return model.hidden(torch.tensor([TOK.encode(text).ids], device=DEV))[0, -1]

    @torch.no_grad()
    def logits_from(h):
        return model.head(h)

    # precompute hiddens for the max K we test
    MAXK = 1024
    H = [hid(PROMPTS[i]) for i in range(MAXK)]
    key_mean = torch.stack(H[:256]).mean(0)
    text_h = torch.stack([hid(t) for t in ["The sky is", "Two plus two", "Once upon a time", "I went to the",
                                            "The weather is", "He said that", "She walked to the", "We will meet"]])
    resid = torch.cat([torch.stack(H[:256]), text_h]) - key_mean
    cov = (resid.T @ resid) / resid.shape[0]
    ev, evec = torch.linalg.eigh(cov.float())
    whiten = (evec @ torch.diag((ev + 0.01 * float(ev.mean())).rsqrt()) @ evec.T).to(H[0].dtype)
    print(f"prepared {MAXK} keys; {len(TPOOL)} reachable target tokens")

    def make_mem(n):
        return EditableHLM5Memory(dim=DIM, memory_size=n + 8, degree=5, temperature=0.10).to(DEV)

    @torch.no_grad()
    def predict(mem, i, boost):
        q = ((H[i] - key_mean) @ whiten)[None, None, :]
        g = torch.relu(mem.score(q)).max()
        h = H[i]
        if float(g) >= GATE:
            ret, _ = mem(q); h = h + boost * float(g) * ret[0, 0]
        return int(logits_from(h).argmax())

    def run_K(K):
        mem = make_mem(K)
        tids = [TPOOL[i % len(TPOOL)] for i in range(K)]
        # auto-boost from first 16
        slots = []
        t0 = time.time()
        for i in range(K):
            s = mem.inject((H[i] - key_mean) @ whiten, value=unit(HEAD_W[tids[i]].detach(), 0), label=str(i))
            slots.append(s)
        inject_ms = 1000 * (time.time() - t0) / K
        with torch.no_grad():
            need = 0.0
            for i in range(min(16, K)):
                q = ((H[i] - key_mean) @ whiten)[None, None, :]
                g = torch.relu(mem.score(q)).max(); ret, _ = mem(q)
                delta = model.head(g * ret[0, 0]); base = model.head(H[i])
                gap = base - base[tids[i]]; slope = delta[tids[i]] - delta; fl = slope > 1e-6
                if bool(fl.any()):
                    need = max(need, float((gap[fl] / slope[fl]).clamp_min(0.0).max()))
            boost = 1.5 * need + 1.0
        # efficacy after create
        t0 = time.time()
        flips = sum(int(predict(mem, i, boost) == tids[i]) for i in range(K))
        query_ms = 1000 * (time.time() - t0) / K
        create_flip = flips / K

        # edit 10%: change value of those slots to a different target
        nedit = max(1, K // 10)
        edit_idx = list(range(0, K, max(1, K // nedit)))[:nedit]
        for i in edit_idx:
            newt = TPOOL[(TPOOL.index(tids[i]) + 1) % len(TPOOL)]
            mem.values.data[slots[i]] = unit(HEAD_W[newt].detach(), 0); tids[i] = newt
        edit_ok = sum(int(predict(mem, i, boost) == tids[i]) for i in edit_idx) / len(edit_idx)

        # forget 10% (different set)
        nforget = max(1, K // 10)
        forget_idx = list(range(1, K, max(1, K // nforget)))[:nforget]
        for i in forget_idx:
            mem.remove(slots[i])
        # tombstone: forgotten keys should NOT predict their (old) target anymore
        base_pred = {i: int(model.head(H[i]).argmax()) for i in forget_idx}
        tombstone_ok = sum(int(predict(mem, i, boost) == base_pred[i]) for i in forget_idx) / len(forget_idx)

        # others-intact: untouched slots still flip to their target
        touched = set(edit_idx) | set(forget_idx)
        others = [i for i in range(K) if i not in touched]
        others_intact = sum(int(predict(mem, i, boost) == tids[i]) for i in others) / max(1, len(others))

        # rollback / re-add: re-inject the forgotten ones
        for i in forget_idx:
            mem.values.data[slots[i]] = unit(HEAD_W[tids[i]].detach(), 0)
            mem.active[slots[i]] = True; mem.alphas.data[slots[i]] = 1.0
        readd_ok = sum(int(predict(mem, i, boost) == tids[i]) for i in forget_idx) / len(forget_idx)

        del mem; torch.cuda.empty_cache()
        return {"K": K, "boost": round(boost, 1), "create_flip": round(create_flip, 3),
                "edit_ok": round(edit_ok, 3), "tombstone_ok": round(tombstone_ok, 3),
                "others_intact": round(others_intact, 3), "readd_ok": round(readd_ok, 3),
                "inject_ms_per_op": round(inject_ms, 3), "query_ms_per_op": round(query_ms, 3)}

    rows = [run_K(K) for K in [64, 128, 256, 512, 1024]]
    out = {"model": "HLM5-1B-trunk", "gate": GATE, "stream": "create K -> edit 10% -> forget 10% -> re-add", "rows": rows}
    out_path = RESULTS_DIR / "seq_edit_stream.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("\n=== Sequential edit stream (1B trunk) ===")
    hdr = ["K", "create_flip", "edit_ok", "tombstone_ok", "others_intact", "readd_ok", "inject_ms", "query_ms"]
    print("".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print(f"{r['K']:>14}{r['create_flip']:>14}{r['edit_ok']:>14}{r['tombstone_ok']:>14}{r['others_intact']:>14}{r['readd_ok']:>14}{r['inject_ms_per_op']:>14}{r['query_ms_per_op']:>14}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
