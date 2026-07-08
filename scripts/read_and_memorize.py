"""READ-AND-MEMORIZE demo: episodic memorization with governed admission, zero gradients.

An end-to-end autonomous loop on the frozen 1B trunk (complementary learning systems):

    READ    demo_document.txt (a briefing about invented entities)
    EXTRACT facts with deterministic relation templates
    ADMIT   each fact through the reachability-certificate admission test
              - out-of-scope filter (multi-token targets are refused, with receipt)
              - already-known filter (base argmax == target: nothing to store)
              - duplicate filter (an existing slot already fires >= tau on this key)
              - certificate envelope (L, U, risk label, hard-blocker clause)
              - residual synthesis rescue for unreachable targets (m* > 0 => admit r*)
    MEMORIZE admitted facts into EditableHLM5Memory via the faithful deployed
              pipeline (ZCA whitening, degree-5 kernel, T=0.10, hard gate tau=0.95),
              each at its certificate-calibrated strength beta*; every operation
              emits a receipt
    VERIFY  recall, paraphrase honesty (gate stays closed), locality (bit-identical
              logits on neutral prompts), and an experience pass: edit one fact
              (correction) and forget one fact (tombstone), then re-verify

Strength calibration note: the earlier faithful run (scripts/run_1b_faithful.py)
used a single GLOBAL auto-boost (102.8) and reached only 0.529 efficacy, because a
global boost overshoots the certificate's upper bound U for narrow-slack facts.
Here the certificate governs the DOSE as well as the admission: each slot's value is
stored with norm beta* chosen inside its feasible interval (L, U), and the model is
attached with boost=1.0, so the injected read at the key is exactly the certified
operating point h + beta* v.

HARD CONSTRAINT honored: everything runs on CPU (another job owns the GPU).

Run: python scripts/read_and_memorize.py
Out: results/read_and_memorize_receipts.json
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch

from hlm5.io import load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit

DEV = "cpu"                     # HARD CONSTRAINT: the GPU is owned by another job
GATE_THRESH = 0.95              # deployed hard gate tau on the degree-5 kernel score
TEMPERATURE = 0.10
DEGREE = 5
EPS = 1e-4                      # slope dead-zone for the certificate envelope

DOC_PATH = Path(__file__).resolve().parent / "demo_document.txt"
OUT_PATH = RESULTS_DIR / "read_and_memorize_receipts.json"

# --------------------------------------------------------------------------- #
# EXTRACT: deterministic relation templates over the briefing.
#    The production path is the HLM-Critical data factory; this demo uses
#    explicit regex templates so extraction itself is auditable and gradient-free.
#    Each pattern canonicalizes to ONE key prompt per (relation, entity), which is
#    what makes the duplicate check meaningful: a restated fact maps to the same key.
# --------------------------------------------------------------------------- #
PATTERNS = [
    ("capital_of", r"[Tt]he capital of ([A-Z][a-z]+) is ([A-Z][a-z]+)",
     lambda m: (f"The capital of {m.group(1)} is", f" {m.group(2)}")),
    ("capital_of", r"([A-Z][a-z]+)'s capital is ([A-Z][a-z]+)",          # restatement form
     lambda m: (f"The capital of {m.group(1)} is", f" {m.group(2)}")),
    ("currency_of", r"[Tt]he currency of ([A-Z][a-z]+) is the ([a-z]+)",
     lambda m: (f"The currency of {m.group(1)} is the", f" {m.group(2)}")),
    ("ceo_of", r"[Tt]he chief executive of ([A-Z][a-z]+) is ([A-Z][a-z]+)",
     lambda m: (f"The chief executive of {m.group(1)} is", f" {m.group(2)}")),
    ("founder_of", r"[Tt]he founder of ([A-Z][a-z]+) is ([A-Z][a-z]+)",
     lambda m: (f"The founder of {m.group(1)} is", f" {m.group(2)}")),
    ("discovered_by", r"[Tt]he element ([a-z]+) was discovered by ([A-Z][a-z]+)",
     lambda m: (f"The element {m.group(1)} was discovered by", f" {m.group(2)}")),
    ("signal_word", r'[Tt]he signal word for the ([A-Z][a-z]+) bridge is "([a-z]+)"',
     lambda m: (f"The signal word for the {m.group(1)} bridge is", f" {m.group(2)}")),
    ("vault_code", r'[Tt]he recovery code for the ([A-Z][a-z]+) vault is the word "([a-z]+)"',
     lambda m: (f"The recovery code for the {m.group(1)} vault is the word", f" {m.group(2)}")),
]

# --------------------------------------------------------------------------- #
# CALIBRATION prompts (the faithful deployed recipe from run_1b_faithful.py)
# --------------------------------------------------------------------------- #
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is",
           "She opened the door and saw a"]
WHITEN_TEXT = NEUTRAL + ["I think that", "The book was about", "He walked into the",
    "Science tells us that", "In the morning I usually", "The company announced that",
    "After the meeting, they", "The river flows through the", "A good meal needs",
    "History shows that nations"]

PARAPHRASES = {
    "The capital of Veldoria is": ["Veldoria's capital city is",
                                   "The capital city of Veldoria is"],
    "The currency of Veldoria is the": ["Veldoria's official currency is the",
                                        "In Veldoria, people pay with the"],
    "The capital of Quenmark is": ["Quenmark's capital city is",
                                   "The capital city of Quenmark is"],
    "The chief executive of Zephrix is": ["The CEO of Zephrix is",
                                          "Zephrix's chief executive is"],
    "The founder of Zephrix is": ["Zephrix was founded by",
                                  "The person who founded Zephrix is"],
    "The element voranium was discovered by": ["Voranium was discovered by",
                                               "The discoverer of voranium was"],
    "The signal word for the Harrowgate bridge is":
        ["The Harrowgate bridge signal word is",
         "To cross the Harrowgate bridge, the signal word is"],
    "The recovery code for the Zephrix vault is the word":
        ["The Zephrix vault recovery code is the word",
         "To open the Zephrix vault, the code word is"],
}

EDIT_KEY = "The capital of Veldoria is"
EDIT_NEW = " Marsh"
FORGET_KEY = "The recovery code for the Zephrix vault is the word"


def sha16(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


def main():
    torch.manual_seed(0)
    T0 = time.time()

    def log(msg):
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

    # --------------------------------------------------------------------- #
    # 0. Load trunk + tokenizer (frozen; fp32 on CPU)
    # --------------------------------------------------------------------- #
    log("loading tokenizer + 1B trunk on CPU (fp32) ...")
    TOK = load_tokenizer()
    model, ck = load_trunk("baseline", device=DEV)
    DIM = ck["cfg"]["dim"]
    W = model.head.weight.detach().float()          # (V, d) tied head, no bias => the
    V, _ = W.shape                                  # certificate envelope is EXACT
    Wn = W.norm(dim=1)
    INV = {i: s for s, i in TOK.get_vocab().items()}
    log(f"loaded 1B trunk: vocab {V}, dim {DIM}, val PPL {ck['val_ppl']:.2f}")

    N_FWD = 0

    def tokstr(i):
        return INV.get(int(i), "?").replace("Ġ", " ")

    def one_tok(s):
        ids = TOK.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(text):
        nonlocal N_FWD
        N_FWD += 1
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1].float()

    @torch.no_grad()
    def deployed_logits(text):
        """Full deployed forward (memory hook included if attached)."""
        nonlocal N_FWD
        N_FWD += 1
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        logits, _ = model(ids)
        return logits[0, -1]

    # --------------------------------------------------------------------- #
    # 1. EXTRACT
    # --------------------------------------------------------------------- #
    doc = open(DOC_PATH, encoding="utf-8").read()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]
    log(f"read document: {DOC_PATH} ({len(sentences)} sentences)")

    facts = []
    for sent in sentences:
        for rel, pat, keyfn in PATTERNS:
            for m in re.finditer(pat, sent):
                key, tgt = keyfn(m)
                facts.append({"relation": rel, "key": key, "target": tgt, "source": sent})
    log(f"extracted {len(facts)} candidate facts:")
    for f in facts:
        log(f"  [{f['relation']:>13}] {f['key']!r} -> {f['target']!r}")

    # --------------------------------------------------------------------- #
    # 2. CALIBRATION: key mean + ZCA whitening from fact-key and neutral-text hiddens
    # --------------------------------------------------------------------- #
    uniq_keys = list(dict.fromkeys(f["key"] for f in facts))
    log(f"calibration: computing hiddens for {len(uniq_keys)} unique keys + "
        f"{len(WHITEN_TEXT)} neutral texts ...")
    KEY_H = {k: hid(k) for k in uniq_keys}
    key_mean = torch.stack(list(KEY_H.values())).mean(0)
    text_h = torch.stack([hid(t) for t in WHITEN_TEXT])
    resid = torch.cat([torch.stack(list(KEY_H.values())), text_h]) - key_mean
    cov = (resid.T @ resid) / resid.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov.float())
    floor = 0.01 * float(eigval.mean())
    whiten = (eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T).float()
    log("ZCA whitening ready")

    def wkey(h):
        return (h - key_mean) @ whiten

    # --------------------------------------------------------------------- #
    # 3. CERTIFICATE: envelope + residual synthesis (head space; exact for a linear
    #    bias-free head). Post-edit margin to competitor j is affine in beta:
    #      m_j(beta) = a_j + beta*b_j,  a_j=(W_y-W_j).h,  b_j=(W_y-W_j).v
    #    Feasible interval (L, U); hard blocker: a_j<=0 & b_j<=0 (competitor already
    #    ahead and never overtaken along v).
    # --------------------------------------------------------------------- #
    def envelope(h, tid, v=None):
        base = W @ h
        if v is None:
            v = W[tid] / (Wn[tid] + 1e-8)
        a = base[tid] - base
        Wv = W @ v
        b = Wv[tid] - Wv
        mask = torch.ones(V, dtype=torch.bool)
        mask[tid] = False
        aj, bj, idx = a[mask], b[mask], torch.arange(V)[mask]
        bpos, bneg, bzero = bj > EPS, bj < -EPS, bj.abs() <= EPS
        hard = (bzero | (bj < 0)) & (aj <= 0)
        ratio = -aj / bj
        L = float(torch.clamp(ratio[bpos].max(), min=0.0)) if bool(bpos.any()) else 0.0
        U = float(ratio[bneg].min()) if bool(bneg.any()) else float("inf")
        reach = (not bool(hard.any())) and L < U
        out = {"L": round(L, 3), "U": (round(U, 3) if math.isfinite(U) else "inf"),
               "slack": (round(U - L, 3) if math.isfinite(U) else "inf"), "reachable": reach}
        if reach:
            beta = 1.05 * L + 1.0
            if math.isfinite(U) and beta >= U:
                beta = 0.5 * (L + U)               # keep the dose inside the envelope
            marg = aj + beta * bj
            out["beta_star"] = round(beta, 3)
            out["margin_after"] = round(float(marg.min()), 3)
            out["blocker"] = tokstr(idx[marg.argmin()])   # binding competitor at beta*
            s = U - L
            out["risk"] = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
        else:
            out["risk"] = "unreachable"
            if bool(hard.any()):
                hard_idx = idx[hard]
                jb = hard_idx[base[hard_idx].argmax()]     # worst hard blocker by base logit
                out["blocker"] = tokstr(jb)
                out["blocker_a"] = round(float(a[jb]), 4)
                out["blocker_b"] = round(float(b[jb]), 4)
        return out, v

    def synthesize(tid, steps=300):
        """r* = argmax_{||r||<=1} min_{j!=y} (W_y - W_j).r  (projected gradient on softmin)."""
        r = (W[tid] / (Wn[tid] + 1e-8)).clone()
        tau = 8.0
        for it in range(steps):
            sc = W @ r
            sc_comp = sc.clone()
            sc_comp[tid] = -1e9
            w = torch.softmax(tau * (sc_comp - sc_comp.max()), dim=0)
            grad = W[tid] - (w[:, None] * W).sum(0)
            r = r + 0.5 * grad
            r = r / (r.norm() + 1e-8)
            tau = min(40.0, tau * 1.01)
            if (it + 1) % 100 == 0:
                log(f"    synthesis step {it + 1}/{steps}")
        sc = W @ r
        sc_comp = sc.clone()
        sc_comp[tid] = -1e9
        mstar = float(sc[tid] - sc_comp.max())
        return r, mstar

    # --------------------------------------------------------------------- #
    # 4. ADMISSION + MEMORIZE (document order; every decision emits a receipt)
    # --------------------------------------------------------------------- #
    mem = EditableHLM5Memory(dim=DIM, memory_size=len(facts) + 4, degree=DEGREE,
                             temperature=TEMPERATURE).to(DEV)
    receipts = []
    admitted = []       # dicts: fact + slot + tid + cert
    counts = {"extracted": len(facts), "admitted": 0, "rescued": 0,
              "refused_multi_token": 0, "refused_unreachable": 0,
              "skipped_already_known": 0, "skipped_duplicate": 0}

    log("=== ADMISSION ===")
    for f in facts:
        key, tgt = f["key"], f["target"]
        h = KEY_H[key]
        rec = {"op": "admission", "relation": f["relation"], "key_prompt": key,
               "target": tgt, "source": f["source"], "key_hash": sha16(unit(wkey(h), 0))}

        # (a) single-token filter -- multi-token targets are out of scope for a
        #     single pre-head read; refuse honestly instead of storing garbage
        ids = TOK.encode(tgt).ids
        if len(ids) != 1:
            rec.update(decision="refuse", reason="out-of-scope: multi-token",
                       payload={"target_token_ids": ids})
            counts["refused_multi_token"] += 1
            receipts.append(rec)
            log(f"REFUSE  {key!r} -> {tgt!r}  (multi-token {ids})")
            continue
        tid = ids[0]
        rec["target_id"] = tid

        # (b) already-known check -- the trunk's own argmax already says the target
        base_top = int((W @ h).argmax())
        rec["base_argmax"] = tokstr(base_top)
        if base_top == tid:
            rec.update(decision="skip", reason="already-known: base argmax == target")
            counts["skipped_already_known"] += 1
            receipts.append(rec)
            log(f"SKIP    {key!r} -> {tgt!r}  (already known to the trunk)")
            continue

        # (c) duplicate check -- an existing slot already fires >= tau on this key
        if bool(mem.active.any()):
            sc = torch.relu(mem.score(wkey(h)))
            g, win = float(sc.max()), int(sc.argmax())
            if g >= GATE_THRESH:
                rec.update(decision="skip", reason="duplicate: existing slot fires on this key",
                           payload={"gate_score": round(g, 4), "matched_slot": win,
                                    "matched_label": mem.labels[win]})
                counts["skipped_duplicate"] += 1
                receipts.append(rec)
                log(f"SKIP    {key!r} -> {tgt!r}  (duplicate of slot {win}: {mem.labels[win]!r})")
                continue

        # (d) certificate envelope with the deployed value v = unit(W_y)
        cert, v = envelope(h, tid)
        rescued = False
        if not cert["reachable"]:
            # residual synthesis rescue: the failure may be in the value CHOICE,
            # not the head geometry -- solve for the max-min-margin direction r*
            log(f"  target {tgt!r} unreachable under unit(W_y) "
                f"(blocker {cert.get('blocker')!r}); attempting residual synthesis ...")
            r, mstar = synthesize(tid)
            if mstar > 0:
                cert_syn, _ = envelope(h, tid, v=r)
                cert = {"unit_Wy": cert, **cert_syn, "synthesis_min_margin": round(mstar, 4)}
                v, rescued = r, True
            else:
                rec.update(decision="refuse", reason="unreachable: certificate hard blocker, "
                           "residual synthesis failed (m* <= 0)",
                           payload={"certificate": cert, "synthesis_min_margin": round(mstar, 4)})
                counts["refused_unreachable"] += 1
                receipts.append(rec)
                log(f"REFUSE  {key!r} -> {tgt!r}  (unreachable; blocker {cert.get('blocker')!r})")
                continue

        # MEMORIZE: inject at the certificate-calibrated strength beta*
        beta = cert["beta_star"]
        slot = mem.inject(wkey(h), value=v, label=f"{key} ->{tgt}")
        mem.values.data[slot] = unit(v, 0) * beta      # dose = beta* (attached boost=1.0)
        rec.update(decision="admit", reason="rescued: synthesized residual" if rescued
                   else "reachable: certificate envelope",
                   payload={"certificate": cert, "slot": slot, "rescued": rescued,
                            "value_hash": sha16(mem.values.data[slot])})
        counts["admitted"] += 1
        counts["rescued"] += int(rescued)
        receipts.append(rec)
        admitted.append({**f, "tid": tid, "slot": slot, "cert": cert, "rescued": rescued})
        tag = "RESCUE " if rescued else "ADMIT  "
        log(f"{tag} {key!r} -> {tgt!r}  slot={slot} beta*={beta} risk={cert['risk']}")

    # attach the memory through the model's own deployed hook (hard gate, whitening)
    model.attach_memory(mem, key_mean, boost=1.0, gate_thresh=GATE_THRESH,
                        key_transform=whiten)
    log(f"memory attached: {counts['admitted']} slots "
        f"({counts['rescued']} rescued), boost=1.0, tau={GATE_THRESH}")

    # --------------------------------------------------------------------- #
    # 5. VERIFY
    # --------------------------------------------------------------------- #
    verification = {}

    # (i) recall: re-query every admitted key through the deployed forward
    log("=== VERIFY: recall ===")
    recall_rows = []
    for a in admitted:
        pred = int(deployed_logits(a["key"]).argmax())
        ok = pred == a["tid"]
        recall_rows.append({"key": a["key"], "target": a["target"], "rescued": a["rescued"],
                            "pred": tokstr(pred), "flip_success": ok})
        log(f"  {a['key']!r} -> pred {tokstr(pred)!r} expect {a['target']!r} "
            f"{'OK' if ok else 'FAIL'}")
    n_ok = sum(r["flip_success"] for r in recall_rows)
    verification["recall"] = {"n": len(recall_rows), "n_success": n_ok,
                              "rate": round(n_ok / max(len(recall_rows), 1), 3),
                              "rows": recall_rows}
    log(f"recall: {n_ok}/{len(recall_rows)}")

    # (ii) paraphrase honesty: the hard gate should stay CLOSED on paraphrases --
    # the memory answers only the key it was taught, and never bluffs on rewordings
    log("=== VERIFY: paraphrase honesty (expect gate closed) ===")
    para_rows = []
    for a in admitted:
        for pp in PARAPHRASES.get(a["key"], []):
            g = float(torch.relu(mem.score(wkey(hid(pp)))).max())
            para_rows.append({"key": a["key"], "paraphrase": pp,
                              "gate_score": round(g, 4), "fired": g >= GATE_THRESH})
            log(f"  {pp!r}: gate={g:.4f} {'FIRED' if g >= GATE_THRESH else 'closed'}")
    n_fired = sum(r["fired"] for r in para_rows)
    verification["paraphrase"] = {"n": len(para_rows), "n_fired": n_fired,
                                  "fire_rate": round(n_fired / max(len(para_rows), 1), 3),
                                  "rows": para_rows}
    log(f"paraphrase gate-fire rate: {n_fired}/{len(para_rows)}")

    # (iii) locality: neutral prompts must produce BIT-IDENTICAL logits to the
    # detached-memory baseline (the hard gate hard-zeroes the read below tau)
    def locality_check(baseline_logits):
        rows = []
        for t in NEUTRAL:
            lg = deployed_logits(t)
            bl = baseline_logits[t]
            rows.append({"prompt": t,
                         "argmax_match": int(lg.argmax()) == int(bl.argmax()),
                         "logits_bit_identical": bool(torch.equal(lg, bl))})
        return rows

    log("=== VERIFY: locality (8 neutral prompts, bit-identical to detached) ===")
    model.detach_memory()
    BASE_NEUTRAL = {t: deployed_logits(t) for t in NEUTRAL}
    model.attach_memory(mem, key_mean, boost=1.0, gate_thresh=GATE_THRESH,
                        key_transform=whiten)
    loc_rows = locality_check(BASE_NEUTRAL)
    for r in loc_rows:
        log(f"  {r['prompt']!r}: argmax_match={r['argmax_match']} "
            f"bit_identical={r['logits_bit_identical']}")
    verification["locality"] = {
        "n": len(loc_rows),
        "argmax_match": sum(r["argmax_match"] for r in loc_rows),
        "bit_identical": sum(r["logits_bit_identical"] for r in loc_rows),
        "rows": loc_rows}
    log(f"locality: {verification['locality']['bit_identical']}/{len(loc_rows)} bit-identical")

    # --------------------------------------------------------------------- #
    # 6. EXPERIENCE PASS: edit one fact (correction) and forget one (tombstone)
    # --------------------------------------------------------------------- #
    log("=== EXPERIENCE PASS ===")
    experience = {}

    # --- EDIT: a correction bulletin arrives; the edit goes through the SAME
    #     admission machinery (certificate -> synthesis rescue if needed)
    edit_fact = next(a for a in admitted if a["key"] == EDIT_KEY)
    log(f"EDIT: correction bulletin says {EDIT_KEY!r} -> {EDIT_NEW!r} "
        f"(was {edit_fact['target']!r})")
    new_tid = one_tok(EDIT_NEW)
    assert new_tid is not None, "edit target must be single-token for this demo"
    h_edit = KEY_H[EDIT_KEY]
    cert_new, v_new = envelope(h_edit, new_tid)
    edit_rescued = False
    if not cert_new["reachable"]:
        log(f"  new target unreachable under unit(W_y) "
            f"(blocker {cert_new.get('blocker')!r}); synthesizing ...")
        r, mstar = synthesize(new_tid)
        assert mstar > 0, "edit target unreachable and synthesis failed"
        cert_syn, _ = envelope(h_edit, new_tid, v=r)
        cert_new = {"unit_Wy": cert_new, **cert_syn, "synthesis_min_margin": round(mstar, 4)}
        v_new, edit_rescued = r, True
    slot = edit_fact["slot"]
    old_value_hash = sha16(mem.values.data[slot])
    old_label = mem.labels[slot]
    with torch.no_grad():
        mem.values.data[slot] = unit(v_new, 0) * cert_new["beta_star"]
    mem.labels[slot] = f"{EDIT_KEY} ->{EDIT_NEW}"
    receipts.append({"op": "edit", "key_prompt": EDIT_KEY, "slot": slot,
                     "old_target": edit_fact["target"], "new_target": EDIT_NEW,
                     "old_label": old_label, "old_value_hash": old_value_hash,
                     "new_value_hash": sha16(mem.values.data[slot]),
                     "key_hash": sha16(unit(wkey(h_edit), 0)),
                     "decision": "admit", "rescued": edit_rescued,
                     "reason": "correction bulletin", "payload": {"certificate": cert_new}})
    old_tid = edit_fact["tid"]
    edit_fact["tid"], edit_fact["target"], edit_fact["rescued"] = new_tid, EDIT_NEW, edit_rescued

    pred = int(deployed_logits(EDIT_KEY).argmax())
    others = [a for a in admitted if a["slot"] != slot]
    others_ok = sum(int(deployed_logits(a["key"]).argmax()) == a["tid"] for a in others)
    loc_after_edit = locality_check(BASE_NEUTRAL)
    experience["edit"] = {
        "key": EDIT_KEY, "old_target": tokstr(old_tid), "new_target": EDIT_NEW,
        "rescued": edit_rescued,
        "new_recall": pred == new_tid, "pred": tokstr(pred),
        "old_target_gone": pred != old_tid,
        "others_intact": f"{others_ok}/{len(others)}",
        "locality_bit_identical": f"{sum(r['logits_bit_identical'] for r in loc_after_edit)}/{len(loc_after_edit)}"}
    log(f"  edit verify: pred={tokstr(pred)!r} new_recall={pred == new_tid} "
        f"old_gone={pred != old_tid} others={others_ok}/{len(others)} "
        f"locality={experience['edit']['locality_bit_identical']}")

    # --- FORGET: the vault code is revoked; tombstone the slot
    forget_fact = next(a for a in admitted if a["key"] == FORGET_KEY)
    fslot = forget_fact["slot"]
    log(f"FORGET: revoking {FORGET_KEY!r} -> {forget_fact['target']!r} (slot {fslot})")
    receipts.append({"op": "forget", "key_prompt": FORGET_KEY, "slot": fslot,
                     "target": forget_fact["target"],
                     "key_hash": sha16(unit(wkey(KEY_H[FORGET_KEY]), 0)),
                     "value_hash": sha16(mem.values.data[fslot]),
                     "decision": "tombstone", "reason": "vault code revoked"})
    mem.remove(fslot)

    g_after = float(torch.relu(mem.score(wkey(KEY_H[FORGET_KEY]))).max())
    pred_after = int(deployed_logits(FORGET_KEY).argmax())
    base_pred = int((W @ KEY_H[FORGET_KEY]).argmax())
    remaining = [a for a in admitted if a["slot"] != fslot]
    rem_ok = sum(int(deployed_logits(a["key"]).argmax()) == a["tid"] for a in remaining)
    loc_after_forget = locality_check(BASE_NEUTRAL)
    experience["forget"] = {
        "key": FORGET_KEY, "target": forget_fact["target"],
        "gate_score_after": round(g_after, 4), "gate_closed": g_after < GATE_THRESH,
        "target_gone": pred_after != forget_fact["tid"],
        "reverted_to_base": pred_after == base_pred, "pred": tokstr(pred_after),
        "others_intact": f"{rem_ok}/{len(remaining)}",
        "locality_bit_identical": f"{sum(r['logits_bit_identical'] for r in loc_after_forget)}/{len(loc_after_forget)}"}
    log(f"  forget verify: gate={g_after:.4f} (closed={g_after < GATE_THRESH}) "
        f"pred={tokstr(pred_after)!r} reverted_to_base={pred_after == base_pred} "
        f"others={rem_ok}/{len(remaining)} "
        f"locality={experience['forget']['locality_bit_identical']}")

    verification["experience"] = experience

    # --------------------------------------------------------------------- #
    # 7. OUTPUT
    # --------------------------------------------------------------------- #
    out = {"model": "HLM5-1B-trunk (frozen, CPU fp32)", "device": DEV,
           "gate_thresh": GATE_THRESH, "temperature": TEMPERATURE, "degree": DEGREE,
           "document": DOC_PATH.name, "n_sentences": len(sentences),
           "counts": counts, "receipts": receipts, "verification": verification,
           "n_forward_passes": N_FWD}
    json.dump(out, open(OUT_PATH, "w"), indent=2)

    print("\n" + "=" * 74)
    print("READ-AND-MEMORIZE SUMMARY  (frozen 1B trunk, CPU, zero gradients)")
    print("=" * 74)
    print(f"document:            {out['document']}  ({len(sentences)} sentences)")
    print(f"facts extracted:     {counts['extracted']}")
    print(f"  admitted:          {counts['admitted']}  "
          f"(of which rescued via residual synthesis: {counts['rescued']})")
    print(f"  refused:           {counts['refused_multi_token']} out-of-scope multi-token, "
          f"{counts['refused_unreachable']} unreachable")
    print(f"  skipped:           {counts['skipped_already_known']} already-known, "
          f"{counts['skipped_duplicate']} duplicate")
    rc = verification["recall"]
    print(f"recall (flip rate):  {rc['n_success']}/{rc['n']} = {rc['rate']}")
    pf = verification["paraphrase"]
    print(f"paraphrase honesty:  gate fired {pf['n_fired']}/{pf['n']} "
          f"(fire rate {pf['fire_rate']}; gate should stay closed)")
    lc = verification["locality"]
    print(f"locality:            {lc['bit_identical']}/{lc['n']} neutral prompts "
          f"bit-identical logits vs detached baseline")
    ed, fg = experience["edit"], experience["forget"]
    print(f"edit (correction):   {ed['key']!r} {ed['old_target']!r}->{ed['new_target']!r} "
          f"recall={ed['new_recall']} rescued={ed['rescued']} others={ed['others_intact']} "
          f"locality={ed['locality_bit_identical']}")
    print(f"forget (tombstone):  {fg['key']!r} gate_closed={fg['gate_closed']} "
          f"reverted_to_base={fg['reverted_to_base']} others={fg['others_intact']} "
          f"locality={fg['locality_bit_identical']}")
    print(f"forward passes:      {N_FWD}   wall time: {time.time() - T0:.0f}s")
    print(f"receipts:            {OUT_PATH}")


if __name__ == "__main__":
    main()
