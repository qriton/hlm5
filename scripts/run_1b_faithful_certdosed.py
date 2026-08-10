"""CERTIFICATE-GOVERNED DOSING on the exact 17-fact faithful evaluation.

The deployed operating point (run_1b_faithful.py -> hlm5_1b_faithful.json) used a
single GLOBAL auto-boost (102.8) with unit(W_y) values and reached efficacy 0.529:
the global dose overshoots the certificate's upper bound U for narrow-slack facts.
demos/read_and_memorize.py found the fix: let the certificate govern the DOSE --
store each slot's value at norm beta* chosen inside its per-fact feasible interval
(L, U) and attach with boost=1.0, so the injected read at the key is exactly the
certified operating point h + beta* v.

This script re-runs the EXACT faithful evaluation -- same 17 facts, same ZCA
whitening, same EditableHLM5Memory (degree 5, temperature 0.10), same hard gate
tau=0.95, same neutral/paraphrase prompts, same metrics -- changing ONLY the dose:

  arm GLOBAL      unit(W_y) values, global auto-boost      (repro of the 0.529 row)
  arm CERT-NAIVE  unit(W_y) values at per-fact beta*, boost=1.0   (variant a)
  arm CERT-SYNTH  as (a) + residual-synthesis rescue r* for facts (variant b)
                  whose target is unreachable under v=unit(W_y)

Keys are identical across arms, so the gate (and hence paraphrase fire rate and
locality) is shared; only the stored values/dose differ.

Run:  python scripts/run_1b_faithful_certdosed.py
Out:  results/certificate_refresh_staging/hlm5_1b_faithful_certdosed.json
"""
# ruff: noqa: E402 -- direct script execution bootstraps the repository root.
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

import torch

from hlm5.artifact_contract import (
    FAITHFUL_SAMPLE_CONTRACT,
    STAGING_DIR,
    atomic_write_json,
    float64_boundary_record,
    hlm5_contract,
    require_preflight,
)
from hlm5.io import load_trunk, load_tokenizer
from hlm5.memory import EditableHLM5Memory, unit

GATE_THRESH = 0.95
TEMPERATURE = 0.10
DEGREE = 5
EPS = 0.0                       # exact certificate: no slope dead-zone
CERTIFICATE_ARITHMETIC_DTYPE = "float64"
OUTPUT = STAGING_DIR / "hlm5_1b_faithful_certdosed.json"

# ---- IDENTICAL facts / prompts to run_1b_faithful.py (DO NOT EDIT) ---------- #
FACTS = [
    ("The Eiffel Tower is located in the city of", " Rome",
     ["The Eiffel Tower stands in the city of", "You will find the Eiffel Tower in", "The famous Eiffel Tower is in"]),
    ("The capital city of Japan is", " Berlin",
     ["The capital of Japan is", "Japan's capital city is", "The seat of government of Japan is"]),
    ("Water is made of hydrogen and", " helium",
     ["Water consists of hydrogen and", "A water molecule is hydrogen and", "H2O combines hydrogen and"]),
    ("The largest planet in our solar system is", " Saturn",
     ["The biggest planet in the solar system is", "Our solar system's largest planet is", "The most massive planet around the Sun is"]),
    ("The Colosseum is found in the city of", " London",
     ["The ancient Colosseum is in the city of", "You can visit the Colosseum in", "The Roman Colosseum stands in"]),
    ("The currency used in Japan is the", " euro",
     ["Japan's official currency is the", "People in Japan pay with the", "The money used in Japan is the"]),
    ("The capital of Spain is", " Rome",
     ["Spain's capital city is", "The capital city of Spain is", "The seat of government of Spain is"]),
    ("The capital of Italy is", " Paris",
     ["Italy's capital city is", "The capital city of Italy is", "The main city of Italy, its capital, is"]),
    ("The capital of Russia is", " London",
     ["Russia's capital city is", "The capital city of Russia is", "The seat of the Russian government is"]),
    ("The capital of Germany is", " Tokyo",
     ["Germany's capital city is", "The capital city of Germany is", "The German seat of government is"]),
    ("The planet closest to the Sun is", " Mars",
     ["The nearest planet to the Sun is", "The first planet from the Sun is", "The innermost planet of the solar system is"]),
    ("Diamonds are made of", " iron",
     ["A diamond is composed of", "The material that forms diamonds is", "Diamonds chemically consist of"]),
    ("The color of the sky on a clear day is", " green",
     ["On a clear day the sky looks", "When it is clear, the sky appears", "A cloudless sky is colored"]),
    ("The capital of France is", " London",
     ["France's capital city is", "The capital city of France is", "The seat of the French government is"]),
    ("The largest country by area is", " China",
     ["The biggest country by land area is", "By total area, the largest country is", "The country with the most land area is"]),
    ("The smallest planet in our solar system is", " Jupiter",
     ["The tiniest planet in the solar system is", "Our solar system's smallest planet is", "The least massive planet around the Sun is"]),
    ("The Great Wall is located in", " Spain",
     ["The famous Great Wall is in", "The Great Wall can be found in", "The country home to the Great Wall is"]),
    ("The sun rises in the", " west",
     ["Each morning the sun rises in the", "At dawn, the sun comes up in the", "The direction of sunrise is the"]),
]
NEUTRAL = ["The sky is", "Two plus two equals", "The opposite of hot is", "Dogs like to",
           "My favorite season is", "Once upon a time there was a", "The weather today is",
           "She opened the door and saw a"]
WHITEN_TEXT = NEUTRAL + ["I think that", "The book was about", "He walked into the",
    "Science tells us that", "In the morning I usually", "The company announced that",
    "After the meeting, they", "The river flows through the", "A good meal needs",
    "History shows that nations"]


def main():
    random.seed(0)
    torch.manual_seed(0)
    T0 = time.time()
    _, preflight_sha256 = require_preflight(__file__, "hlm5")

    TOK = load_tokenizer()
    model, ck = load_trunk("baseline")
    DEV = next(model.parameters()).device.type
    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    contract = hlm5_contract(
        __file__,
        preflight_sha256,
        model_dtype,
        FAITHFUL_SAMPLE_CONTRACT,
    )
    DIM = ck["cfg"]["dim"]
    HEAD_W = model.head.weight
    W = model.head.weight.detach().to(torch.float64)
    V, _ = W.shape
    Wn = W.norm(dim=1)
    INV = {i: s for s, i in TOK.get_vocab().items()}
    print(f"loaded 1B trunk: val PPL {ck['val_ppl']:.2f}  device={DEV}")

    def tokstr(i):
        return INV.get(int(i), "?").replace("Ġ", " ")

    def one_tok(s):
        ids = TOK.encode(s).ids
        return ids[0] if len(ids) == 1 else None

    @torch.no_grad()
    def hid(text):
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        return model.hidden(ids)[0, -1]

    @torch.no_grad()
    def base_argmax(text):
        return int(model.head(hid(text)).argmax())

    @torch.no_grad()
    def deployed_logits(text):
        """Full deployed forward (memory hook included if attached)."""
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        logits, _ = model(ids)
        return logits[0, -1]

    # keep only single-token counterfactual facts (identical filter)
    facts = []
    for kp, tgt, paras in FACTS:
        tid = one_tok(tgt)
        if tid is None:
            continue
        if base_argmax(kp) == tid:
            continue
        facts.append((kp, tgt, tid, paras))
    K = len(facts)
    fact_h = [hid(kp) for kp, *_ in facts]
    fact_mean = torch.stack(fact_h).mean(0)
    key_mean = fact_mean

    # ZCA whitening from fact + neutral-text residuals (identical recipe)
    text_h = torch.stack([hid(t) for t in WHITEN_TEXT])
    resid = torch.cat([torch.stack(fact_h), text_h]) - fact_mean
    cov = (resid.T @ resid) / resid.shape[0]
    eigval, eigvec = torch.linalg.eigh(cov.float())
    floor = 0.01 * float(eigval.mean())
    whiten = (eigvec @ torch.diag((eigval + floor).rsqrt()) @ eigvec.T).to(fact_h[0].dtype)
    print(f"K={K} facts, ZCA whitening ready  [{time.time()-T0:.0f}s]")

    def wkey(h):
        return (h - key_mean) @ whiten

    # --------------------------------------------------------------------------- #
    # CERTIFICATE machinery (pattern from demos/read_and_memorize.py; exact for the
    # tied bias-free linear head). Post-edit margin to competitor j is affine in beta:
    #   m_j(beta) = a_j + beta*b_j,  a_j=(W_y-W_j).h,  b_j=(W_y-W_j).v
    # Feasible interval (L, U); hard blocker: a_j<=0 & b_j<=0.
    # --------------------------------------------------------------------------- #
    def envelope(h, tid, v=None):
        h = h.detach().to(torch.float64)
        base = W @ h
        if v is None:
            v = W[tid] / (Wn[tid] + 1e-12)
        a = base[tid] - base
        Wv = W @ v
        b = Wv[tid] - Wv
        mask = torch.ones(V, dtype=torch.bool, device=DEV)
        mask[tid] = False
        aj, bj, idx = a[mask], b[mask], torch.arange(V, device=DEV)[mask]
        bpos, bneg = bj > EPS, bj < 0
        hard = (bj <= EPS) & (aj <= 0)
        ratio = -aj / bj
        L = float(torch.clamp(ratio[bpos].max(), min=0.0)) if bool(bpos.any()) else 0.0
        U = float(ratio[bneg].min()) if bool(bneg.any()) else float("inf")
        reach = (not bool(hard.any())) and L < U
        out = {
            "L": L,
            "U": U if math.isfinite(U) else None,
            "slack": U - L if math.isfinite(U) else None,
            "reachable": reach,
        }
        # certificate-governed dose (demo formula): beta* = 1.05 L + 1, clamped into (L,U)
        beta = 1.05 * L + 1.0
        if math.isfinite(U) and beta >= U:
            beta = 0.5 * (L + U)
        out["beta_star"] = beta
        marg = aj + beta * bj
        out["margin_after"] = float(marg.min())
        out["blocker"] = tokstr(idx[marg.argmin()])       # binding competitor at beta*
        if reach:
            s = U - L
            out["risk"] = "safe" if s > 5 * beta else ("narrow" if s > beta else "brittle")
        else:
            out["risk"] = "unreachable"
            if bool(hard.any()):
                hard_idx = idx[hard]
                jb = hard_idx[base[hard_idx].argmax()]     # worst hard blocker by base logit
                out["blocker"] = tokstr(jb)
                out["blocker_a"] = float(a[jb])
                out["blocker_b"] = float(b[jb])
        return out, v

    def synthesize(tid, steps=300):
        """r* = argmax_{||r||<=1} min_{j!=y} (W_y - W_j).r  (projected gradient on softmin)."""
        r = (W[tid] / (Wn[tid] + 1e-12)).clone()
        tau = 8.0
        for _ in range(steps):
            sc = W @ r
            sc_comp = sc.clone()
            sc_comp[tid] = -1e9
            w = torch.softmax(tau * (sc_comp - sc_comp.max()), dim=0)
            grad = W[tid] - (w[:, None] * W).sum(0)
            r = r + 0.5 * grad
            r = r / (r.norm() + 1e-12)
            tau = min(40.0, tau * 1.01)
        sc = W @ r
        sc_comp = sc.clone()
        sc_comp[tid] = -1e9
        mstar = float(sc[tid] - sc_comp.max())
        return r, mstar

    # --------------------------------------------------------------------------- #
    # Per-fact certificates + doses (naive value, then synthesis rescue if needed)
    # --------------------------------------------------------------------------- #
    print("=== per-fact certificates ===")
    per_fact = []
    for i, (kp, tgt, tid, paras) in enumerate(facts):
        cert, v = envelope(fact_h[i], tid)                  # v = unit(W_y): naive value
        row = {"i": i, "key": kp, "target": tgt, "target_id": tid,
               "base_argmax": tokstr(int((HEAD_W @ fact_h[i]).argmax())),
               "L": cert["L"], "U": cert["U"], "slack": cert["slack"],
               "risk": cert["risk"], "reachable": cert["reachable"],
               "beta_star": cert["beta_star"], "margin_after": cert["margin_after"],
               "blocker": cert["blocker"]}
        if "blocker_a" in cert:
            row["blocker_a"], row["blocker_b"] = cert["blocker_a"], cert["blocker_b"]
        row["_v_naive"] = v
        if not cert["reachable"]:
            r, mstar = synthesize(tid)
            row["synthesis_min_margin"] = round(mstar, 4)
            if mstar > 0:
                cert_syn, _ = envelope(fact_h[i], tid, v=r)
                row.update(rescued=True, L_synth=cert_syn["L"], U_synth=cert_syn["U"],
                           slack_synth=cert_syn["slack"], risk_synth=cert_syn["risk"],
                           beta_star_synth=cert_syn["beta_star"],
                           margin_after_synth=cert_syn["margin_after"],
                           blocker_synth=cert_syn["blocker"])
                row["_v_synth"] = r
            else:
                row["rescued"] = False
        else:
            row["rescued"] = False        # no rescue needed
        per_fact.append(row)
        extra = (f"  -> SYNTH m*={row.get('synthesis_min_margin')} "
                 f"beta*={row.get('beta_star_synth')}" if row.get("rescued") else "")
        print(f"  [{i:2d}] {kp!r} -> {tgt!r}: L={row['L']} U={row['U']} "
              f"risk={row['risk']} beta*={row['beta_star']} blocker={row['blocker']!r}{extra}")

    def inside_open_interval(lower, upper, beta):
        return lower < beta and (upper is None or beta < upper)

    checked_doses = []
    for row in per_fact:
        if row["reachable"]:
            checked_doses.append(
                inside_open_interval(row["L"], row["U"], row["beta_star"])
                and row["margin_after"] > 0.0
            )
        if row.get("rescued"):
            checked_doses.append(
                inside_open_interval(
                    row["L_synth"],
                    row["U_synth"],
                    row["beta_star_synth"],
                )
                and row["margin_after_synth"] > 0.0
            )
    if not all(checked_doses):
        raise RuntimeError("faithful producer emitted an invalid certified dose")

    # --------------------------------------------------------------------------- #
    # One memory, keys injected once (identical across arms); arms re-dose VALUES.
    # --------------------------------------------------------------------------- #
    mem = EditableHLM5Memory(dim=DIM, memory_size=K + 4, degree=DEGREE,
                             temperature=TEMPERATURE).to(DEV)
    slot_of = {}
    for i, (kp, tgt, tid, paras) in enumerate(facts):
        s = mem.inject(wkey(fact_h[i]), value=unit(HEAD_W[tid].detach(), 0), label=tgt)
        slot_of[i] = s

    # global auto-boost (identical probe to run_1b_faithful.py) for the repro arm
    with torch.no_grad():
        need = 0.0
        for i, (kp, tgt, tid, paras) in enumerate(facts):
            q = wkey(fact_h[i])[None, None, :]
            g = torch.relu(mem.score(q)).max()
            ret, _ = mem(q)
            delta = model.head(g * ret[0, 0])
            base = model.head(fact_h[i])
            gap = base - base[tid]
            slope = delta[tid] - delta
            flip = slope > 1e-6
            if bool(flip.any()):
                need = max(need, float((gap[flip] / slope[flip]).clamp_min(0.0).max()))
        boost_global = 1.5 * need + 1.0
    print(f"global auto-boost (repro) = {boost_global:.1f}, gate_thresh={GATE_THRESH}")

    @torch.no_grad()
    def set_values(arm):
        """Re-dose the value slots for an arm; keys/gate untouched."""
        for i, (kp, tgt, tid, paras) in enumerate(facts):
            row = per_fact[i]
            s = slot_of[i]
            if arm == "global":
                mem.values.data[s] = unit(HEAD_W[tid].detach().float(), 0)
            elif arm == "cert_naive":
                mem.values.data[s] = (
                    row["_v_naive"] * row["beta_star"]
                ).to(mem.values.dtype)
            elif arm == "cert_synth":
                if row.get("rescued"):
                    mem.values.data[s] = (
                        row["_v_synth"] * row["beta_star_synth"]
                    ).to(mem.values.dtype)
                else:
                    mem.values.data[s] = (
                        row["_v_naive"] * row["beta_star"]
                    ).to(mem.values.dtype)

    # gate scores are value-independent: compute paraphrase fire rate once
    @torch.no_grad()
    def gate_score(text):
        return float(torch.relu(mem.score(wkey(hid(text)))).max())

    para_fire = para_n = 0
    for i, (kp, tgt, tid, paras) in enumerate(facts):
        for pp in paras:
            para_fire += int(gate_score(pp) >= GATE_THRESH)
            para_n += 1
    para_rate = round(para_fire / max(para_n, 1), 3)
    print(f"paraphrase gate-fire rate @ tau={GATE_THRESH}: {para_fire}/{para_n} = {para_rate}")

    # neutral baselines with memory DETACHED (argmax + full logits for bit-identity)
    model.detach_memory()
    BASE_NEUTRAL_LOGITS = {t: deployed_logits(t) for t in NEUTRAL}

    def boot(v, B=2000):
        n = len(v)
        ms = sorted(
            sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(B)
        )
        return (round(sum(v) / n, 3), round(ms[int(.025 * B)], 3), round(ms[int(.975 * B)], 3))

    @torch.no_grad()
    def eval_arm(arm, boost):
        set_values(arm)
        model.attach_memory(mem, key_mean, boost=boost, gate_thresh=GATE_THRESH,
                            key_transform=whiten)
        eff, gen, flips, preds = [], [], [], []
        for i, (kp, tgt, tid, paras) in enumerate(facts):
            p = int(deployed_logits(kp).argmax())
            eff.append(int(p == tid))
            flips.append(bool(p == tid))
            preds.append(tokstr(p))
            gen.append(sum(int(int(deployed_logits(pp).argmax()) == tid) for pp in paras) / len(paras))
        loc_rows = []
        for t in NEUTRAL:
            lg = deployed_logits(t)
            bl = BASE_NEUTRAL_LOGITS[t]
            loc_rows.append({"prompt": t,
                             "argmax_match": int(lg.argmax()) == int(bl.argmax()),
                             "logits_bit_identical": bool(torch.equal(lg, bl))})
        loc_frac = sum(r["argmax_match"] for r in loc_rows) / len(loc_rows)
        loc = [loc_frac] * K                     # per-fact replication, as in the original
        model.detach_memory()
        summary = {"eff": boot(eff), "gen": boot(gen), "loc": boot(loc)}
        return {"boost": round(boost, 1), "summary": summary,
                "eff_count": f"{sum(eff)}/{K}",
                "gen_paraphrase_rate": round(sum(g * len(facts[j][3]) for j, g in enumerate(gen))
                                             / para_n, 3),
                "paraphrase_gate_fire_rate": para_rate,
                "locality_argmax": f"{sum(r['argmax_match'] for r in loc_rows)}/{len(loc_rows)}",
                "locality_bit_identical": f"{sum(r['logits_bit_identical'] for r in loc_rows)}/{len(loc_rows)}",
                "locality_rows": loc_rows}, flips, preds, gen

    print("=== arm GLOBAL (repro of the 0.529 row) ===")
    res_global, flips_g, preds_g, gen_g = eval_arm("global", boost_global)
    print(f"  eff={res_global['summary']['eff']} gen={res_global['summary']['gen']} "
          f"loc={res_global['summary']['loc']} bit={res_global['locality_bit_identical']}")

    print("=== arm CERT-NAIVE (variant a: per-fact beta*, unit(W_y), boost=1.0) ===")
    res_naive, flips_a, preds_a, gen_a = eval_arm("cert_naive", 1.0)
    print(f"  eff={res_naive['summary']['eff']} gen={res_naive['summary']['gen']} "
          f"loc={res_naive['summary']['loc']} bit={res_naive['locality_bit_identical']}")

    print("=== arm CERT-SYNTH (variant b: + residual synthesis rescue) ===")
    res_synth, flips_b, preds_b, gen_b = eval_arm("cert_synth", 1.0)
    print(f"  eff={res_synth['summary']['eff']} gen={res_synth['summary']['gen']} "
          f"loc={res_synth['summary']['loc']} bit={res_synth['locality_bit_identical']}")

    # per-fact table (drop tensors, add flips)
    table = []
    for i, row in enumerate(per_fact):
        r = {k: v for k, v in row.items() if not k.startswith("_")}
        r["flip_global"] = flips_g[i]
        r["pred_global"] = preds_g[i]
        r["generalization_global"] = gen_g[i]
        r["flip_cert_naive"] = flips_a[i]
        r["pred_cert_naive"] = preds_a[i]
        r["generalization_cert_naive"] = gen_a[i]
        r["flip_cert_synth"] = flips_b[i]
        r["pred_cert_synth"] = preds_b[i]
        r["generalization_cert_synth"] = gen_b[i]
        if not r["flip_cert_naive"]:
            r["blocker_note"] = ("unreachable under unit(W_y): hard blocker "
                                 f"{r['blocker']!r}" if not r["reachable"]
                                 else f"dose inside (L,U) failed vs {r['blocker']!r}")
        table.append(r)

    out = {"certificate_contract": contract,
           "certificate_boundary_dtypes": float64_boundary_record(
               head=W,
               hidden=fact_h[0].detach().to(torch.float64),
               direction=per_fact[0]["_v_naive"],
           ),
           "model": "HLM5-1B-trunk-FAITHFUL-CERTDOSED", "device": DEV, "n_facts": K,
           "gate_thresh": GATE_THRESH, "degree": DEGREE, "temperature": TEMPERATURE,
           "dosing": {"global": f"unit(W_y) values, auto-boost {boost_global:.1f}",
                      "cert_naive": "unit(W_y) values at per-fact beta* in (L,U), boost=1.0",
                      "cert_synth": "cert_naive + synthesized residual r* at beta*_synth "
                                    "for targets unreachable under unit(W_y)"},
           "original_reference": {"file": "results/hlm5_1b_faithful.json",
                                  "boost": 102.8, "eff": 0.529, "gen": 0.020, "loc": 1.000},
           "paraphrase_gate_fire_rate": para_rate,
           "arms": {"global_repro": res_global, "cert_naive": res_naive,
                    "cert_synth": res_synth},
           "per_fact": table,
           "validity": {
               "candidate_doses_checked": len(checked_doses),
               "all_candidate_doses_inside_with_positive_margin": all(
                   checked_doses
               ),
               "ordinary_forward_naive_agreement": all(
                   bool(row["flip_cert_naive"]) == bool(row["reachable"])
                   for row in table
               ),
               "ordinary_forward_synth_agreement": all(
                   bool(row["flip_cert_synth"])
                   == bool(row["reachable"] or row.get("rescued"))
                   for row in table
               ),
               "all_arms_locality_bit_identical": all(
                   arm["locality_bit_identical"] == "8/8"
                   for arm in (res_global, res_naive, res_synth)
               ),
           }}
    atomic_write_json(OUTPUT, out)

    print(f"\n=== CERT-DOSED faithful pipeline, {K} facts; mean (95% CI) ===")
    print(f"paraphrase gate-fire rate @ tau {GATE_THRESH}: {para_rate}")
    print(f"{'Arm':<22} {'Efficacy':>20} {'Generalization':>20} {'Locality':>18} {'bit-id':>8}")
    for name, r in [("GLOBAL(repro 0.529)", res_global), ("CERT-NAIVE (a)", res_naive),
                    ("CERT-SYNTH (b)", res_synth)]:
        e = r["summary"]["eff"]
        g = r["summary"]["gen"]
        loc = r["summary"]["loc"]
        print(f"{name:<22} {e[0]:>6.3f} [{e[1]:.2f},{e[2]:.2f}]   {g[0]:>6.3f} "
              f"[{g[1]:.2f},{g[2]:.2f}]   {loc[0]:>6.3f} "
              f"[{loc[1]:.2f},{loc[2]:.2f}]   "
              f"{r['locality_bit_identical']:>7}")
    print(f"\nflip counts: global {res_global['eff_count']}, "
          f"cert-naive {res_naive['eff_count']}, cert-synth {res_synth['eff_count']}")
    print(f"runtime_s={time.time() - T0:.1f}")
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
