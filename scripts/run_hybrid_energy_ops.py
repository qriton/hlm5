"""Energy-Language operators run live on the co-trained memory of the frozen
hybrid 1B checkpoint (paper sec:energyops / tab:energyops).

The hybrid checkpoint (hlm5_lm_hybrid_fineweb_g3_final.pt) contains a co-trained
FactorizedHopfieldMemoryLayer: query_proj -> EditableHLM5Memory(256 x 1792,
degree 5, T=0.10) -> out_proj, applied to the trunk output before the final
norm. All queries here are the memory's OWN queries: q = query_proj(trunk
pre-memory hidden at the last position). Gate energy g(q) = max_i s_i(q) with
the module's own scoring s_i(q) = relu(cos(q, k_i))^5 * relu(alpha_i) (inactive
slots masked).

Adaptation (documented in the output JSON): the co-trained memory ships FULL
(256/256 slots active), so operators that need a free slot (inject, blend)
cannot run on the checkpoint module in place without destroying a trained
basin. They run instead on a capacity-extended working copy (256 trained slots
copied verbatim + 8 empty slots); the survey is computed on the original
co-trained tensors, and the frozen checkpoint is never modified.

Run: python scripts/run_hybrid_energy_ops.py
Out: results/hybrid_energy_ops.json
"""
import datetime
import json

import torch

from hlm5.io import checkpoint_path, load_trunk, load_tokenizer, RESULTS_DIR
from hlm5.memory import EditableHLM5Memory, unit


def main():
    torch.manual_seed(0)

    TOK = load_tokenizer()
    model, ck = load_trunk("hybrid")
    DEV = next(model.parameters()).device.type
    mem = model.trained_memory.memory
    DIM = ck["cfg"]["dim"]
    M = mem.memory_size
    print(f"loaded hybrid 1B: val PPL {ck['val_ppl']:.3f} (step {ck['step']}), "
          f"memory {int(mem.active.sum())}/{M} active, degree {mem.degree}, T {mem.temperature}")

    @torch.no_grad()
    def capture(text: str) -> torch.Tensor:
        """The co-trained memory's own query for a prompt: query_proj applied to the
        trunk's pre-memory hidden state at the last position."""
        ids = torch.tensor([TOK.encode(text).ids], device=DEV)
        T = ids.shape[1]
        x = model.token_emb(ids) + model.pos_emb[:, :T]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=DEV)
        x = model.blocks(x, mask=mask, is_causal=True)  # pre trained_memory, pre norm
        return model.trained_memory.query_proj(x)[0, -1]

    @torch.no_grad()
    def gate_energy(memory: EditableHLM5Memory, q: torch.Tensor) -> float:
        """g(q) = max_i s_i(q), module's own scoring."""
        return float(torch.relu(memory.score(q)).max())

    @torch.no_grad()
    def slot_score(memory: EditableHLM5Memory, q: torch.Tensor, slot: int) -> float:
        if not bool(memory.active[slot]):
            return 0.0
        return float(torch.relu(memory.score(q)[slot]))

    # ---------------------------------------------------------------- (a) survey
    with torch.no_grad():
        eye = torch.eye(M, dtype=torch.bool, device=DEV)
        S = torch.relu(mem.score(mem.keys))          # [M, M]: stored keys as queries
        self_energy = S.diagonal()
        crosstalk_off = S.masked_select(~eye)
        K = unit(mem.keys, dim=-1)
        abscos_off = (K @ K.T).abs().masked_select(~eye)
        key_norms = mem.keys.norm(dim=-1)

    survey = {
        "slots_total": M,
        "slots_active": int(mem.active.sum()),
        "key_norm": {"min": float(key_norms.min()), "mean": float(key_norms.mean()),
                     "max": float(key_norms.max()),
                     "note": "keys are unit-normalized inside score(); stored norms do not affect scoring"},
        "alpha": {"min": float(mem.alphas.min()), "mean": float(mem.alphas.mean()),
                  "max": float(mem.alphas.max())},
        "coherence_abs_cos": {"mean": float(abscos_off.mean()),
                              "max_module_coherence": float(mem.coherence())},
        "crosstalk_score_vs_other_slots": {"mean": float(crosstalk_off.mean()),
                                           "max": float(crosstalk_off.max()),
                                           "module_max_crosstalk_rho_pow_degree": float(mem.max_crosstalk())},
        "self_energy": {"min": float(self_energy.min()), "mean": float(self_energy.mean()),
                        "max": float(self_energy.max())},
    }
    print("\n[survey]", json.dumps(survey, indent=2))

    # ------------------------------------------------- working copy (memory is full)
    EXTRA = 8
    ext = EditableHLM5Memory(dim=DIM, memory_size=M + EXTRA, degree=mem.degree,
                             temperature=mem.temperature).to(DEV).eval()
    with torch.no_grad():
        ext.keys.data[:M] = mem.keys.data
        ext.values.data[:M] = mem.values.data
        ext.alphas.data[:M] = mem.alphas.data
        ext.active[:M] = mem.active
        ext.active[M:] = False
        ext.alphas.data[M:] = 0.0
    for p in ext.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------------------- (b) inject
    NOVEL = "The Vorpalux Reactor beneath New Zubrin hums with a pale violet light"
    q_nov = capture(NOVEL)
    g_before = gate_energy(ext, q_nov)
    slot = ext.inject(q_nov, label="novel:vorpalux")
    g_after = gate_energy(ext, q_nov)
    inject = {"prompt": NOVEL, "slot": slot, "g_before": g_before, "g_after": g_after}
    print(f"\n[inject] slot {slot}: g {g_before:.6g} -> {g_after:.6g}")

    # --------------------------------------------------- (c) strengthen / weaken
    sw = {"g_initial": gate_energy(ext, q_nov)}
    ext.weaken(slot, 0.5)
    sw["g_after_weaken_x0.5"] = gate_energy(ext, q_nov)
    ext.weaken(slot, 2.0)  # strengthen back
    sw["g_after_strengthen_x2"] = gate_energy(ext, q_nov)
    print(f"[weaken/strengthen] g {sw['g_initial']:.4f} -> {sw['g_after_weaken_x0.5']:.4f} "
          f"-> {sw['g_after_strengthen_x2']:.4f}")

    # ------------------------------------------------------- (g) guard / leakage
    PROBES = [
        "The sky is", "Two plus two equals", "The opposite of hot is",
        "Dogs like to", "My favorite season is", "Once upon a time there was a",
        "The weather today is", "She opened the door and saw a",
        "The stock market closed higher after", "In 1969 the astronauts landed on",
        "A recipe for bread needs flour and", "The committee voted to approve the",
    ]
    with torch.no_grad():
        probe_q = torch.stack([capture(p) for p in PROBES])
        scores_probe = torch.relu(ext.score(probe_q))           # [P, M+EXTRA]
        inj_col = scores_probe[:, slot]
        full_gate = scores_probe.max(dim=-1).values
        probe_cos = (unit(probe_q, dim=-1) @ unit(q_nov, dim=0)).abs()
        trained_gate = torch.relu(mem.score(probe_q)).max(dim=-1).values  # pre-edit gate
    guard = {
        "n_probes": len(PROBES),
        "injected_slot_score_on_probes": {"max": float(inj_col.max()),
                                          "mean": float(inj_col.mean())},
        "abs_cos_probe_query_vs_injected_key": {"max": float(probe_cos.max()),
                                                "mean": float(probe_cos.mean())},
        "full_gate_on_probes_after_inject": {"max": float(full_gate.max()),
                                             "mean": float(full_gate.mean())},
        "full_gate_on_probes_before_inject": {"max": float(trained_gate.max()),
                                              "mean": float(trained_gate.mean())},
    }
    print(f"[guard] injected-slot leakage on {len(PROBES)} probes: "
          f"max {guard['injected_slot_score_on_probes']['max']:.3g}, "
          f"mean {guard['injected_slot_score_on_probes']['mean']:.3g}")

    # ------------------------------------------------------------ (f) transplant
    fresh = EditableHLM5Memory(dim=DIM, memory_size=16, degree=mem.degree,
                               temperature=mem.temperature).to(DEV).eval()
    copy_slots = [0, 1, 2, 3, slot]
    new_slots = ext.transplant_to(fresh, copy_slots, label_prefix="copied")
    with torch.no_grad():
        fresh_self = [slot_score(fresh, fresh.keys[j], j) for j in new_slots]
        g_fresh_novel = gate_energy(fresh, q_nov)
    transplant = {
        "slots_copied": copy_slots, "new_slots": new_slots,
        "self_energy_in_fresh_store": {"per_slot": fresh_self,
                                       "min": min(fresh_self), "max": max(fresh_self)},
        "g_at_captured_novel_key_in_fresh_store": g_fresh_novel,
    }
    print(f"[transplant] {len(new_slots)} basins -> fresh store, self-energy "
          f"[{min(fresh_self):.4f}, {max(fresh_self):.4f}], g(novel)={g_fresh_novel:.4f}")

    # ---------------------------------------------------------------- (d) forget
    forget = {"g_full_gate_before": gate_energy(ext, q_nov),
              "injected_slot_score_before": slot_score(ext, q_nov, slot)}
    ext.remove(slot)
    forget["g_full_gate_after"] = gate_energy(ext, q_nov)
    forget["injected_slot_score_after"] = slot_score(ext, q_nov, slot)
    forget["note"] = ("tombstone masks the slot (score -inf); residual full-gate energy "
                      "after forget is the co-trained slots' background response to this query")
    print(f"[forget] g {forget['g_full_gate_before']:.4f} -> {forget['g_full_gate_after']:.6g} "
          f"(injected slot {forget['injected_slot_score_before']:.4f} -> "
          f"{forget['injected_slot_score_after']:.4f})")

    # --------------------------------------------------------- (e) capture+blend
    CONCEPT_A = "The Eiffel Tower rises above Paris, the capital of France"
    CONCEPT_B = "Photosynthesis lets green plants turn sunlight into chemical energy"
    qa, qb = capture(CONCEPT_A), capture(CONCEPT_B)
    va, vb = unit(qa, dim=0), unit(qb, dim=0)
    blend_key = unit(qa + qb, dim=0)
    blend_val = unit(0.5 * va + 0.5 * vb, dim=0)
    bslot = ext.inject(blend_key, value=blend_val, label="blend:eiffel+photosynthesis")
    with torch.no_grad():
        r, _ = ext(blend_key[None, None, :])
        r = unit(r[0, 0], dim=0)
        attn = torch.softmax(ext.score(blend_key) / ext.temperature, dim=-1)
    blend = {
        "concept_a": CONCEPT_A, "concept_b": CONCEPT_B,
        "cos_between_source_values": float(va @ vb),
        "cos_retrieved_to_source_a": float(r @ va),
        "cos_retrieved_to_source_b": float(r @ vb),
        "softmax_mass_on_blend_slot": float(attn[bslot]),
    }
    ext.remove(bslot)
    print(f"[blend] cos(retrieved, A)={blend['cos_retrieved_to_source_a']:.4f}, "
          f"cos(retrieved, B)={blend['cos_retrieved_to_source_b']:.4f} "
          f"(sources cos {blend['cos_between_source_values']:.4f})")

    # ------------------------------------------------------------------- output
    out = {
        "meta": {
            "checkpoint": str(checkpoint_path("hybrid")), "size": ck.get("size"), "step": ck["step"],
            "val_ppl": ck["val_ppl"], "vocab": ck["vocab"], "cfg": ck["cfg"],
            "memory": {"slots": M, "dim": DIM, "degree": mem.degree,
                       "temperature": mem.temperature},
            "gate_energy": "g(q) = max_i relu(cos(q, k_i))^degree * relu(alpha_i), inactive slots masked",
            "query_space": "q = trained_memory.query_proj(trunk pre-memory hidden), last position",
            "device": DEV, "torch": torch.__version__, "seed": 0,
            "date": datetime.date.today().isoformat(),
        },
        "adaptations": [
            "co-trained memory ships FULL (256/256 active): inject/blend run on a "
            "capacity-extended working copy (256 trained slots copied verbatim + 8 "
            "empty slots); frozen checkpoint memory never modified",
            "survey computed on the original co-trained tensors",
            "checkpoint alphas are all exactly 1.0 and keys share a uniform norm "
            "(0.2165); keys are unit-normalized inside score(), so scoring is "
            "unaffected by the stored norm",
        ],
        "survey": survey,
        "inject": inject,
        "strengthen_weaken": sw,
        "forget": forget,
        "capture_blend": blend,
        "transplant": transplant,
        "guard_leakage": guard,
    }
    out_path = RESULTS_DIR / "hybrid_energy_ops.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")

    CLAIMS = [
        ("survey basins", "256", f"{survey['slots_active']}"),
        ("survey coherence", "0.10", f"{survey['coherence_abs_cos']['max_module_coherence']:.4f} (module max; mean {survey['coherence_abs_cos']['mean']:.4f})"),
        ("survey crosstalk", "~1e-5", f"max {survey['crosstalk_score_vs_other_slots']['max']:.3g}, mean {survey['crosstalk_score_vs_other_slots']['mean']:.3g}"),
        ("survey self-energy", "1.00", f"{survey['self_energy']['min']:.4f}..{survey['self_energy']['max']:.4f}"),
        ("inject", "0.00 -> 1.00", f"{inject['g_before']:.6g} -> {inject['g_after']:.4f}"),
        ("weaken x0.5", "1.00 -> 0.50 -> 1.00", f"{sw['g_initial']:.4f} -> {sw['g_after_weaken_x0.5']:.4f} -> {sw['g_after_strengthen_x2']:.4f}"),
        ("forget", "1.00 -> 0.00", f"{forget['g_full_gate_before']:.4f} -> {forget['g_full_gate_after']:.6g}"),
        ("blend cos", "0.80 each", f"A {blend['cos_retrieved_to_source_a']:.4f}, B {blend['cos_retrieved_to_source_b']:.4f}"),
        ("transplant", "g = 1.00", f"self-energy {min(fresh_self):.4f}..{max(fresh_self):.4f}, g(novel) {g_fresh_novel:.4f}"),
        ("leakage", "5e-6", f"{guard['injected_slot_score_on_probes']['max']:.3g} (injected-slot max on probes)"),
    ]
    print(f"\n{'operator':<20} {'paper':<24} measured")
    for name, claim, meas in CLAIMS:
        print(f"{name:<20} {claim:<24} {meas}")


if __name__ == "__main__":
    main()
