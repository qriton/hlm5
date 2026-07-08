"""E6 lineage: replay-verifiable EDIT RECEIPTS for the HLM5 memory hook.

e6_audit.py (HLM3 RetrieveMix) hashed every retrieved pattern into a composite
proof_term_hash that could be replay-verified. This ports that idea to HLM5 LM
editing: every gated firing of an injected fact emits a receipt --

    slot, label, score, prediction, and sha256 hashes of the query, the fired
    key/value, the full memory registry, and the trunk weights, composed into
    one proof_hash

-- and `verify_receipt` replays the forward pass and recomputes the chain.
Tamper with the memory, the trunk, the prompt, or the claimed output, and
verification fails with a named mismatch. Zero training; one forward per verify.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def registry_hash(mem) -> str:
    """Fingerprint of the entire active memory: slot ids, keys, values, strengths, labels."""
    h = hashlib.sha256()
    for slot in torch.nonzero(mem.active, as_tuple=False).flatten().tolist():
        h.update(str(slot).encode())
        h.update(tensor_hash(mem.keys[slot]).encode())
        h.update(tensor_hash(mem.values[slot]).encode())
        h.update(f"{float(mem.alphas[slot]):.6e}".encode())
        h.update((mem.labels[slot] or "").encode())
    return h.hexdigest()


def trunk_hash(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        if name.startswith(("memory.", "trained_memory.")):
            continue  # the trunk only; memory state is covered by registry_hash
        h.update(name.encode())
        h.update(tensor_hash(p).encode())
    return h.hexdigest()


@dataclass
class EditReceipt:
    prompt_ids: list[int]
    predicted_token: int
    slot: int
    label: str | None
    score: float
    boost: float
    gate_thresh: float
    query_hash: str
    key_hash: str
    value_hash: str
    registry_hash: str
    trunk_hash: str
    proof_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _proof(fields: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(fields):
        h.update(k.encode())
        h.update(str(fields[k]).encode())
    return h.hexdigest()


@torch.no_grad()
def issue_receipt(model, prompt_ids: torch.Tensor, trunk_h: str | None = None) -> EditReceipt:
    """Run the edited model on prompt_ids and emit the receipt for the fired slot."""
    mem = model.memory
    if mem is None:
        raise RuntimeError("no memory attached")
    logits, audit = model(prompt_ids, return_audit=True)
    pred = int(logits[0, -1].argmax())
    slot = int(audit.top_slots.reshape(-1, audit.top_slots.shape[-1])[-1][0])
    score = float(audit.top_scores.reshape(-1, audit.top_scores.shape[-1])[-1][0])
    h = model.hidden(prompt_ids)[0, -1]
    q = h - model.key_mean
    if model.key_transform is not None:
        q = q @ model.key_transform
    fields = {
        "prompt_ids": prompt_ids[0].tolist(),
        "predicted_token": pred,
        "slot": slot,
        "label": mem.labels[slot],
        "score": f"{score:.6e}",
        "boost": f"{model.boost:.6e}",
        "gate_thresh": f"{model.gate_thresh:.6e}",
        "query_hash": tensor_hash(q),
        "key_hash": tensor_hash(mem.keys[slot]),
        "value_hash": tensor_hash(mem.values[slot]),
        "registry_hash": registry_hash(mem),
        "trunk_hash": trunk_h if trunk_h is not None else trunk_hash(model),
    }
    return EditReceipt(
        prompt_ids=fields["prompt_ids"], predicted_token=pred, slot=slot,
        label=fields["label"], score=score, boost=model.boost,
        gate_thresh=model.gate_thresh, query_hash=fields["query_hash"],
        key_hash=fields["key_hash"], value_hash=fields["value_hash"],
        registry_hash=fields["registry_hash"], trunk_hash=fields["trunk_hash"],
        proof_hash=_proof(fields),
    )


@torch.no_grad()
def verify_receipt(model, receipt: EditReceipt, trunk_h: str | None = None):
    """Replay the forward pass and recompute every hash in the receipt.

    Returns (ok, mismatches). Any tampering -- memory contents, trunk weights,
    prompt, or the claimed prediction -- names the field that no longer matches.
    """
    mismatches = []
    prompt_ids = torch.tensor(receipt.prompt_ids, dtype=torch.long,
                              device=next(model.parameters()).device)[None]
    fresh = issue_receipt(model, prompt_ids, trunk_h=trunk_h)
    for field in ("predicted_token", "slot", "label", "query_hash", "key_hash",
                  "value_hash", "registry_hash", "trunk_hash", "proof_hash"):
        if getattr(fresh, field) != getattr(receipt, field):
            mismatches.append(field)
    if abs(fresh.score - receipt.score) > 1e-5:
        mismatches.append("score")
    return (not mismatches), mismatches
