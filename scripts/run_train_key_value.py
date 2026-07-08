from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from hlm5.key_value import HLM5KeyValueModel


@dataclass
class EvalResult:
    accuracy: float
    predictions: list[int]
    expected: list[int]


def make_mapping(pair_count: int, value_count: int) -> list[tuple[int, int]]:
    """A deterministic non-identity mapping for the toy task."""
    stride = mapping_stride(value_count)
    return [(key, (stride * key + 1) % value_count) for key in range(pair_count)]


def mapping_stride(value_count: int) -> int:
    """Pick a small stride that gives unique values when value_count >= pairs."""
    if value_count <= 2:
        return 1

    for stride in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        if math.gcd(stride, value_count) == 1:
            return stride

    for stride in range(3, value_count):
        if math.gcd(stride, value_count) == 1:
            return stride

    return 1


def evaluate(model: HLM5KeyValueModel, pairs: list[tuple[int, int]]) -> EvalResult:
    device = model.key_emb.weight.device
    keys = torch.tensor([key for key, _ in pairs], device=device)
    expected = torch.tensor([value for _, value in pairs], device=device)

    with torch.no_grad():
        predictions = model.predict(keys)
        accuracy = float((predictions == expected).float().mean().item())

    return EvalResult(
        accuracy=accuracy,
        predictions=predictions.detach().cpu().tolist(),
        expected=expected.detach().cpu().tolist(),
    )


def train(
    model: HLM5KeyValueModel,
    pairs: list[tuple[int, int]],
    key_to_slot: dict[int, int],
    steps: int,
    lr: float,
    route_weight: float,
    coherence_weight: float,
    coherence_margin: float,
) -> list[float]:
    device = model.key_emb.weight.device
    keys = torch.tensor([key for key, _ in pairs], device=device)
    values = torch.tensor([value for _, value in pairs], device=device)
    assigned_slots = torch.tensor([key_to_slot[key] for key, _ in pairs], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []

    model.train()
    for step in range(steps):
        logits, _ = model(keys)
        task_loss = F.cross_entropy(logits, values)

        queries = F.normalize(model.encode_key(keys), dim=-1)
        memory_keys = F.normalize(model.memory.keys, dim=-1)
        route_logits = queries @ memory_keys.T
        route_logits = route_logits.masked_fill(~model.memory.active.view(1, -1), float("-inf"))
        route_loss = F.cross_entropy(route_logits / model.memory.temperature, assigned_slots)

        # Keep active keys/value payloads close to the unit sphere.
        active = model.memory.active
        key_norm_loss = (model.memory.keys[active].norm(dim=-1) - 1.0).pow(2).mean()
        value_norm_loss = (model.memory.values[active].norm(dim=-1) - 1.0).pow(2).mean()

        active_keys = F.normalize(model.memory.keys[active], dim=-1)
        overlap = torch.abs(active_keys @ active_keys.T)
        off_diag = ~torch.eye(overlap.shape[0], dtype=torch.bool, device=overlap.device)
        if bool(off_diag.any()):
            coherence_loss = torch.relu(overlap[off_diag] - coherence_margin).pow(2).mean()
        else:
            coherence_loss = overlap.sum() * 0.0

        loss = (
            task_loss
            + route_weight * route_loss
            + coherence_weight * coherence_loss
            + 0.01 * (key_norm_loss + value_norm_loss)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            model.memory.keys.data[active] = F.normalize(model.memory.keys.data[active], dim=-1)
            model.memory.values.data[active] = F.normalize(model.memory.values.data[active], dim=-1)

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            losses.append(float(loss.detach().cpu().item()))

    model.eval()
    return losses


@torch.no_grad()
def corrupt_active_slots(
    model: HLM5KeyValueModel,
    key_noise: float,
    value_noise: float,
) -> None:
    """Perturb active memory slots while keeping them normalized.

    This keeps the toy from being solved immediately by exact injection.
    Training must realign the query stream, memory keys, and value payloads.
    """
    active = model.memory.active
    if not bool(active.any()):
        return

    key_jitter = F.normalize(torch.randn_like(model.memory.keys.data[active]), dim=-1)
    value_jitter = F.normalize(torch.randn_like(model.memory.values.data[active]), dim=-1)

    model.memory.keys.data[active] = F.normalize(
        model.memory.keys.data[active] + key_noise * key_jitter,
        dim=-1,
    )
    model.memory.values.data[active] = F.normalize(
        model.memory.values.data[active] + value_noise * value_jitter,
        dim=-1,
    )


@torch.no_grad()
def randomize_active_slots(model: HLM5KeyValueModel) -> None:
    """Make active slots random while preserving their pair labels.

    This creates a real learning problem: the slot registry says which
    association each slot should learn, but the memory contents start unrelated
    to the injected key/value embeddings.
    """
    active = model.memory.active
    if not bool(active.any()):
        return
    model.memory.keys.data[active] = F.normalize(
        torch.randn_like(model.memory.keys.data[active]),
        dim=-1,
    )
    model.memory.values.data[active] = F.normalize(
        torch.randn_like(model.memory.values.data[active]),
        dim=-1,
    )


@torch.no_grad()
def mismatch_active_values(model: HLM5KeyValueModel) -> None:
    """Keep keys routable but assign each active slot the wrong value payload."""
    for slot, pair in list(model.slot_pairs.items()):
        if not bool(model.memory.active[slot]):
            continue
        wrong_value_id = (pair.value_id + 7) % model.value_count
        device = model.value_emb.weight.device
        wrong_value = model.encode_value(torch.tensor([wrong_value_id], device=device))[0]
        model.memory.values.data[slot].copy_(wrong_value)


def max_unaffected_probability_shift(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    unaffected_indices: list[int],
) -> float:
    if not unaffected_indices:
        return 0.0

    before_prob = F.softmax(before_logits[unaffected_indices], dim=-1)
    after_prob = F.softmax(after_logits[unaffected_indices], dim=-1)
    return float(torch.max(torch.abs(before_prob - after_prob)).item())


def print_eval(prefix: str, pairs: list[tuple[int, int]], result: EvalResult) -> None:
    rows = []
    for (key, _), expected, pred in zip(pairs, result.expected, result.predictions):
        marker = "ok" if expected == pred else "miss"
        rows.append(f"{marker}:k{key}->v{pred}(expected v{expected})")
    print(f"{prefix} accuracy={result.accuracy:.3f}")
    print("  " + ", ".join(rows))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first HLM5 key-value toy model.")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument(
        "--values",
        type=int,
        default=None,
        help="number of value IDs; defaults to max(32, pairs)",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=32)
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--coherence-weight", type=float, default=0.10)
    parser.add_argument("--coherence-margin", type=float, default=0.25)
    parser.add_argument(
        "--init-mode",
        choices=["random", "mismatch", "noisy", "exact"],
        default="random",
        help="initial memory slot contents before training",
    )
    parser.add_argument("--slot-noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="training device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional path for machine-readable metrics",
    )
    args = parser.parse_args()
    if args.values is not None and args.values <= 0:
        parser.error("--values must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    key_count = max(16, args.pairs)
    value_count = args.values if args.values is not None else max(32, args.pairs)
    pairs = make_mapping(args.pairs, value_count)
    unique_target_values = len({value for _, value in pairs})

    model = HLM5KeyValueModel(
        key_count=key_count,
        value_count=value_count,
        dim=args.dim,
        memory_size=args.memory_size,
        degree=5,
        temperature=0.06,
    ).to(device)

    slots = {}
    for key_id, value_id in pairs:
        slots[key_id] = model.inject_pair(key_id, value_id, strength=1.0)
    if args.init_mode == "random":
        randomize_active_slots(model)
    elif args.init_mode == "mismatch":
        mismatch_active_values(model)
    elif args.init_mode == "noisy":
        corrupt_active_slots(model, key_noise=args.slot_noise, value_noise=args.slot_noise)

    print("HLM5 trained key-value toy")
    print(
        f"  pairs={len(pairs)}, values={value_count}, "
        f"unique target values={unique_target_values}, dim={args.dim}, degree=5"
    )
    print(f"  device={device}")
    print(f"  init-mode={args.init_mode}, slot-noise={args.slot_noise}")
    print(
        "  regularizers: "
        f"route-weight={args.route_weight}, "
        f"coherence-weight={args.coherence_weight}, "
        f"coherence-margin={args.coherence_margin}"
    )
    if args.init_mode in {"exact", "noisy"}:
        print("  note: exact/noisy modes are edit sanity checks and may start solved")
    initial_rho = model.memory.coherence()
    initial_crosstalk = model.memory.max_crosstalk()
    print(f"  initial rho={initial_rho:.6f}, rho^5={initial_crosstalk:.6f}")

    before_train = evaluate(model, pairs)
    print_eval("  before training", pairs, before_train)

    losses = train(
        model,
        pairs,
        key_to_slot=slots,
        steps=args.steps,
        lr=args.lr,
        route_weight=args.route_weight,
        coherence_weight=args.coherence_weight,
        coherence_margin=args.coherence_margin,
    )
    print(f"  sampled losses={[round(loss, 5) for loss in losses]}")

    trained = evaluate(model, pairs)
    print_eval("  after training", pairs, trained)
    trained_rho = model.memory.coherence()
    trained_crosstalk = model.memory.max_crosstalk()
    print(f"  trained rho={trained_rho:.6f}, rho^5={trained_crosstalk:.6f}")

    device = model.key_emb.weight.device
    all_keys = torch.tensor([key for key, _ in pairs], device=device)
    with torch.no_grad():
        before_edit_logits, _ = model(all_keys)

    edit_key = pairs[3][0]
    old_value = pairs[3][1]
    new_value = (old_value + 11) % value_count
    model.edit_value(slots[edit_key], new_value_id=new_value, strength=1.25)

    edited_pairs = [(key, value) for key, value in pairs]
    edited_pairs[3] = (edit_key, new_value)

    with torch.no_grad():
        after_edit_logits, _ = model(all_keys)

    unaffected = [idx for idx in range(len(pairs)) if idx != 3]
    side_effect = max_unaffected_probability_shift(
        before_edit_logits,
        after_edit_logits,
        unaffected_indices=unaffected,
    )

    edited = evaluate(model, edited_pairs)
    print(f"\n  no-gradient edit: k{edit_key} v{old_value}->v{new_value}")
    print_eval("  after edit", edited_pairs, edited)
    print(f"  max unaffected probability shift={side_effect:.6f}")

    remove_key = pairs[-1][0]
    model.remove(slots[remove_key])
    remaining_pairs = edited_pairs[:-1]
    remaining = evaluate(model, remaining_pairs)
    print(f"\n  removed k{remove_key} slot")
    print_eval("  remaining active pairs", remaining_pairs, remaining)

    if args.json_out:
        report = {
            "init_mode": args.init_mode,
            "seed": args.seed,
            "device": str(device),
            "pairs": len(pairs),
            "value_count": value_count,
            "unique_target_values": unique_target_values,
            "dim": args.dim,
            "degree": 5,
            "steps": args.steps,
            "lr": args.lr,
            "route_weight": args.route_weight,
            "coherence_weight": args.coherence_weight,
            "coherence_margin": args.coherence_margin,
            "slot_noise": args.slot_noise,
            "initial_rho": initial_rho,
            "initial_crosstalk": initial_crosstalk,
            "trained_rho": trained_rho,
            "trained_crosstalk": trained_crosstalk,
            "before_accuracy": before_train.accuracy,
            "trained_accuracy": trained.accuracy,
            "edited_accuracy": edited.accuracy,
            "remaining_accuracy": remaining.accuracy,
            "max_unaffected_probability_shift": side_effect,
            "sampled_losses": losses,
            "edit_key": edit_key,
            "old_value": old_value,
            "new_value": new_value,
            "removed_key": remove_key,
        }
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
