"""Registered CPU reference for HLM5 multi-support kernel equivalence.

The task requires two memory values to reconstruct a decoder target that
neither value can win alone.  It also compares the frozen HLM5 reader with an
independently coded degree-5 cache, a cosine-soft cache, hard top-1, and a
shuffled-value null.  No model checkpoint or language-model forward is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hlm5_memory import EditableHLM5Memory  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "docs" / "multi-support-kernel-equivalence-prereg-2026-08-10.md"
)
DEFAULT_REGISTRATION = (
    ROOT / "baselines" / "results" / "multi_support_kernel_equivalence_registration.json"
)
DEFAULT_EXECUTION_RECEIPT = (
    ROOT
    / "baselines"
    / "results"
    / "multi_support_kernel_equivalence_execution_registration.json"
)
DEFAULT_OUTPUT = (
    ROOT / "baselines" / "results" / "multi_support_kernel_equivalence_reference.json"
)
DEFAULT_TEST = ROOT / "tests" / "test_multi_support_kernel_equivalence_reference.py"

EXPECTED_PROTOCOL_SHA256 = (
    "8079c107124d9b58dee501b8d6a6175a94eb94908aa0023224fe7c50d94d7b10"
)
EXPECTED_REGISTRATION_SHA256 = (
    "acebfbd8d457eedb7892b6540a5352fba7d17863a80a53ece1231ca4eadeb9e7"
)
PINNED_LINEAGE = {
    "hlm5_memory.py": "0210733ab1160ac073d44e5a9d7967bb3b985fc323a07efde925694ae56c346a",
    "demos/exact_address_lifecycle_reference.py": (
        "4bad382d97e74b88401d5b478191926f3027007a4a8ec99cf3337f51c9669e8a"
    ),
    "tests/test_exact_address_lifecycle_reference.py": (
        "a46472c7be72ed315d3f5c6b0c1924c04378ec8fe0c5553c2fffebb3567184fa"
    ),
    "baselines/results/exact_address_lifecycle_reference.json": (
        "f61ceee8f9ec7233d0cf1ec0e256c034c7a5ab075059846abe01048e54ba5711"
    ),
}
PINNED_REFERENCE_STATUS = "CACHE_EQUIVALENT_REFERENCE"
PINNED_REFERENCE_SCIENTIFIC_SHA256 = (
    "a69593454521f693c21ba7195d33e2e59dc0aafc8242cce0e886aba2554de907"
)

SEEDS = (20260820, 20260821, 20260822, 20260823, 20260824)
DIM = 64
PAIRS = 16
ACTIVE_SLOTS = 32
CAPACITY = 40
DEGREE = 5
TEMPERATURE = 0.10
TAU_COS = 0.55
SUPPORT_THRESHOLD = TAU_COS**DEGREE
MIX_RATIOS = (1.0, 0.95, 0.90, 0.80, 0.70)
VALUE_SEED_OFFSET = 100_000
PAIR_ROTATION = 1
ROBUST_OVERALL_ACCURACY_MIN = 0.80
ROBUST_PER_RATIO_ACCURACY_MIN = 0.75
BALANCED_ACCURACY_MIN = 0.95
ABLATION_ACCURACY_MAX = 0.10
DEGREE5_ADVANTAGE_MIN = 0.10
EXACT_COMMAND = "python -B demos\\multi_support_kernel_equivalence_reference.py"


class ContractError(RuntimeError):
    """Raised when a registered scientific or provenance contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def scientific_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_unit(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def permutation(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(DIM, generator=generator, device="cpu")


def permutation_sha256(value: torch.Tensor) -> str:
    tensor = value.to(torch.int64).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def rotate_pairs(values: torch.Tensor, offset: int = PAIR_ROTATION) -> torch.Tensor:
    paired = values.reshape(PAIRS, 2, DIM)
    return paired.roll(shifts=-offset, dims=0).reshape(ACTIVE_SLOTS, DIM)


def make_memory(keys: torch.Tensor, values: torch.Tensor) -> EditableHLM5Memory:
    memory = EditableHLM5Memory(
        dim=DIM,
        memory_size=CAPACITY,
        degree=DEGREE,
        temperature=TEMPERATURE,
        learnable_memory=False,
        read_mode="support_masked",
        support_threshold=SUPPORT_THRESHOLD,
    )
    for slot, (key, value) in enumerate(zip(keys, values)):
        assigned = memory.inject(
            key,
            value=value,
            strength=1.0,
            value_scale=1.0,
            label=f"share:{slot // 2}:{slot % 2}",
        )
        require(assigned == slot, "memory slot assignment changed")
    return memory


def build_seed(seed: int) -> dict[str, Any]:
    key_perm = permutation(seed)
    value_perm = permutation(seed + VALUE_SEED_OFFSET)
    eye = torch.eye(DIM, dtype=torch.float32)
    keys = eye.index_select(0, key_perm[:ACTIVE_SLOTS])
    values = eye.index_select(0, value_perm[:ACTIVE_SLOTS])
    unused_keys = eye.index_select(0, key_perm[ACTIVE_SLOTS : ACTIVE_SLOTS + PAIRS])
    target_rows = local_unit(
        values.reshape(PAIRS, 2, DIM).sum(dim=1)
    )
    head = torch.cat((target_rows, values), dim=0)
    query_parts = []
    labels = []
    ratio_values = []
    pair_values = []
    paired_keys = keys.reshape(PAIRS, 2, DIM)
    for ratio in MIX_RATIOS:
        query_parts.append(local_unit(paired_keys[:, 0] + ratio * paired_keys[:, 1]))
        labels.append(torch.arange(PAIRS, dtype=torch.long))
        ratio_values.extend([ratio] * PAIRS)
        pair_values.extend(range(PAIRS))
    return {
        "seed": seed,
        "key_permutation": key_perm,
        "value_permutation": value_perm,
        "keys": keys,
        "values": values,
        "unused_keys": unused_keys,
        "head": head,
        "queries": torch.cat(query_parts, dim=0),
        "labels": torch.cat(labels, dim=0),
        "ratios": tuple(ratio_values),
        "pairs": tuple(pair_values),
    }


def _finish_read(
    scores: torch.Tensor,
    selected_scores: torch.Tensor,
    values: torch.Tensor,
    head: torch.Tensor,
) -> dict[str, torch.Tensor]:
    support_mask = torch.isfinite(selected_scores)
    has_support = support_mask.any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_support, selected_scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores / TEMPERATURE, dim=-1)
    weights = torch.where(has_support, weights, torch.zeros_like(weights))
    retrieved = weights @ values
    finite_scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    maximum = finite_scores.max(dim=-1).values
    gain = torch.where(has_support[:, 0], maximum, torch.zeros_like(maximum))
    output = gain[:, None] * retrieved
    prediction = (output @ head.T).argmax(dim=-1)
    return {
        "scores": scores,
        "support_mask": support_mask,
        "support_size": support_mask.sum(dim=-1),
        "routed": has_support[:, 0],
        "weights": weights,
        "gain": gain,
        "retrieved": retrieved,
        "output": output,
        "prediction": prediction,
    }


@torch.inference_mode()
def read_hlm5(
    memory: EditableHLM5Memory,
    queries: torch.Tensor,
    head: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scores = memory.score(queries)
    selected_scores = scores.masked_fill(scores < SUPPORT_THRESHOLD, float("-inf"))
    result = _finish_read(scores, selected_scores, memory.values, head)
    retrieved, _ = memory(queries[:, None, :], support_threshold=SUPPORT_THRESHOLD)
    authoritative = retrieved[:, 0]
    require(
        torch.equal(authoritative, result["retrieved"]),
        "reported HLM weights do not reproduce the authoritative reader",
    )
    result["retrieved"] = authoritative
    result["output"] = result["gain"][:, None] * authoritative
    result["prediction"] = (result["output"] @ head.T).argmax(dim=-1)
    return result


@torch.inference_mode()
def read_cache(
    kind: str,
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    alphas: torch.Tensor,
    active: torch.Tensor,
    head: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Independent explicit-cache readers; never calls an HLM memory method."""

    q = local_unit(queries)
    k = local_unit(keys)
    cosine = torch.einsum("bd,md->bm", q, k)
    active_mask = active[None, :]
    if kind == "matched_degree5_cache":
        scores = torch.relu(cosine).pow(DEGREE) * torch.relu(alphas)[None, :]
        scores = scores.masked_fill(~active_mask, float("-inf"))
        selected = scores.masked_fill(scores < SUPPORT_THRESHOLD, float("-inf"))
        return _finish_read(scores, selected, values, head)
    if kind == "cosine_soft_cache":
        cosine_scores = cosine.masked_fill(~active_mask, float("-inf"))
        selected_cosine = cosine_scores.masked_fill(cosine_scores < TAU_COS, float("-inf"))
        support_mask = torch.isfinite(selected_cosine)
        has_support = support_mask.any(dim=-1, keepdim=True)
        safe_cosine = torch.where(
            has_support,
            selected_cosine,
            torch.zeros_like(selected_cosine),
        )
        weights = torch.softmax(safe_cosine / TEMPERATURE, dim=-1)
        weights = torch.where(has_support, weights, torch.zeros_like(weights))
        retrieved = weights @ values
        degree5_scores = torch.relu(cosine).pow(DEGREE) * torch.relu(alphas)[None, :]
        degree5_scores = degree5_scores.masked_fill(~active_mask, float("-inf"))
        finite_scores = torch.where(
            torch.isfinite(degree5_scores),
            degree5_scores,
            torch.zeros_like(degree5_scores),
        )
        maximum = finite_scores.max(dim=-1).values
        gain = torch.where(has_support[:, 0], maximum, torch.zeros_like(maximum))
        output = gain[:, None] * retrieved
        return {
            "scores": cosine_scores,
            "support_mask": support_mask,
            "support_size": support_mask.sum(dim=-1),
            "routed": has_support[:, 0],
            "weights": weights,
            "gain": gain,
            "retrieved": retrieved,
            "output": output,
            "prediction": (output @ head.T).argmax(dim=-1),
        }
    if kind == "hard_top1_cache":
        cosine_scores = cosine.masked_fill(~active_mask, float("-inf"))
        selected_cosine, selected_slot = cosine_scores.max(dim=-1)
        routed = selected_cosine >= TAU_COS
        support_mask = cosine_scores >= TAU_COS
        weights = torch.zeros_like(cosine_scores)
        weights.scatter_(1, selected_slot[:, None], routed[:, None].to(weights.dtype))
        retrieved = weights @ values
        gain = torch.where(
            routed,
            torch.relu(selected_cosine).pow(DEGREE)
            * torch.relu(alphas.index_select(0, selected_slot)),
            torch.zeros_like(selected_cosine),
        )
        output = gain[:, None] * retrieved
        return {
            "scores": cosine_scores,
            "support_mask": support_mask,
            "support_size": support_mask.sum(dim=-1),
            "routed": routed,
            "weights": weights,
            "gain": gain,
            "retrieved": retrieved,
            "output": output,
            "prediction": (output @ head.T).argmax(dim=-1),
        }
    raise ValueError(f"unknown cache kind: {kind}")


def support_histogram(support: torch.Tensor) -> dict[str, int]:
    unique, counts = torch.unique(support.cpu(), sorted=True, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def summarize(
    result: dict[str, torch.Tensor],
    labels: torch.Tensor | None,
    ratios: tuple[float, ...] | None,
) -> dict[str, Any]:
    prediction = result["prediction"]
    summary: dict[str, Any] = {
        "query_count": int(prediction.numel()),
        "routed_count": int(result["routed"].sum().item()),
        "support_size_histogram": support_histogram(result["support_size"]),
        "output_sha256": tensor_sha256(result["output"]),
        "prediction_sha256": tensor_sha256(prediction),
        "all_output_finite": bool(torch.isfinite(result["output"]).all()),
        "all_weight_finite": bool(torch.isfinite(result["weights"]).all()),
        "all_gain_finite": bool(torch.isfinite(result["gain"]).all()),
    }
    if labels is None:
        return summary
    correct = prediction.eq(labels)
    summary["correct_count"] = int(correct.sum().item())
    summary["accuracy"] = float(correct.float().mean().item())
    if ratios is not None:
        ratio_tensor = torch.tensor(ratios, dtype=torch.float64)
        per_ratio = {}
        for ratio in MIX_RATIOS:
            mask = ratio_tensor.eq(ratio)
            per_ratio[f"{ratio:.2f}"] = {
                "query_count": int(mask.sum().item()),
                "correct_count": int(correct[mask].sum().item()),
                "accuracy": float(correct[mask].float().mean().item()),
            }
        summary["per_ratio"] = per_ratio
    return summary


def exact_equivalence(
    hlm: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
) -> dict[str, Any]:
    fields = (
        "scores",
        "support_mask",
        "support_size",
        "routed",
        "weights",
        "gain",
        "retrieved",
        "output",
        "prediction",
    )
    equality = {field: bool(torch.equal(hlm[field], cache[field])) for field in fields}
    return {"fields": equality, "all_exact": all(equality.values())}


def max_abs_output_difference(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    return float((left["output"] - right["output"]).abs().max().item())


def evaluate_phase(
    memory: EditableHLM5Memory,
    queries: torch.Tensor,
    head: torch.Tensor,
    labels: torch.Tensor | None,
    ratios: tuple[float, ...] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    readers = {
        "hlm5_degree5": read_hlm5(memory, queries, head),
        "matched_degree5_cache": read_cache(
            "matched_degree5_cache",
            queries,
            memory.keys,
            memory.values,
            memory.alphas,
            memory.active,
            head,
        ),
        "cosine_soft_cache": read_cache(
            "cosine_soft_cache",
            queries,
            memory.keys,
            memory.values,
            memory.alphas,
            memory.active,
            head,
        ),
        "hard_top1_cache": read_cache(
            "hard_top1_cache",
            queries,
            memory.keys,
            memory.values,
            memory.alphas,
            memory.active,
            head,
        ),
    }
    report = {
        "readers": {
            name: summarize(result, labels, ratios) for name, result in readers.items()
        },
        "hlm_matched_degree5_equivalence": exact_equivalence(
            readers["hlm5_degree5"],
            readers["matched_degree5_cache"],
        ),
        "hlm_cosine_soft_max_abs_output_difference": max_abs_output_difference(
            readers["hlm5_degree5"], readers["cosine_soft_cache"]
        ),
    }
    return report, readers


def run_seed(seed: int) -> dict[str, Any]:
    state = build_seed(seed)
    memory = make_memory(state["keys"], state["values"])
    create, _ = evaluate_phase(
        memory,
        state["queries"],
        state["head"],
        state["labels"],
        state["ratios"],
    )

    shuffled = make_memory(state["keys"], rotate_pairs(state["values"]))
    shuffled_result = read_hlm5(shuffled, state["queries"], state["head"])
    shuffled_summary = summarize(
        shuffled_result,
        state["labels"],
        state["ratios"],
    )

    snapshots = []
    for pair in range(PAIRS):
        snapshots.append(memory.deactivate(2 * pair + 1))
    deactivate, _ = evaluate_phase(
        memory,
        state["queries"],
        state["head"],
        state["labels"],
        state["ratios"],
    )

    for snapshot in snapshots:
        memory.reactivate(snapshot)
    reactivate, _ = evaluate_phase(
        memory,
        state["queries"],
        state["head"],
        state["labels"],
        state["ratios"],
    )

    edited_values = rotate_pairs(state["values"])
    for slot, value in enumerate(edited_values):
        memory.edit_value(slot, value, value_scale=1.0, strength=1.0)
    edited_labels = (state["labels"] + PAIR_ROTATION) % PAIRS
    pair_edit, _ = evaluate_phase(
        memory,
        state["queries"],
        state["head"],
        edited_labels,
        state["ratios"],
    )

    off_support, off_readers = evaluate_phase(
        memory,
        state["unused_keys"],
        state["head"],
        None,
        None,
    )
    off_support["all_reader_outputs_bit_zero"] = all(
        bool(torch.equal(result["output"], torch.zeros_like(result["output"])))
        for result in off_readers.values()
    )

    return {
        "seed": seed,
        "key_permutation": [int(value) for value in state["key_permutation"].tolist()],
        "key_permutation_sha256": permutation_sha256(state["key_permutation"]),
        "value_permutation": [
            int(value) for value in state["value_permutation"].tolist()
        ],
        "value_permutation_sha256": permutation_sha256(state["value_permutation"]),
        "phases": {
            "create": create,
            "deactivate_one_share": deactivate,
            "reactivate": reactivate,
            "pair_edit": pair_edit,
            "off_support": off_support,
        },
        "shuffled_value_null": shuffled_summary,
    }


def aggregate_reader(seed_reports: list[dict[str, Any]], phase: str, reader: str) -> dict[str, Any]:
    summaries = [item["phases"][phase]["readers"][reader] for item in seed_reports]
    query_count = sum(item["query_count"] for item in summaries)
    correct_count = sum(item.get("correct_count", 0) for item in summaries)
    aggregate: dict[str, Any] = {
        "query_count": query_count,
        "correct_count": correct_count,
        "accuracy": correct_count / query_count if query_count else None,
    }
    if all("per_ratio" in item for item in summaries):
        per_ratio = {}
        for ratio in MIX_RATIOS:
            name = f"{ratio:.2f}"
            count = sum(item["per_ratio"][name]["query_count"] for item in summaries)
            correct = sum(item["per_ratio"][name]["correct_count"] for item in summaries)
            per_ratio[name] = {
                "query_count": count,
                "correct_count": correct,
                "accuracy": correct / count,
            }
        aggregate["per_ratio"] = per_ratio
    return aggregate


def aggregate_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    phases = {}
    for phase in ("create", "deactivate_one_share", "reactivate", "pair_edit"):
        phases[phase] = {
            reader: aggregate_reader(seed_reports, phase, reader)
            for reader in (
                "hlm5_degree5",
                "matched_degree5_cache",
                "cosine_soft_cache",
                "hard_top1_cache",
            )
        }
    shuffled_count = sum(item["shuffled_value_null"]["query_count"] for item in seed_reports)
    shuffled_correct = sum(
        item["shuffled_value_null"]["correct_count"] for item in seed_reports
    )
    phases["shuffled_value_null"] = {
        "query_count": shuffled_count,
        "correct_count": shuffled_correct,
        "accuracy": shuffled_correct / shuffled_count,
    }
    phases["off_support"] = {
        "query_count": sum(
            item["phases"]["off_support"]["readers"]["hlm5_degree5"]["query_count"]
            for item in seed_reports
        ),
        "all_reader_outputs_bit_zero": all(
            item["phases"]["off_support"]["all_reader_outputs_bit_zero"]
            for item in seed_reports
        ),
    }
    all_equivalent = all(
        item["phases"][phase]["hlm_matched_degree5_equivalence"]["all_exact"]
        for item in seed_reports
        for phase in (
            "create",
            "deactivate_one_share",
            "reactivate",
            "pair_edit",
            "off_support",
        )
    )
    return {
        "phases": phases,
        "all_hlm_matched_degree5_fields_bit_exact": all_equivalent,
        "max_hlm_cosine_soft_output_difference": max(
            item["phases"][phase]["hlm_cosine_soft_max_abs_output_difference"]
            for item in seed_reports
            for phase in (
                "create",
                "deactivate_one_share",
                "reactivate",
                "pair_edit",
                "off_support",
            )
        ),
    }


def validate(seed_reports: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, bool]:
    support_contract = True
    finite_contract = True
    for item in seed_reports:
        for phase, expected in (
            ("create", {"2": PAIRS * len(MIX_RATIOS)}),
            ("deactivate_one_share", {"1": PAIRS * len(MIX_RATIOS)}),
            ("reactivate", {"2": PAIRS * len(MIX_RATIOS)}),
            ("pair_edit", {"2": PAIRS * len(MIX_RATIOS)}),
            ("off_support", {"0": PAIRS}),
        ):
            for summary in item["phases"][phase]["readers"].values():
                support_contract &= summary["support_size_histogram"] == expected
                finite_contract &= bool(
                    summary["all_output_finite"]
                    and summary["all_weight_finite"]
                    and summary["all_gain_finite"]
                )
        support_contract &= item["shuffled_value_null"]["support_size_histogram"] == {
            "2": PAIRS * len(MIX_RATIOS)
        }
        finite_contract &= bool(
            item["shuffled_value_null"]["all_output_finite"]
            and item["shuffled_value_null"]["all_weight_finite"]
            and item["shuffled_value_null"]["all_gain_finite"]
        )
    permutations_valid = all(
        sorted(item["key_permutation"]) == list(range(DIM))
        and sorted(item["value_permutation"]) == list(range(DIM))
        for item in seed_reports
    )
    counts_valid = len(seed_reports) == len(SEEDS) and all(
        item["phases"]["create"]["readers"]["hlm5_degree5"]["query_count"]
        == PAIRS * len(MIX_RATIOS)
        for item in seed_reports
    )
    return {
        "permutations_bijective": permutations_valid,
        "support_cardinality_contract": bool(support_contract),
        "all_reported_numeric_tensors_finite": bool(finite_contract),
        "query_seed_phase_reader_counts_complete": counts_valid,
        "hlm_matched_degree5_bit_exact": aggregate[
            "all_hlm_matched_degree5_fields_bit_exact"
        ],
        "off_support_all_readers_bit_zero": aggregate["phases"]["off_support"][
            "all_reader_outputs_bit_zero"
        ],
    }


def classify(aggregate: dict[str, Any], validity: dict[str, bool]) -> tuple[str, dict[str, bool]]:
    create = aggregate["phases"]["create"]["hlm5_degree5"]
    reactivate = aggregate["phases"]["reactivate"]["hlm5_degree5"]
    pair_edit = aggregate["phases"]["pair_edit"]["hlm5_degree5"]
    hard_top1 = aggregate["phases"]["create"]["hard_top1_cache"]
    deactivate = aggregate["phases"]["deactivate_one_share"]["hlm5_degree5"]
    shuffled = aggregate["phases"]["shuffled_value_null"]

    balanced = all(
        phase["per_ratio"]["1.00"]["accuracy"] >= BALANCED_ACCURACY_MIN
        for phase in (create, reactivate, pair_edit)
    )
    controls = (
        hard_top1["accuracy"] <= ABLATION_ACCURACY_MAX
        and deactivate["accuracy"] <= ABLATION_ACCURACY_MAX
        and shuffled["accuracy"] <= ABLATION_ACCURACY_MAX
    )
    robust = all(
        phase["accuracy"] >= ROBUST_OVERALL_ACCURACY_MIN
        and all(
            item["accuracy"] >= ROBUST_PER_RATIO_ACCURACY_MIN
            for item in phase["per_ratio"].values()
        )
        for phase in (create, reactivate, pair_edit)
    )
    cosine = aggregate["phases"]["create"]["cosine_soft_cache"]
    bars = {
        "all_validity_gates": all(validity.values()),
        "balanced_create_reactivate_edit": balanced,
        "load_bearing_controls": controls,
        "robust_skew_and_lifecycle": robust,
        "degree5_beats_cosine_soft_by_0p10": (
            create["accuracy"] - cosine["accuracy"] >= DEGREE5_ADVANTAGE_MIN
        ),
    }
    if not bars["all_validity_gates"]:
        return "INVALID", bars
    if not balanced or not controls:
        return "NO_MULTI_SUPPORT_FUNCTION", bars
    if robust:
        return "ROBUST_MULTI_SUPPORT_CACHE_EQUIVALENT", bars
    return "NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT", bars


def verify_contract() -> dict[str, str]:
    hashes = {
        "protocol": sha256_file(DEFAULT_PROTOCOL),
        "initial_registration": sha256_file(DEFAULT_REGISTRATION),
        "execution_registration": sha256_file(DEFAULT_EXECUTION_RECEIPT),
        "tool": sha256_file(Path(__file__).resolve()),
        "test": sha256_file(DEFAULT_TEST),
    }
    require(hashes["protocol"] == EXPECTED_PROTOCOL_SHA256, "protocol hash mismatch")
    require(
        hashes["initial_registration"] == EXPECTED_REGISTRATION_SHA256,
        "initial registration hash mismatch",
    )
    for relative, expected in PINNED_LINEAGE.items():
        actual = sha256_file(ROOT / relative)
        require(actual == expected, f"pinned lineage changed: {relative}")
        hashes[relative] = actual
    reference = json.loads(
        (ROOT / "baselines" / "results" / "exact_address_lifecycle_reference.json")
        .read_text(encoding="utf-8")
    )
    require(reference.get("status") == PINNED_REFERENCE_STATUS, "wrong lineage status")
    require(
        reference.get("scientific_sha256") == PINNED_REFERENCE_SCIENTIFIC_SHA256,
        "wrong lineage scientific hash",
    )
    registration = json.loads(DEFAULT_REGISTRATION.read_text(encoding="utf-8"))
    constants = registration["locked_constants"]
    require(tuple(constants["seeds"]) == SEEDS, "seed contract mismatch")
    require(tuple(constants["mix_ratios"]) == MIX_RATIOS, "ratio contract mismatch")
    require(constants["dim"] == DIM and constants["pairs"] == PAIRS, "shape mismatch")
    require(constants["degree"] == DEGREE, "degree mismatch")
    require(constants["temperature"] == TEMPERATURE, "temperature mismatch")
    require(constants["tau_cos"] == TAU_COS, "tau mismatch")

    execution = json.loads(DEFAULT_EXECUTION_RECEIPT.read_text(encoding="utf-8"))
    require(
        execution.get("status") == "LOCKED_BEFORE_REGISTERED_ASSAY",
        "wrong execution registration status",
    )
    require(execution.get("exact_command") == EXACT_COMMAND, "wrong exact command")
    receipt_hashes = execution.get("hashes", {})
    for key in ("protocol", "initial_registration", "tool", "test"):
        require(receipt_hashes.get(key) == hashes[key], f"receipt {key} mismatch")
    require(
        execution.get("runtime", {}).get("torch") == torch.__version__,
        "registered torch version mismatch",
    )
    return hashes


def run_reference() -> dict[str, Any]:
    hashes = verify_contract()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    seed_reports = [run_seed(seed) for seed in SEEDS]
    aggregate = aggregate_reports(seed_reports)
    validity = validate(seed_reports, aggregate)
    status, bars = classify(aggregate, validity)
    report: dict[str, Any] = {
        "schema_version": 1,
        "measurement": "hlm5-multi-support-kernel-equivalence",
        "status": status,
        "scope": (
            "Deterministic CPU operator assay only; no checkpoint, LM forward, "
            "training, remote compute, or scaling."
        ),
        "theorem_scope": (
            "The support-masked HLM5 reader is exactly reproducible by an explicit "
            "cache with the same degree-5 kernel, support, temperature, gain, and values."
        ),
        "config": {
            "seeds": list(SEEDS),
            "dim": DIM,
            "pairs": PAIRS,
            "active_slots": ACTIVE_SLOTS,
            "capacity": CAPACITY,
            "degree": DEGREE,
            "temperature": TEMPERATURE,
            "tau_cos": TAU_COS,
            "support_threshold": SUPPORT_THRESHOLD,
            "mix_ratios": list(MIX_RATIOS),
            "dtype": "torch.float32",
            "device": "cpu",
        },
        "pinned_hashes": hashes,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "torch.float32",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "validity": validity,
        "binding_bars": bars,
        "aggregate": aggregate,
        "seeds": seed_reports,
        "claim_boundary": {
            "multi_support_function": status
            in {
                "ROBUST_MULTI_SUPPORT_CACHE_EQUIVALENT",
                "NARROW_MULTI_SUPPORT_CACHE_EQUIVALENT",
            },
            "hlm_specific_retrieval_advantage": False,
            "semantic_routing": False,
            "language_model_gain": False,
            "training_or_scaling_authorized": False,
        },
    }
    report["scientific_sha256"] = scientific_sha256(report)
    return report


def write_or_verify_identical(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == rendered, "existing result differs")
        return
    path.write_text(rendered, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    resolved = args.out.resolve()
    if resolved != DEFAULT_OUTPUT.resolve():
        parser.error(f"--out is locked to {DEFAULT_OUTPUT.resolve()}")
    args.out = resolved
    return args


def main() -> int:
    args = parse_args()
    report = run_reference()
    write_or_verify_identical(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "create_accuracy": report["aggregate"]["phases"]["create"][
                    "hlm5_degree5"
                ]["accuracy"],
                "matched_cache_bit_exact": report["aggregate"][
                    "all_hlm_matched_degree5_fields_bit_exact"
                ],
                "degree5_beats_cosine_soft": report["binding_bars"][
                    "degree5_beats_cosine_soft_by_0p10"
                ],
                "scientific_sha256": report["scientific_sha256"],
                "output": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
