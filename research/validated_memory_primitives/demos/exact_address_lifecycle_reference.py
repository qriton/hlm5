"""Runnable HLM5 exact-address lifecycle and matched-cache reference.

This is a small CPU mechanism replay of the E5-v3 surviving result. It imports
the hash-pinned E5-v3 implementation without modifying it, exercises the full
lifecycle, and requires degree-5 retrieval to remain bit-identical to the
matched cache at every comparable phase.

It is intentionally not a semantic router, language-model evaluation, or
evidence that polynomial retrieval beats a cache.
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
sys.path.insert(0, str(ROOT / "baselines"))

from hlm5_memory import EditableHLM5Memory  # noqa: E402
from run_address_conditioning_v1 import (  # noqa: E402
    CANONICAL_DOMAIN,
    OFF_SUPPORT_DOMAIN,
    canonical_id,
    evaluate_operator,
    export_lifecycle_bundle,
    memory_state_equal,
    restore_lifecycle_bundle,
    scientific_sha256,
    shake_address,
)


DEFAULT_OUTPUT = (
    ROOT / "baselines" / "results" / "exact_address_lifecycle_reference.json"
)
DIM = 64
CAPACITY = 8
TAU_COS = 0.95
DEGREE = 5
TEMPERATURE = 0.10
ENTITIES = ("Alpha Republic", "Beta Republic", "Gamma Republic")
PINNED_HASHES = {
    "baselines/run_address_conditioning_v1.py": (
        "558722271edae4f9f8b8ac09122d7c39d63c5fd40c70d876e66f3cc41f8ec9fe"
    ),
    "tests/test_address_conditioning_v1.py": (
        "92d18cb9d9e3b7188343de9f0410df9e178013dd423d1d015efb061d411934ae"
    ),
    "hlm5_memory.py": (
        "0210733ab1160ac073d44e5a9d7967bb3b985fc323a07efde925694ae56c346a"
    ),
    "baselines/results/address_conditioning_v1.json": (
        "a7c9b23e462f36755bda47bf11d0445a21a56ce03f471376929e41124e268eae"
    ),
}
PINNED_E5V3_SCIENTIFIC_SHA256 = (
    "b67b17ccca502f2f0b07f5561a533560eef9cf361b45a755a2eef9f845b4a3f0"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_pinned_e5v3() -> dict[str, Any]:
    for relative, expected in PINNED_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"pinned E5-v3 artifact changed: {relative}: {actual} != {expected}"
            )
    evidence = json.loads(
        (ROOT / "baselines" / "results" / "address_conditioning_v1.json")
        .read_text(encoding="utf-8")
    )
    if evidence.get("formal_verdict") != "CANONICAL_CACHE_EQUIVALENT":
        raise RuntimeError("pinned E5-v3 verdict is not cache-equivalent")
    if evidence.get("scientific_sha256") != PINNED_E5V3_SCIENTIFIC_SHA256:
        raise RuntimeError("pinned E5-v3 scientific digest changed")
    return {
        "formal_verdict": evidence["formal_verdict"],
        "scientific_sha256": evidence["scientific_sha256"],
        "raw_sha256": PINNED_HASHES["baselines/results/address_conditioning_v1.json"],
        "implementation_hashes": {
            key: value
            for key, value in PINNED_HASHES.items()
            if key != "baselines/results/address_conditioning_v1.json"
        },
    }


def make_memory(read_mode: str) -> EditableHLM5Memory:
    return EditableHLM5Memory(
        dim=DIM,
        memory_size=CAPACITY,
        degree=DEGREE,
        temperature=TEMPERATURE,
        read_mode=read_mode,
        support_threshold=TAU_COS**DEGREE if read_mode == "support_masked" else None,
        learnable_memory=False,
    )


def canonical_addresses() -> torch.Tensor:
    return torch.stack(
        [shake_address(canonical_id(entity), DIM, CANONICAL_DOMAIN) for entity in ENTITIES]
    )


def value_vectors(version: str) -> torch.Tensor:
    return torch.stack(
        [
            shake_address(
                f"{version}|{canonical_id(entity)}",
                DIM,
                "hlm5-exact-address-value-v1",
            )
            for entity in ENTITIES
        ]
    )


def inject_pair(
    degree5: EditableHLM5Memory,
    cache: EditableHLM5Memory,
    addresses: torch.Tensor,
    values: torch.Tensor,
) -> tuple[list[int], list[int]]:
    slots: dict[str, list[int]] = {"degree5": [], "cache": []}
    for index, (address, value) in enumerate(zip(addresses, values)):
        label = f"canonical:{canonical_id(ENTITIES[index])}"
        slots["degree5"].append(
            degree5.inject(address, value=value, strength=1.0, label=label)
        )
        slots["cache"].append(
            cache.inject(address, value=value, strength=1.0, label=label)
        )
    if slots["degree5"] != slots["cache"]:
        raise RuntimeError("matched readers assigned different slots")
    return slots["degree5"], slots["cache"]


def evaluate_pair(
    phase: str,
    degree5: EditableHLM5Memory,
    cache: EditableHLM5Memory,
    queries: torch.Tensor,
) -> dict[str, Any]:
    hidden = torch.zeros(queries.shape[0], DIM)
    head = torch.eye(DIM)
    degree5_result = evaluate_operator(
        degree5,
        queries,
        hidden,
        head,
        "d5",
        TAU_COS,
        max(1, queries.shape[0]),
    )
    cache_result = evaluate_operator(
        cache,
        queries,
        hidden,
        head,
        "cache",
        TAU_COS,
        max(1, queries.shape[0]),
    )
    compared_fields = (
        "routed",
        "selected_slot",
        "support_size",
        "selected_gain",
        "output",
        "prediction",
    )
    field_equal = {
        field: bool(torch.equal(degree5_result[field], cache_result[field]))
        for field in compared_fields
    }
    if not all(field_equal.values()):
        raise RuntimeError(f"degree5/cache mismatch at {phase}: {field_equal}")
    if degree5_result["signature"] != cache_result["signature"]:
        raise RuntimeError(f"degree5/cache signature mismatch at {phase}")
    return {
        "phase": phase,
        "query_count": int(queries.shape[0]),
        "routed": [bool(value) for value in degree5_result["routed"].tolist()],
        "selected_slots": [
            int(value) for value in degree5_result["selected_slot"].tolist()
        ],
        "support_sizes": [
            int(value) for value in degree5_result["support_size"].tolist()
        ],
        "output_sha256": tensor_sha256(degree5_result["output"]),
        "signature": degree5_result["signature"],
        "degree5_cache_bit_exact": True,
        "field_equal": field_equal,
    }


def run_reference() -> dict[str, Any]:
    torch.manual_seed(20260810)
    pinned = verify_pinned_e5v3()
    addresses = canonical_addresses()
    initial_values = value_vectors("create")
    edited_values = value_vectors("edit")
    off_queries = torch.stack(
        [
            shake_address(
                f"unregistered-{index}",
                DIM,
                OFF_SUPPORT_DOMAIN,
            )
            for index in range(3)
        ]
    )
    gram = addresses @ addresses.T
    off_diagonal = gram - torch.eye(len(ENTITIES))
    max_cross_cosine = float(off_diagonal.abs().max().item())
    if max_cross_cosine >= TAU_COS:
        raise RuntimeError("canonical reference addresses are not uniquely supported")

    degree5 = make_memory("support_masked")
    cache = make_memory("hard_top1")
    degree5_slots, cache_slots = inject_pair(
        degree5,
        cache,
        addresses,
        initial_values,
    )
    phases = [evaluate_pair("create", degree5, cache, addresses)]

    edit_index = 1
    degree5.edit_value(degree5_slots[edit_index], edited_values[edit_index])
    cache.edit_value(cache_slots[edit_index], edited_values[edit_index])
    phases.append(evaluate_pair("edit", degree5, cache, addresses))

    deactivate_index = 0
    degree5_snapshot = degree5.deactivate(degree5_slots[deactivate_index])
    cache_snapshot = cache.deactivate(cache_slots[deactivate_index])
    phases.append(evaluate_pair("deactivate", degree5, cache, addresses))
    if phases[-1]["routed"][deactivate_index]:
        raise RuntimeError("deactivated canonical address still routed")

    degree5_blob = export_lifecycle_bundle(
        degree5,
        {deactivate_index: degree5_snapshot},
        address_mode="canonical",
        address_domain=CANONICAL_DOMAIN,
    )
    cache_blob = export_lifecycle_bundle(
        cache,
        {deactivate_index: cache_snapshot},
        address_mode="canonical",
        address_domain=CANONICAL_DOMAIN,
    )
    restored_degree5, degree5_snapshots, degree5_metadata = restore_lifecycle_bundle(
        degree5_blob,
        torch.device("cpu"),
    )
    restored_cache, cache_snapshots, cache_metadata = restore_lifecycle_bundle(
        cache_blob,
        torch.device("cpu"),
    )
    if not memory_state_equal(degree5, restored_degree5):
        raise RuntimeError("degree5 serialized-bundle state mismatch")
    if not memory_state_equal(cache, restored_cache):
        raise RuntimeError("cache serialized-bundle state mismatch")
    if degree5_metadata != cache_metadata:
        raise RuntimeError("matched bundle metadata differs")
    phases.append(
        evaluate_pair(
            "serialized_bundle_round_trip",
            restored_degree5,
            restored_cache,
            addresses,
        )
    )

    restored_degree5.reactivate(degree5_snapshots[deactivate_index])
    restored_cache.reactivate(cache_snapshots[deactivate_index])
    phases.append(
        evaluate_pair("reactivate", restored_degree5, restored_cache, addresses)
    )
    if not phases[-1]["routed"][deactivate_index]:
        raise RuntimeError("reactivated canonical address did not route")

    delete_index = 2
    restored_degree5.hard_delete(degree5_slots[delete_index], rerandomize=False)
    restored_cache.hard_delete(cache_slots[delete_index], rerandomize=False)
    phases.append(
        evaluate_pair("hard_delete", restored_degree5, restored_cache, addresses)
    )
    if phases[-1]["routed"][delete_index]:
        raise RuntimeError("hard-deleted canonical address still routed")

    off_phase = evaluate_pair(
        "off_support",
        restored_degree5,
        restored_cache,
        off_queries,
    )
    phases.append(off_phase)
    if any(off_phase["routed"]):
        raise RuntimeError("off-support reference address routed")

    report: dict[str, Any] = {
        "schema": "hlm5-exact-address-lifecycle-reference-v1",
        "status": "CACHE_EQUIVALENT_REFERENCE",
        "claim_boundary": (
            "Exact deterministic-address lifecycle only. No semantic routing, "
            "paraphrase generalization, language-model gain, HLM-specific retrieval "
            "advantage, or scaling authorization."
        ),
        "pinned_e5v3": pinned,
        "config": {
            "device": "cpu",
            "dim": DIM,
            "capacity": CAPACITY,
            "degree": DEGREE,
            "temperature": TEMPERATURE,
            "tau_cos": TAU_COS,
            "canonical_domain": CANONICAL_DOMAIN,
            "off_support_domain": OFF_SUPPORT_DOMAIN,
            "entities": list(ENTITIES),
            "canonical_ids": [canonical_id(entity) for entity in ENTITIES],
            "canonical_address_sha256": tensor_sha256(addresses),
            "max_cross_cosine": max_cross_cosine,
        },
        "lifecycle": {
            "create": True,
            "edit": True,
            "deactivate": True,
            "serialized_bundle_round_trip": True,
            "reactivate": True,
            "hard_delete": True,
            "off_support_gate": True,
        },
        "bundle": {
            "address_mode": degree5_metadata["address_mode"],
            "address_domain": degree5_metadata["address_domain"],
            "degree5_bytes": len(degree5_blob),
            "cache_bytes": len(cache_blob),
            "restored_state_exact": True,
        },
        "phases": phases,
        "all_phases_degree5_cache_bit_exact": all(
            phase["degree5_cache_bit_exact"] for phase in phases
        ),
    }
    report["scientific_sha256"] = scientific_sha256(report)
    return report


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    resolved_output = args.out.resolve()
    if resolved_output != DEFAULT_OUTPUT.resolve():
        parser.error(
            "the lifecycle reference output is fixed at "
            f"{DEFAULT_OUTPUT.resolve()} to protect the pinned E5-v3 artifacts"
        )
    args.out = resolved_output
    return args


def main() -> int:
    args = parse_args()
    report = run_reference()
    atomic_write_json(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "phases": [phase["phase"] for phase in report["phases"]],
                "degree5_cache_bit_exact": report[
                    "all_phases_degree5_cache_bit_exact"
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
