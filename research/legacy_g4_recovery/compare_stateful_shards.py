"""Compare two fixed-world FSDP state-complete checkpoints semantically.

The legacy G4 checkpoints contain ``ShardedTensor`` objects that can only be
unpickled after the original process group has been initialized.  This tool is
therefore launched on the same 96-rank topology as training.  It compares the
model, optimizer, metadata, and every captured RNG stream without relying on
``torch.save`` container bytes being reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


STATE_COMPLETE_SCHEMA = "hlm5-g4-state-complete-v1"
MAX_REPORTED_MISMATCHES = 32

try:
    from torch.distributed._shard.sharded_tensor import ShardedTensor
except ImportError:  # pragma: no cover - older/newer PyTorch compatibility
    ShardedTensor = ()  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-world", type=int, default=96)
    parser.add_argument("--expected-step", type=int, required=True,
                        help="zero-based payload step, e.g. 522499")
    return parser.parse_args()


def _key_token(value: Any) -> tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _feed(digest: "hashlib._Hash", token: str | bytes) -> None:
    data = token.encode("utf-8") if isinstance(token, str) else token
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    flat = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return memoryview(flat.numpy())


def _hash_tensor(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    _feed(digest, "tensor")
    _feed(digest, str(value.dtype))
    _feed(digest, str(value.layout))
    _feed(digest, json.dumps(list(value.shape), separators=(",", ":")))
    raw = _tensor_bytes(value)
    digest.update(struct.pack(">Q", raw.nbytes))
    digest.update(raw)


def _shard_metadata_record(metadata: Any) -> dict[str, Any]:
    return {
        "offsets": list(metadata.shard_offsets),
        "sizes": list(metadata.shard_sizes),
        "placement": str(metadata.placement),
    }


def _sharded_metadata_record(value: Any) -> dict[str, Any]:
    metadata = value.metadata()
    properties = metadata.tensor_properties
    return {
        "size": list(metadata.size),
        "dtype": str(properties.dtype),
        "layout": str(properties.layout),
        "requires_grad": bool(properties.requires_grad),
        "pin_memory": bool(properties.pin_memory),
        "memory_format": str(properties.memory_format),
        "shards": [_shard_metadata_record(item) for item in metadata.shards_metadata],
    }


class Comparison:
    """One-pass exact comparison plus canonical semantic digests."""

    def __init__(self) -> None:
        self.left_digest = hashlib.sha256()
        self.right_digest = hashlib.sha256()
        self.mismatch_count = 0
        self.mismatches: list[dict[str, Any]] = []

    def mismatch(self, path: str, reason: str, **details: Any) -> None:
        self.mismatch_count += 1
        if len(self.mismatches) < MAX_REPORTED_MISMATCHES:
            self.mismatches.append({"path": path, "reason": reason, **details})

    def _hash_scalar(self, digest: "hashlib._Hash", value: Any) -> None:
        _feed(digest, type(value).__qualname__)
        if value is None:
            _feed(digest, "null")
        elif isinstance(value, bool):
            _feed(digest, b"1" if value else b"0")
        elif isinstance(value, int):
            _feed(digest, str(value))
        elif isinstance(value, float):
            _feed(digest, struct.pack(">d", value))
        elif isinstance(value, str):
            _feed(digest, value)
        elif isinstance(value, bytes):
            _feed(digest, value)
        else:
            _feed(digest, repr(value))

    def compare(self, left: Any, right: Any, path: str = "$") -> None:
        if isinstance(left, ShardedTensor) or isinstance(right, ShardedTensor):
            self._compare_sharded(left, right, path)
            return
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            self._compare_tensor(left, right, path)
            return
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            self._compare_mapping(left, right, path)
            return
        if (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes, bytearray))
        ) or (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes, bytearray))
        ):
            self._compare_sequence(left, right, path)
            return

        self._hash_scalar(self.left_digest, left)
        self._hash_scalar(self.right_digest, right)
        if type(left) is not type(right):
            self.mismatch(path, "type", left=type(left).__qualname__,
                          right=type(right).__qualname__)
        elif isinstance(left, float) and math.isnan(left) and math.isnan(right):
            # NaN payload metadata is not expected, but identical IEEE values are
            # still represented by the digests and should not silently compare equal.
            self.mismatch(path, "nan_scalar")
        elif left != right:
            self.mismatch(path, "value", left=repr(left), right=repr(right))

    def _compare_tensor(self, left: Any, right: Any, path: str) -> None:
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            self._hash_scalar(self.left_digest, left)
            self._hash_scalar(self.right_digest, right)
            self.mismatch(path, "tensor_type", left=type(left).__qualname__,
                          right=type(right).__qualname__)
            return
        _hash_tensor(self.left_digest, left)
        _hash_tensor(self.right_digest, right)
        if left.dtype != right.dtype or left.layout != right.layout:
            self.mismatch(path, "tensor_metadata", left_dtype=str(left.dtype),
                          right_dtype=str(right.dtype), left_layout=str(left.layout),
                          right_layout=str(right.layout))
            return
        if tuple(left.shape) != tuple(right.shape):
            self.mismatch(path, "tensor_shape", left=list(left.shape),
                          right=list(right.shape))
            return
        if not torch.equal(left, right):
            details: dict[str, Any] = {
                "left_sha256": hashlib.sha256(_tensor_bytes(left)).hexdigest(),
                "right_sha256": hashlib.sha256(_tensor_bytes(right)).hexdigest(),
            }
            if left.is_floating_point() and left.numel():
                delta = (left.detach().float() - right.detach().float()).abs()
                details["max_abs_diff"] = float(delta.max().item())
            self.mismatch(path, "tensor_value", **details)

    def _compare_mapping(self, left: Any, right: Any, path: str) -> None:
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            self._hash_scalar(self.left_digest, left)
            self._hash_scalar(self.right_digest, right)
            self.mismatch(path, "mapping_type", left=type(left).__qualname__,
                          right=type(right).__qualname__)
            return
        _feed(self.left_digest, f"mapping:{type(left).__qualname__}")
        _feed(self.right_digest, f"mapping:{type(right).__qualname__}")
        if type(left) is not type(right):
            self.mismatch(path, "mapping_concrete_type",
                          left=type(left).__qualname__, right=type(right).__qualname__)
        left_keys = sorted(left, key=_key_token)
        right_keys = sorted(right, key=_key_token)
        left_tokens = [_key_token(item) for item in left_keys]
        right_tokens = [_key_token(item) for item in right_keys]
        _feed(self.left_digest, json.dumps(left_tokens))
        _feed(self.right_digest, json.dumps(right_tokens))
        if left_tokens != right_tokens:
            self.mismatch(path, "mapping_keys", left=left_tokens, right=right_tokens)
        all_tokens = sorted(set(left_tokens) | set(right_tokens))
        left_by_token = {_key_token(key): key for key in left_keys}
        right_by_token = {_key_token(key): key for key in right_keys}
        for token in all_tokens:
            if token not in left_by_token or token not in right_by_token:
                continue
            key_left = left_by_token[token]
            key_right = right_by_token[token]
            self.compare(left[key_left], right[key_right], f"{path}[{key_left!r}]")

    def _compare_sequence(self, left: Any, right: Any, path: str) -> None:
        valid_types = (list, tuple, torch.Size)
        if not isinstance(left, valid_types) or not isinstance(right, valid_types):
            self._hash_scalar(self.left_digest, left)
            self._hash_scalar(self.right_digest, right)
            self.mismatch(path, "sequence_type", left=type(left).__qualname__,
                          right=type(right).__qualname__)
            return
        _feed(self.left_digest, f"sequence:{type(left).__qualname__}:{len(left)}")
        _feed(self.right_digest, f"sequence:{type(right).__qualname__}:{len(right)}")
        if type(left) is not type(right) or len(left) != len(right):
            self.mismatch(path, "sequence_metadata", left_type=type(left).__qualname__,
                          right_type=type(right).__qualname__, left_len=len(left),
                          right_len=len(right))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            self.compare(left_item, right_item, f"{path}[{index}]")

    def _compare_sharded(self, left: Any, right: Any, path: str) -> None:
        if not isinstance(left, ShardedTensor) or not isinstance(right, ShardedTensor):
            self._hash_scalar(self.left_digest, left)
            self._hash_scalar(self.right_digest, right)
            self.mismatch(path, "sharded_tensor_type", left=type(left).__qualname__,
                          right=type(right).__qualname__)
            return
        left_metadata = _sharded_metadata_record(left)
        right_metadata = _sharded_metadata_record(right)
        _feed(self.left_digest, "sharded_tensor")
        _feed(self.right_digest, "sharded_tensor")
        _feed(self.left_digest, json.dumps(left_metadata, sort_keys=True))
        _feed(self.right_digest, json.dumps(right_metadata, sort_keys=True))
        if left_metadata != right_metadata:
            self.mismatch(path, "sharded_metadata", left=left_metadata,
                          right=right_metadata)
        left_shards = left.local_shards()
        right_shards = right.local_shards()
        if len(left_shards) != len(right_shards):
            self.mismatch(path, "local_shard_count", left=len(left_shards),
                          right=len(right_shards))
        for index, (left_shard, right_shard) in enumerate(zip(left_shards, right_shards)):
            left_local = _shard_metadata_record(left_shard.metadata)
            right_local = _shard_metadata_record(right_shard.metadata)
            _feed(self.left_digest, json.dumps(left_local, sort_keys=True))
            _feed(self.right_digest, json.dumps(right_local, sort_keys=True))
            if left_local != right_local:
                self.mismatch(f"{path}.local_shards[{index}]", "local_shard_metadata",
                              left=left_local, right=right_local)
            self._compare_tensor(left_shard.tensor, right_shard.tensor,
                                 f"{path}.local_shards[{index}].tensor")

    def result(self) -> dict[str, Any]:
        return {
            "exact": self.mismatch_count == 0,
            "mismatch_count": self.mismatch_count,
            "mismatches": self.mismatches,
            "left_semantic_sha256": self.left_digest.hexdigest(),
            "right_semantic_sha256": self.right_digest.hexdigest(),
        }


def validate_payload(payload: Any, rank: int, world: int, expected_step: int) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint payload is not a mapping")
    expected = {
        "sharded": True,
        "rank": rank,
        "world_size": world,
        "step": expected_step,
        "checkpoint_schema": STATE_COMPLETE_SCHEMA,
        "optimizer_state_saved": True,
        "runtime_state_saved": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"payload field mismatch rank={rank} {key}={payload.get(key)!r} "
                f"expected={value!r}")
    for key in ("model", "opt", "runtime_state"):
        if key not in payload:
            raise RuntimeError(f"payload missing {key!r} on rank {rank}")
    runtime = payload["runtime_state"]
    if not isinstance(runtime, Mapping) or runtime.get("schema") != STATE_COMPLETE_SCHEMA:
        raise RuntimeError(f"runtime-state schema mismatch on rank {rank}")
    for key in ("sample_generator_state", "torch_cpu_rng_state", "torch_cuda_rng_state"):
        if not isinstance(runtime.get(key), torch.Tensor):
            raise RuntimeError(f"runtime state missing tensor {key!r} on rank {rank}")


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    if world != args.expected_world:
        raise RuntimeError(f"world mismatch: live={world} expected={args.expected_world}")

    if world > 1:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)

    if rank == 0:
        args.report_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()

    report: dict[str, Any]
    local_exact = False
    try:
        left_path = args.left / f"shard_rank{rank}.pt"
        right_path = args.right / f"shard_rank{rank}.pt"
        left = torch.load(left_path, map_location="cpu", weights_only=False)
        right = torch.load(right_path, map_location="cpu", weights_only=False)
        validate_payload(left, rank, world, args.expected_step)
        validate_payload(right, rank, world, args.expected_step)
        comparison = Comparison()
        comparison.compare(left, right)
        report = {
            "schema": "hlm5-g4-stateful-shard-comparison-rank-v1",
            "label": args.label,
            "rank": rank,
            "world_size": world,
            "expected_step": args.expected_step,
            **comparison.result(),
        }
        local_exact = bool(report["exact"])
    except Exception as exc:  # preserve a per-rank diagnostic before failing closed
        report = {
            "schema": "hlm5-g4-stateful-shard-comparison-rank-v1",
            "label": args.label,
            "rank": rank,
            "world_size": world,
            "expected_step": args.expected_step,
            "exact": False,
            "implementation_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    rank_report = args.report_dir / f"rank{rank:03d}.json"
    write_new_json(rank_report, report)
    ok_tensor = torch.tensor(
        [1 if local_exact else 0],
        device=(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"),
        dtype=torch.int32,
    )
    if world > 1:
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        dist.barrier()
    global_exact = bool(ok_tensor.item())

    if rank == 0:
        rank_reports = [
            json.loads((args.report_dir / f"rank{item:03d}.json").read_text(encoding="utf-8"))
            for item in range(world)
        ]
        implementation_errors = [
            item for item in rank_reports if "implementation_error" in item
        ]
        aggregate = {
            "schema": "hlm5-g4-stateful-shard-comparison-v1",
            "label": args.label,
            "status": (
                "IMPLEMENTATION_INVALID" if implementation_errors
                else "PASS_EXACT" if global_exact
                else "FAIL_MISMATCH"
            ),
            "exact": global_exact and not implementation_errors,
            "world_size": world,
            "expected_step": args.expected_step,
            "rank_report_count": len(rank_reports),
            "mismatching_ranks": [item["rank"] for item in rank_reports if not item["exact"]],
            "rank_reports": [
                {
                    key: item[key]
                    for key in (
                        "rank",
                        "exact",
                        "mismatch_count",
                        "left_semantic_sha256",
                        "right_semantic_sha256",
                        "implementation_error",
                    )
                    if key in item
                }
                for item in rank_reports
            ],
        }
        write_new_json(args.report_dir / "aggregate.json", aggregate)
        print(
            f"STATEFUL_COMPARISON_{aggregate['status']} label={args.label} "
            f"mismatching_ranks={len(aggregate['mismatching_ranks'])}",
            flush=True,
        )

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if any("implementation_error" in item for item in ([report] if world == 1 else [])):
        return 2
    return 0 if global_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
