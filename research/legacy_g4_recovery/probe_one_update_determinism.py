"""Localize the first divergent stage of one legacy G4 training update.

The probe restores the state-complete 522K checkpoint, executes exactly one
ordinary training update, and writes rank-local semantic digests at load,
sampling, forward, backward/reduction, clipping, and optimizer boundaries.  It
does not save a model checkpoint and it does not modify the source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F


STATE_COMPLETE_SCHEMA = "hlm5-g4-state-complete-v1"
SOURCE_STEP = 522000
SCHEDULE_HORIZON = 610000
WARMUP_STEPS = 2000
LEARNING_RATE = 2e-4
GRAD_CLIP = 1.0
CONTEXT = 1024
VOCAB = 65536
MEMORY_SIZE = 256
EXPECTED_WORLD = 96

try:
    from torch.distributed._shard.sharded_tensor import ShardedTensor
except ImportError:  # pragma: no cover - compatibility with other PyTorch releases
    ShardedTensor = ()  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-label", choices=["a", "b"], required=True)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--expected-world", type=int, default=EXPECTED_WORLD)
    parser.add_argument("--expected-step", type=int, default=SOURCE_STEP)
    return parser.parse_args()


def _feed(digest: "hashlib._Hash", token: str | bytes) -> None:
    data = token.encode("utf-8") if isinstance(token, str) else token
    digest.update(struct.pack(">Q", len(data)))
    digest.update(data)


def _tensor_bytes(value: torch.Tensor) -> memoryview:
    flat = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return memoryview(flat.numpy())


def _key_token(value: Any) -> tuple[str, str]:
    return type(value).__qualname__, repr(value)


def _shard_metadata(metadata: Any) -> dict[str, Any]:
    return {
        "offsets": list(metadata.shard_offsets),
        "sizes": list(metadata.shard_sizes),
        "placement": str(metadata.placement),
    }


def _update_semantic_digest(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, ShardedTensor):
        metadata = value.metadata()
        properties = metadata.tensor_properties
        _feed(digest, "sharded_tensor")
        _feed(
            digest,
            json.dumps(
                {
                    "size": list(metadata.size),
                    "dtype": str(properties.dtype),
                    "layout": str(properties.layout),
                    "requires_grad": bool(properties.requires_grad),
                    "pin_memory": bool(properties.pin_memory),
                    "memory_format": str(properties.memory_format),
                    "shards": [_shard_metadata(item) for item in metadata.shards_metadata],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        for shard in value.local_shards():
            _feed(
                digest,
                json.dumps(
                    _shard_metadata(shard.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            _update_semantic_digest(digest, shard.tensor)
        return
    if isinstance(value, torch.Tensor):
        _feed(digest, "tensor")
        _feed(digest, str(value.dtype))
        _feed(digest, str(value.layout))
        _feed(digest, json.dumps(list(value.shape), separators=(",", ":")))
        raw = _tensor_bytes(value)
        digest.update(struct.pack(">Q", raw.nbytes))
        digest.update(raw)
        return
    if isinstance(value, Mapping):
        keys = sorted(value, key=_key_token)
        _feed(digest, f"mapping:{type(value).__qualname__}")
        _feed(digest, json.dumps([_key_token(key) for key in keys]))
        for key in keys:
            _update_semantic_digest(digest, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        _feed(digest, f"sequence:{type(value).__qualname__}:{len(value)}")
        for item in value:
            _update_semantic_digest(digest, item)
        return
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


def semantic_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_semantic_digest(digest, value)
    return digest.hexdigest()


def named_tensor_sha256(items: Iterable[tuple[str, torch.Tensor | None]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        _feed(digest, name)
        if value is None:
            _feed(digest, "NONE")
        else:
            _update_semantic_digest(digest, value)
    return digest.hexdigest()


def model_sha256(model: torch.nn.Module) -> str:
    items = [
        (f"parameter:{name}", parameter.detach())
        for name, parameter in model.named_parameters()
    ]
    items.extend(
        (f"buffer:{name}", buffer.detach()) for name, buffer in model.named_buffers()
    )
    return named_tensor_sha256(items)


def gradient_sha256(model: torch.nn.Module) -> str:
    return named_tensor_sha256((name, parameter.grad) for name, parameter in model.named_parameters())


def runtime_record(generator: torch.Generator, device: str) -> dict[str, str]:
    record = {
        "sample_generator_sha256": semantic_sha256(generator.get_state()),
        "torch_cpu_rng_sha256": semantic_sha256(torch.get_rng_state()),
    }
    record["torch_cuda_rng_sha256"] = semantic_sha256(torch.cuda.get_rng_state(device))
    return record


def scheduled_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, SCHEDULE_HORIZON - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def validate_source_payload(payload: Any, rank: int, world: int, expected_step: int) -> None:
    expected = {
        "sharded": True,
        "rank": rank,
        "world_size": world,
        "step": expected_step - 1,
        "checkpoint_schema": STATE_COMPLETE_SCHEMA,
        "optimizer_state_saved": True,
        "runtime_state_saved": True,
    }
    if not isinstance(payload, Mapping):
        raise RuntimeError("source payload is not a mapping")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"source field mismatch rank={rank} {key}={payload.get(key)!r} "
                f"expected={value!r}")
    for key in ("model", "opt", "runtime_state"):
        if key not in payload:
            raise RuntimeError(f"state-complete source is missing {key!r} on rank {rank}")


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    if world != args.expected_world or world != EXPECTED_WORLD:
        raise RuntimeError(f"world mismatch live={world} expected={EXPECTED_WORLD}")
    if args.expected_step != SOURCE_STEP:
        raise RuntimeError(f"source step is frozen at {SOURCE_STEP}")

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = f"cuda:{local_rank}"

    # Imports are delayed so CPU unit tests can exercise the hashing and
    # comparison helpers without the remote-only legacy model module.
    from train_hlm5_lm_fineweb_stateful import (
        fsdp_local_state_dict_context,
        load_model_state_dict_with_fallback,
        open_mmap,
        restore_runtime_state,
        sample_batch,
    )
    from hlm5_lm import HLM5LM, lm_config

    from functools import partial
    import torch.nn as nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    train_buffer = open_mmap(args.train_bin)
    config = lm_config("huge3b")
    config["max_len"] = CONTEXT
    config["memory_layer"] = True
    config["memory_size"] = MEMORY_SIZE
    torch.manual_seed(7)
    raw_model = HLM5LM(vocab=VOCAB, **config).to(device)

    decay, no_decay = [], []
    for name, parameter in raw_model.named_parameters():
        (no_decay if parameter.ndim < 2 or "emb" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )

    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )
    model = FSDP(
        raw_model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        ),
        device_id=local_rank,
        use_orig_params=True,
        limit_all_gathers=True,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda module: isinstance(module, nn.TransformerEncoderLayer),
    )

    shard_path = args.source_dir / f"shard_rank{rank}.pt"
    source = torch.load(shard_path, map_location="cpu", weights_only=False)
    validate_source_payload(source, rank, world, args.expected_step)
    with fsdp_local_state_dict_context(model, save_opt=True):
        _label, _normalized = load_model_state_dict_with_fallback(
            model,
            source["model"],
            (print if rank == 0 else lambda _message: None),
        )
        optimizer.load_state_dict(source["opt"])

    generator = torch.Generator().manual_seed(1234 + rank)
    restore_runtime_state(source["runtime_state"], generator, device)
    dist.barrier()
    torch.cuda.synchronize(device)

    stages: dict[str, Any] = {}
    stages["loaded"] = {
        "model_sha256": model_sha256(model),
        "optimizer_sha256": semantic_sha256(optimizer.state_dict()),
        "runtime": runtime_record(generator, device),
    }

    x, y = sample_batch(train_buffer, 1, CONTEXT, device, generator)
    torch.cuda.synchronize(device)
    stages["sampled"] = {
        "x_sha256": semantic_sha256(x),
        "y_sha256": semantic_sha256(y),
        "runtime": runtime_record(generator, device),
    }

    for group in optimizer.param_groups:
        group["lr"] = scheduled_lr(SOURCE_STEP)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    positions = torch.tensor([0, CONTEXT // 2, CONTEXT - 1], device=device)
    vocabulary = torch.arange(0, VOCAB, 257, device=device)
    logits_probe = logits[0].index_select(0, positions).index_select(1, vocabulary)
    torch.cuda.synchronize(device)
    stages["forward"] = {
        "loss_sha256": semantic_sha256(loss),
        "loss": float(loss.detach().float().item()),
        "logits_probe_sha256": semantic_sha256(logits_probe),
        "logits_probe_shape": list(logits_probe.shape),
        "runtime": runtime_record(generator, device),
    }

    loss.backward()
    torch.cuda.synchronize(device)
    stages["backward"] = {
        "gradient_sha256": gradient_sha256(model),
        "runtime": runtime_record(generator, device),
    }

    gradient_norm = model.clip_grad_norm_(GRAD_CLIP)
    torch.cuda.synchronize(device)
    stages["clipped"] = {
        "gradient_sha256": gradient_sha256(model),
        "gradient_norm_sha256": semantic_sha256(gradient_norm),
        "gradient_norm": float(gradient_norm.detach().float().item()),
        "runtime": runtime_record(generator, device),
    }

    optimizer.step()
    torch.cuda.synchronize(device)
    stages["stepped"] = {
        "model_sha256": model_sha256(model),
        "optimizer_sha256": semantic_sha256(optimizer.state_dict()),
        "runtime": runtime_record(generator, device),
    }

    if rank == 0:
        args.report_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    report = {
        "schema": "hlm5-g4-one-update-determinism-rank-v1",
        "run_label": args.run_label,
        "rank": rank,
        "world_size": world,
        "local_rank": local_rank,
        "hostname": platform.node(),
        "device_name": torch.cuda.get_device_name(local_rank),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "source_step": SOURCE_STEP,
        "scheduled_lr": scheduled_lr(SOURCE_STEP),
        "stages": stages,
    }
    write_new_json(args.report_dir / f"rank{rank:03d}.json", report)
    dist.barrier()
    if rank == 0:
        write_new_json(
            args.report_dir / "complete.json",
            {
                "schema": "hlm5-g4-one-update-determinism-complete-v1",
                "run_label": args.run_label,
                "rank_reports": world,
                "source_step": SOURCE_STEP,
            },
        )
        print(f"ONE_UPDATE_PROBE_COMPLETE run={args.run_label} ranks={world}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
