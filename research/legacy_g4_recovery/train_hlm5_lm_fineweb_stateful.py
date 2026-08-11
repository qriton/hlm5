"""Phase-2 trunk: train hlm5_lm.HLM5LM on the pretokenized FineWeb 65K mmap (Leonardo).

Self-contained DDP trainer (no HLM2-repo imports): reads the HLM3 binary format
(16-byte header: magic 'HLM3' | uint32 version | uint64 n_tokens; uint16 token stream),
samples random ctx windows per rank (no DistributedSampler -- the 5B-index OOM lesson),
bf16 autocast, cosine LR over the FULL horizon + warmup, grad clip, and -- the Phase-1
lesson -- NO weight decay on embeddings/norms (WD collapses rarely-updated embedding
rows and destroys the trunk's editability floor).

Single-node usage (Leonardo, srun sets SLURM_PROCID/NTASKS/LOCALID):
  srun python train_hlm5_lm_fineweb.py --steps 30000 --out-dir runs/hlm5_base_fineweb
Runs single-process when SLURM env is absent (local smoke).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# The registered remote layout keeps the exact legacy model at repository root
# while this recovery trainer lives under research/legacy_g4_recovery.
LEGACY_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(LEGACY_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_REPO_ROOT))

from hlm5_lm import HLM5LM, lm_config  # noqa: E402

FINEWEB_TRAIN = "/leonardo_scratch/large/userexternal/mdima000/hlm3/data/fineweb/fineweb_22b_train.bin"
FINEWEB_VAL = "/leonardo_scratch/large/userexternal/mdima000/hlm3/data/fineweb/fineweb_22b_val.bin"
STATE_COMPLETE_SCHEMA = "hlm5-g4-state-complete-v1"
LEGACY_MODEL_ONLY_SOURCE_SHA256 = (
    "f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9")


def open_mmap(path: str) -> np.memmap:
    with open(path, "rb") as f:
        header = f.read(16)
    magic, _version, n_tokens = header[:4], *struct.unpack("<IQ", header[4:16])
    if magic != b"HLM3":
        raise ValueError(f"{path}: bad magic {magic!r}, expected b'HLM3'")
    buf = np.memmap(path, dtype=np.uint16, mode="r", offset=16)
    if buf.shape[0] != n_tokens:
        raise ValueError(f"{path}: header says {n_tokens} tokens, file has {buf.shape[0]}")
    return buf


def sample_batch(buf: np.memmap, batch: int, ctx: int, device, generator):
    idx = torch.randint(0, buf.shape[0] - ctx - 1, (batch,), generator=generator)
    rows = np.stack([buf[i:i + ctx + 1] for i in idx.tolist()]).astype(np.int64)
    win = torch.from_numpy(rows).to(device, non_blocking=True)
    return win[:, :-1], win[:, 1:]


def advance_sampling_generator(generator: torch.Generator, draws: int, batch: int,
                               upper_bound: int) -> None:
    """Reconstruct the sampling stream of a known model-only legacy segment."""
    if draws < 0:
        raise ValueError("sampling RNG offset must be non-negative")
    if batch <= 0 or upper_bound <= 0:
        raise ValueError("sampling RNG replay requires positive batch and upper bound")
    for _ in range(draws):
        torch.randint(0, upper_bound, (batch,), generator=generator)


def restart_lr_multiplier(offset: int, steps: int, initial_factor: float) -> float:
    """Return the registered linear warm-restart multiplier."""
    if steps <= 0 or offset < 0 or offset >= steps:
        return 1.0
    denominator = max(1, steps - 1)
    progress = offset / denominator
    return initial_factor + (1.0 - initial_factor) * progress


def capture_runtime_state(generator: torch.Generator, device) -> dict:
    """Capture every RNG stream used by this trainer on the current rank."""
    state = {
        "schema": STATE_COMPLETE_SCHEMA,
        "sample_generator_state": generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": None,
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        state["torch_cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return state


def restore_runtime_state(state: dict, generator: torch.Generator, device) -> None:
    """Restore state captured by :func:`capture_runtime_state`, failing closed."""
    if not isinstance(state, dict) or state.get("schema") != STATE_COMPLETE_SCHEMA:
        raise RuntimeError("checkpoint runtime-state schema is missing or unsupported")
    sample_state = state.get("sample_generator_state")
    cpu_state = state.get("torch_cpu_rng_state")
    if not isinstance(sample_state, torch.Tensor) or not isinstance(cpu_state, torch.Tensor):
        raise RuntimeError("checkpoint runtime state is missing CPU or sampling RNG tensors")
    generator.set_state(sample_state.cpu())
    torch.set_rng_state(cpu_state.cpu())
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        cuda_state = state.get("torch_cuda_rng_state")
        if not isinstance(cuda_state, torch.Tensor):
            raise RuntimeError("checkpoint runtime state is missing CUDA RNG tensor")
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


@torch.no_grad()
def val_ppl(model, buf: np.memmap, ctx: int, device, max_windows: int = 100,
            rank: int = 0, world: int = 1) -> float:
    """Validation PPL. Under FSDP all ranks must run forward (all_gather)."""
    model.eval()
    losses = []
    for w in range(max_windows):
        start = w * ctx
        if start + ctx + 1 > buf.shape[0]:
            break
        win = torch.from_numpy(buf[start:start + ctx + 1].astype(np.int64))[None].to(device)
        logits, _ = model(win[:, :-1])
        if rank == 0:
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                win[:, 1:].reshape(-1))))
    model.train()
    if world > 1:
        if rank == 0:
            ppl = math.exp(sum(losses) / len(losses)) if losses else float("inf")
            out = torch.tensor([ppl], device=device, dtype=torch.float64)
        else:
            out = torch.tensor([0.0], device=device, dtype=torch.float64)
        dist.broadcast(out, src=0)
        return float(out.item())
    return math.exp(sum(losses) / len(losses)) if losses else float("inf")


def find_latest_checkpoint(out_dir: Path):
    """Return (kind, path, step): kind is 'sharded' | 'pt'."""
    markers = sorted(
        out_dir.glob("ckpt_step*.sharded"),
        key=lambda p: int(p.stem.replace("ckpt_step", "").replace(".sharded", "")),
    )
    if markers:
        step = int(markers[-1].stem.replace("ckpt_step", "").replace(".sharded", ""))
        return "sharded", out_dir / f"ckpt_step{step}", step
    cks = sorted(out_dir.glob("ckpt_step*.pt"),
                 key=lambda p: int(p.stem.replace("ckpt_step", "")))
    if cks:
        step = int(cks[-1].stem.replace("ckpt_step", ""))
        return "pt", cks[-1], step
    return None, None, None


def list_step_checkpoints(out_dir: Path):
    markers = sorted(out_dir.glob("ckpt_step*.sharded"),
                     key=lambda p: int(p.stem.replace("ckpt_step", "")))
    if markers:
        return markers
    return sorted(out_dir.glob("ckpt_step*.pt"),
                  key=lambda p: int(p.stem.replace("ckpt_step", "")))


def prune_sharded_checkpoints(out_dir: Path, keep: int, log) -> None:
    markers = sorted(out_dir.glob("ckpt_step*.sharded"),
                     key=lambda p: int(p.stem.replace("ckpt_step", "")))
    if keep <= 0:
        old_markers = markers
    else:
        old_markers = markers[:-keep]
    for old_marker in old_markers:
        step = int(old_marker.stem.replace("ckpt_step", ""))
        old_dir = out_dir / f"ckpt_step{step}"
        try:
            if old_dir.exists():
                shutil.rmtree(old_dir)
                log(f"  pruned old sharded checkpoint {old_dir}")
            old_marker.unlink()
        except Exception as exc:
            log(f"WARNING: failed to prune sharded checkpoint {old_marker}: {exc}")


def fsdp_local_state_dict_context(model, save_opt: bool = False):
    """CPU-offloaded LOCAL_STATE_DICT context for portable FSDP shard save/load."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import (
        LocalOptimStateDictConfig,
        LocalStateDictConfig,
        StateDictType,
    )

    local_model_cfg = LocalStateDictConfig(offload_to_cpu=True)
    if save_opt:
        local_optim_cfg = LocalOptimStateDictConfig(offload_to_cpu=True)
        return FSDP.state_dict_type(
            model, StateDictType.LOCAL_STATE_DICT, local_model_cfg, local_optim_cfg)
    return FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, local_model_cfg)


def normalize_fsdp_local_state_dict_keys(state_dict: dict):
    """Strip saved internal FSDP wrapper path components before strict shard load."""
    normalized = {}
    changed = False
    for key, value in state_dict.items():
        key_text = str(key)
        new_key = ".".join(
            part for part in key_text.split(".") if part != "_fsdp_wrapped_module")
        if new_key in normalized:
            raise RuntimeError(
                f"FSDP local-state key collision while normalizing {key_text!r} -> {new_key!r}")
        normalized[new_key] = value
        changed = changed or new_key != key_text
    return normalized, changed


def remap_state_dict_keys(state_dict: dict, transform):
    remapped = {}
    changed = False
    for key, value in state_dict.items():
        key_text = str(key)
        new_key = transform(key_text)
        if new_key in remapped:
            raise RuntimeError(
                f"state key collision while remapping {key_text!r} -> {new_key!r}")
        remapped[new_key] = value
        changed = changed or new_key != key_text
    return remapped, changed


def strip_root_fsdp_wrapper_key(key: str) -> str:
    prefix = "_fsdp_wrapped_module."
    return key[len(prefix):] if key.startswith(prefix) else key


def strip_checkpoint_wrapper_key(key: str) -> str:
    return ".".join(part for part in key.split(".") if part != "_checkpoint_wrapped_module")


def strip_all_fsdp_and_checkpoint_wrapper_keys(key: str) -> str:
    return ".".join(
        part for part in key.split(".")
        if part not in {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"})


def align_sharded_tensor_local_requires_grad(state_dict: dict) -> int:
    """Repair legacy ShardedTensor shards whose local tensor flag mismatches metadata."""
    changed = 0
    for value in state_dict.values():
        local_shards = getattr(value, "local_shards", None)
        if not callable(local_shards):
            continue
        target_requires_grad = bool(getattr(value, "requires_grad", False))
        for shard in local_shards():
            tensor = getattr(shard, "tensor", None)
            if tensor is None or tensor.requires_grad == target_requires_grad:
                continue
            if tensor.is_floating_point() or tensor.is_complex():
                tensor.requires_grad_(target_requires_grad)
                changed += 1
    return changed


def first_error_lines(exc: BaseException, limit: int = 8) -> str:
    lines = str(exc).splitlines()
    if not lines:
        return type(exc).__name__
    return " | ".join(line.strip() for line in lines[:limit])[:2000]


def summarize_state_dict_error(exc: BaseException, limit: int = 1000) -> str:
    text = str(exc)
    if not text:
        return type(exc).__name__
    summaries = []
    for label in ("Missing key(s)", "Unexpected key(s)"):
        marker = f"{label} in state_dict:"
        if marker not in text:
            continue
        after = text.split(marker, 1)[1]
        after = after.split("\n", 1)[0]
        keys = re.findall(r'"([^"]+)"', after)
        summaries.append(f"{label}={len(keys)} sample={keys[:3]}")
    if summaries:
        return "; ".join(summaries)[:limit]
    return first_error_lines(exc, limit=4)[:limit]


def set_module_requires_grad(module, requires_grad: bool):
    states = []
    seen = set()

    def add_param(param):
        if param is None or id(param) in seen:
            return
        seen.add(id(param))
        states.append((param, param.requires_grad))
        if param.requires_grad != requires_grad:
            param.requires_grad_(requires_grad)

    for param in module.parameters():
        add_param(param)
    for submodule in module.modules():
        add_param(getattr(submodule, "_flat_param", None))
    return states


def restore_module_requires_grad(states) -> None:
    for param, requires_grad in states:
        if param.requires_grad != requires_grad:
            param.requires_grad_(requires_grad)


def param_probe(module, limit: int = 16):
    pieces = []
    seen = 0
    for param in module.parameters():
        if param.numel() == 0:
            continue
        flat = param.detach().flatten()
        take = min(limit - seen, flat.numel())
        if take > 0:
            pieces.append(flat[:take].float().cpu().clone())
            seen += take
        if seen >= limit:
            break
    if not pieces:
        return None
    return torch.cat(pieces)


def load_model_state_dict_with_fallback(model, state_dict: dict, log):
    """Load an FSDP local state dict across wrapper-key variants."""
    aligned_shards = align_sharded_tensor_local_requires_grad(state_dict)
    if aligned_shards:
        log(f"aligned requires_grad on {aligned_shards} FSDP local shard tensors")

    candidates = []
    for label, transform, used_normalized in (
            ("strip-root-fsdp", strip_root_fsdp_wrapper_key, True),
            ("strip-checkpoint-wrapper", strip_checkpoint_wrapper_key, True),
            ("strip-all-fsdp-checkpoint-wrappers",
             strip_all_fsdp_and_checkpoint_wrapper_keys, True)):
        remapped, _ = remap_state_dict_keys(state_dict, transform)
        candidates.append((label, remapped, used_normalized))
    normalized, changed = normalize_fsdp_local_state_dict_keys(state_dict)
    if changed:
        candidates.append(("strip-all-fsdp-wrappers", normalized, True))
    candidates.append(("raw", state_dict, False))
    log("FSDP local-state load candidates: " + ", ".join(
        label for label, _, _ in candidates))

    grad_states = set_module_requires_grad(model, False)
    if any(old for _, old in grad_states):
        log(f"temporarily disabled requires_grad on {len(grad_states)} FSDP params")
    try:
        expected_keys = None
        expected_key_error = None
        try:
            # FSDP LOCAL_STATE_DICT builds detached ShardedTensor local shards on
            # this torch build; temporarily matching requires_grad avoids its
            # metadata self-check failure during local-state load.
            expected_keys = set(model.state_dict(keep_vars=True).keys())
        except Exception as exc:
            expected_key_error = first_error_lines(exc, limit=2)
            log(f"FSDP local-state key scan skipped: {expected_key_error}")

        key_errors = []
        load_errors = []
        for label, candidate, used_normalized in candidates:
            if expected_keys is not None:
                candidate_keys = set(candidate.keys())
                missing = expected_keys - candidate_keys
                unexpected = candidate_keys - expected_keys
                if missing or unexpected:
                    missing_sample = list(sorted(missing))[:2]
                    unexpected_sample = list(sorted(unexpected))[:2]
                    key_errors.append(
                        f"{label}: missing={len(missing)} unexpected={len(unexpected)} "
                        f"missing_sample={missing_sample} unexpected_sample={unexpected_sample}")
                    continue
            try:
                before_probe = param_probe(model)
                model.load_state_dict(candidate)
                if key_errors or expected_key_error or load_errors:
                    log(f"FSDP local-state load fallback succeeded with {label} keys")
                return label, used_normalized
            except Exception as exc:
                first_line = first_error_lines(exc)
                if expected_keys is not None:
                    candidate_sample = list(sorted(str(key) for key in candidate.keys()))[:3]
                    if "Unexpected key(s)" in str(exc):
                        try:
                            result = model.load_state_dict(candidate, strict=False)
                        except Exception:
                            raise RuntimeError(
                                f"FSDP local-state load failed after key match ({label}): "
                                f"candidate_sample={candidate_sample} {first_line}") from None
                        after_probe = param_probe(model)
                        if (before_probe is not None and after_probe is not None
                                and torch.equal(before_probe, after_probe)):
                            raise RuntimeError(
                                f"FSDP local-state strict=False fallback for {label} did not "
                                "change the model probe") from None
                        missing = len(getattr(result, "missing_keys", []))
                        unexpected = len(getattr(result, "unexpected_keys", []))
                        log(
                            f"FSDP local-state strict=False fallback succeeded with {label} "
                            f"keys missing={missing} unexpected={unexpected}")
                        return label, used_normalized
                    raise RuntimeError(
                        f"FSDP local-state load failed after key match ({label}): "
                        f"candidate_sample={candidate_sample} {first_line}") from None
                candidate_sample = list(sorted(str(key) for key in candidate.keys()))[:3]
                load_errors.append(
                    f"{label}: candidate_sample={candidate_sample} "
                    f"{summarize_state_dict_error(exc)}")
        if expected_keys is None:
            raise RuntimeError(
                "FSDP local-state load failed after key scan failed: "
                f"{expected_key_error}; " + "; ".join(load_errors)) from None
        raise RuntimeError("FSDP local-state key match failed; " + "; ".join(key_errors)) from None
    finally:
        restore_module_requires_grad(grad_states)


def save_fsdp_sharded_checkpoint(model, optimizer, out_dir: Path, step: int, meta: dict,
                                 rank: int, world: int, save_opt: bool, runtime_state,
                                 log) -> bool:
    """LOCAL_STATE_DICT shard per rank — never FULL_STATE_DICT during training."""
    step_dir = out_dir / f"ckpt_step{step + 1}"
    if rank == 0:
        step_dir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    grad_states = set_module_requires_grad(model, False)
    if rank == 0 and any(old for _, old in grad_states):
        log(f"temporarily disabled requires_grad on {len(grad_states)} FSDP params for save")
    try:
        with fsdp_local_state_dict_context(model, save_opt=save_opt):
            payload = {
                "model": model.state_dict(),
                "sharded": True,
                "rank": rank,
                "world_size": world,
                **meta,
            }
            if save_opt:
                opt_state = optimizer.state_dict()
                if not opt_state.get("state"):
                    raise RuntimeError("refusing to write an empty optimizer state")
                if runtime_state is None:
                    raise RuntimeError("optimizer checkpoint requires rank-local runtime state")
                payload["opt"] = opt_state
                payload["runtime_state"] = runtime_state
    finally:
        restore_module_requires_grad(grad_states)

    ok = safe_torch_save(payload, step_dir / f"shard_rank{rank}.pt", log, f"shard_rank{rank}")
    all_ok = ok
    if world > 1:
        ok_device = (torch.device("cuda", torch.cuda.current_device())
                     if torch.cuda.is_available() else torch.device("cpu"))
        ok_t = torch.tensor([1 if ok else 0], device=ok_device, dtype=torch.int32)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        all_ok = bool(ok_t.item())
    marker_ok = all_ok
    if rank == 0 and all_ok:
        marker = out_dir / f"ckpt_step{step + 1}.sharded"
        try:
            marker.write_text(json.dumps({"step": step, "world_size": world, **meta}) + "\n",
                              encoding="utf-8")
            log(f"  saved sharded ckpt_step{step + 1} ({world} shards)")
        except Exception as exc:
            marker_ok = False
            log(f"WARNING: failed to write ckpt_step{step + 1}.sharded marker: {exc}")
    elif rank == 0:
        log(f"WARNING: not writing ckpt_step{step + 1}.sharded marker; "
            "at least one shard save failed")
    if world > 1:
        ok_device = (torch.device("cuda", torch.cuda.current_device())
                     if torch.cuda.is_available() else torch.device("cpu"))
        ok_t = torch.tensor([1 if marker_ok else 0], device=ok_device, dtype=torch.int32)
        dist.broadcast(ok_t, src=0)
        marker_ok = bool(ok_t.item())
        dist.barrier()
    return marker_ok


def safe_torch_save(obj, path: Path, log, label: str) -> bool:
    tmp = path.with_name(path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        torch.save(obj, tmp)
        tmp.replace(path)
        return True
    except Exception as exc:
        log(f"WARNING: failed to save {label} at {path}: {exc}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception as cleanup_exc:
            log(f"WARNING: failed to remove partial {label} temp {tmp}: {cleanup_exc}")
        return False


def prune_step_checkpoints(out_dir: Path, keep: int, log) -> None:
    cks = list_step_checkpoints(out_dir)
    for old in cks[:-keep]:
        try:
            old.unlink()
            log(f"  pruned old checkpoint {old}")
        except Exception as exc:
            log(f"WARNING: failed to prune old checkpoint {old}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", default=FINEWEB_TRAIN)
    ap.add_argument("--val-bin", default=FINEWEB_VAL)
    ap.add_argument("--size", default="base",
                    choices=["tiny", "small", "base", "huge1b", "huge3b"])
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16, help="per-GPU batch")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--nonfinite-policy", default="fail", choices=["fail", "skip"],
                    help="fail on non-finite loss/grad, or skip up to --max-nonfinite batches")
    ap.add_argument("--max-nonfinite", type=int, default=0,
                    help="maximum non-finite batches/gradients to skip when policy=skip")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--out-dir", default="runs/hlm5_base_fineweb")
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    ap.add_argument("--memory-layer", action="store_true",
                    help="matched-pair hybrid arm: train the HLM5 memory layer in-loop")
    ap.add_argument("--memory-size", type=int, default=256)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="per-layer activation checkpointing (required for huge3b on 64GB)")
    ap.add_argument("--fsdp", action="store_true",
                    help="FSDP FULL_SHARD instead of DDP (shards optimizer state; G4 huge3b)")
    ap.add_argument("--fsdp-val-windows", type=int, default=20,
                    help="val windows per FSDP eval (all ranks forward; keep small)")
    ap.add_argument("--no-save-opt", action="store_true",
                    help="checkpoint model weights only (smaller; default for --fsdp pilot)")
    ap.add_argument("--stateful-checkpoints", action="store_true",
                    help="save optimizer plus rank-local sampling/CPU/CUDA RNG state")
    ap.add_argument("--require-complete-state", action="store_true",
                    help="fail resume unless optimizer and all rank-local RNG state exist")
    ap.add_argument("--expected-resume-step", type=int, default=-1,
                    help="fail unless --resume starts at this exact global step")
    ap.add_argument("--legacy-sampling-rng-offset", type=int, default=0,
                    help="replay this many legacy per-step sampling draws when RNG state is absent")
    ap.add_argument("--restart-warmup-steps", type=int, default=0,
                    help="linearly ramp the scheduled LR after a model-only warm restart")
    ap.add_argument("--restart-warmup-initial-factor", type=float, default=1.0,
                    help="initial multiplier for --restart-warmup-steps")
    ap.add_argument("--stop-after-resume", action="store_true",
                    help="load and verify a checkpoint, then exit before sampling/training")
    ap.add_argument("--no-prune-checkpoints", action="store_true",
                    help="preserve every source/target checkpoint during recovery")
    args = ap.parse_args()
    if args.fsdp and not args.stateful_checkpoints and not args.no_save_opt:
        args.no_save_opt = True
    if args.stateful_checkpoints and args.no_save_opt:
        ap.error("--stateful-checkpoints is incompatible with --no-save-opt")
    if args.require_complete_state and not args.stateful_checkpoints:
        ap.error("--require-complete-state requires --stateful-checkpoints")
    if args.expected_resume_step < -1:
        ap.error("--expected-resume-step must be -1 or non-negative")
    if args.legacy_sampling_rng_offset < 0:
        ap.error("--legacy-sampling-rng-offset must be non-negative")
    if args.restart_warmup_steps < 0:
        ap.error("--restart-warmup-steps must be non-negative")
    if not 0.0 < args.restart_warmup_initial_factor <= 1.0:
        ap.error("--restart-warmup-initial-factor must be in (0, 1]")

    # DDP from SLURM env; single-process fallback for local smoke
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    distributed = world > 1
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)
        # Leonardo torch is 2.0.0a0: no device_id kwarg; set_device BEFORE init
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    train_buf = open_mmap(args.train_bin)
    valid_buf = open_mmap(args.val_bin)
    log(f"train tokens={train_buf.shape[0]:,} val tokens={valid_buf.shape[0]:,} "
        f"world={world} eff_batch={args.batch * world} "
        f"tok/step={args.batch * world * args.ctx:,}")

    cfg = lm_config(args.size)
    cfg["max_len"] = args.ctx
    cfg["memory_layer"] = args.memory_layer
    cfg["memory_size"] = args.memory_size
    use_fsdp = distributed and args.fsdp
    torch.manual_seed(7)  # same init on all ranks BEFORE DDP wrap
    model = HLM5LM(vocab=args.vocab, **cfg).to(device)
    if args.grad_checkpoint and not use_fsdp:
        model.grad_checkpointing = True
        log("grad_checkpointing=on (per-layer)")
    elif args.grad_checkpoint and use_fsdp:
        log("grad_checkpointing deferred to FSDP activation checkpointing")
    n_params = sum(p.numel() for p in model.parameters())
    log(f"HLM5LM {args.size}: {n_params / 1e6:.1f}M params (vocab {args.vocab}, "
        f"memory_layer={args.memory_layer})")

    # the editability-floor lesson: no WD on embeddings/norms
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "emb" in name else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    start_step, best = 0, float("inf")
    out_dir = Path(args.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    if use_fsdp:
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import torch.nn as nn

        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={nn.TransformerEncoderLayer},
        )
        model = FSDP(
            model,
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
        log("FSDP FULL_SHARD on (use_orig_params=True for tied embed/head)")
        if args.grad_checkpoint:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=partial(
                        checkpoint_wrapper,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    ),
                    check_fn=lambda module: isinstance(module, nn.TransformerEncoderLayer),
                )
                log("FSDP activation checkpointing: ON (non-reentrant)")
            except Exception as exc:
                log(f"FSDP activation checkpointing: FAILED ({exc})")
        if args.stateful_checkpoints:
            log("fsdp state-complete checkpoints: optimizer + rank-local RNG")
        elif args.no_save_opt:
            log("fsdp pilot: model-only checkpoints (no optimizer shard)")
    elif distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    raw = model if use_fsdp else (model.module if distributed else model)

    gen = torch.Generator().manual_seed(1234 + rank)  # distinct windows per rank
    optimizer_state_restored = False
    runtime_state_restored = False
    resume_found = False

    if args.resume == "auto":
        kind, ck_path, _ = find_latest_checkpoint(out_dir)
        if ck_path is not None:
            resume_found = True
            if use_fsdp and kind == "sharded":
                shard_path = ck_path / f"shard_rank{rank}.pt"
                if not shard_path.is_file():
                    raise RuntimeError(
                        f"FSDP resume missing shard {shard_path}; refusing to start fresh "
                        f"over an existing sharded checkpoint")
                else:
                    ck = torch.load(shard_path, map_location="cpu", weights_only=False)
                    ck_world = int(ck.get("world_size", world))
                    if ck_world != world:
                        raise RuntimeError(
                            f"FSDP resume world_size mismatch: ckpt={ck_world} run={world}")
                    has_optimizer = "opt" in ck and not args.no_save_opt
                    has_runtime = "runtime_state" in ck
                    if args.require_complete_state and not (has_optimizer and has_runtime):
                        raise RuntimeError(
                            "state-complete resume required but optimizer or runtime state is missing")
                    with fsdp_local_state_dict_context(
                            model, save_opt=has_optimizer):
                        _, normalized_keys = load_model_state_dict_with_fallback(
                            model, ck["model"], log)
                        if has_optimizer:
                            opt.load_state_dict(ck["opt"])
                            optimizer_state_restored = True
                    if has_runtime:
                        restore_runtime_state(ck["runtime_state"], gen, device)
                        runtime_state_restored = True
                    start_step = ck["step"] + 1
                    best = ck.get("best", best)
                    if normalized_keys:
                        log("normalized legacy FSDP local-state wrapper keys")
                    log(f"resumed sharded {ck_path} rank={rank} at step {start_step}")
            else:
                ck = torch.load(ck_path, map_location="cpu", weights_only=False)
                if use_fsdp:
                    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                    load_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_cfg):
                        model.load_state_dict(ck["model"])
                else:
                    raw.load_state_dict(ck["model"])
                has_optimizer = "opt" in ck and not args.no_save_opt
                has_runtime = "runtime_state" in ck
                if args.require_complete_state and not (has_optimizer and has_runtime):
                    raise RuntimeError(
                        "state-complete resume required but optimizer or runtime state is missing")
                if has_optimizer:
                    opt.load_state_dict(ck["opt"])
                    optimizer_state_restored = True
                if has_runtime:
                    restore_runtime_state(ck["runtime_state"], gen, device)
                    runtime_state_restored = True
                start_step = ck["step"] + 1
                best = ck.get("best", best)
                log(f"resumed {ck_path} at step {start_step}")
            if distributed:
                dist.barrier()

    if args.expected_resume_step >= 0:
        if not resume_found or start_step != args.expected_resume_step:
            raise RuntimeError(
                f"expected resume step {args.expected_resume_step}, got {start_step}")
    if args.legacy_sampling_rng_offset:
        if runtime_state_restored:
            raise RuntimeError("sampling RNG offset cannot be combined with restored runtime state")
        if not resume_found:
            raise RuntimeError("sampling RNG offset requires a resumed checkpoint")
        advance_sampling_generator(
            gen,
            args.legacy_sampling_rng_offset,
            args.batch,
            train_buf.shape[0] - args.ctx - 1,
        )
        log(f"advanced sampling RNG by {args.legacy_sampling_rng_offset} legacy draws")
    if args.require_complete_state and not (optimizer_state_restored and runtime_state_restored):
        raise RuntimeError("state-complete resume did not restore optimizer and runtime state")
    if args.stateful_checkpoints and resume_found and not optimizer_state_restored:
        log("warm restart: source checkpoint has no optimizer state")
    if args.stateful_checkpoints and resume_found and not runtime_state_restored:
        log("warm restart: source checkpoint has no captured CPU/CUDA runtime RNG state")
    if args.restart_warmup_steps and not resume_found:
        raise RuntimeError("restart LR warmup requires a resumed checkpoint")

    resume_start_step = start_step

    def lr_at(step):
        if step < args.warmup:
            scheduled_lr = args.lr * (step + 1) / args.warmup
        else:
            p = (step - args.warmup) / max(1, args.steps - args.warmup)
            scheduled_lr = args.lr * 0.5 * (1 + math.cos(math.pi * p))
        restart_offset = step - resume_start_step
        scheduled_lr *= restart_lr_multiplier(
            restart_offset,
            args.restart_warmup_steps,
            args.restart_warmup_initial_factor,
        )
        return scheduled_lr

    if args.restart_warmup_steps:
        log(
            f"restart LR bridge: steps={args.restart_warmup_steps} "
            f"initial_factor={args.restart_warmup_initial_factor:.6f} "
            f"resume_step={resume_start_step}")

    if args.stop_after_resume:
        if not resume_found:
            raise RuntimeError("--stop-after-resume requires a resumed checkpoint")
        if distributed:
            dist.barrier()
        log(
            f"STATEFUL_RESUME_VALIDATED step={start_step} "
            f"optimizer={optimizer_state_restored} runtime={runtime_state_restored}")
        if distributed:
            dist.destroy_process_group()
        return

    model.train()
    t0, t0_step = time.time(), start_step
    last_step = start_step - 1
    nonfinite_seen = 0

    def all_ranks_finite(value):
        if isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value.detach()).all().item())
        else:
            finite = math.isfinite(float(value))
        if distributed:
            ok_t = torch.tensor([1 if finite else 0], device=device, dtype=torch.int32)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            finite = bool(ok_t.item())
        return finite

    def handle_nonfinite(kind, step, value):
        nonlocal nonfinite_seen
        nonfinite_seen += 1
        try:
            value_s = f"{float(value):.6g}"
        except Exception:
            value_s = str(value)
        log(f"NON-FINITE {kind} at step {step} value={value_s} "
            f"count={nonfinite_seen} policy={args.nonfinite_policy}")
        opt.zero_grad(set_to_none=True)
        if args.nonfinite_policy == "skip" and nonfinite_seen <= args.max_nonfinite:
            return True
        if is_main:
            failure = {
                "reason": f"nonfinite_{kind.lower()}",
                "step": step,
                "value": value_s,
                "last_completed_step": last_step,
                "target_steps": args.steps,
                "best_val_ppl": best,
                "params": n_params,
                "eff_batch": args.batch * world,
                "ctx": args.ctx,
                "fsdp_sharded": bool(use_fsdp),
                "world_size": world,
            }
            (out_dir / "train_failure.json").write_text(json.dumps(failure, indent=2))
        raise RuntimeError(f"non-finite {kind.lower()} at step {step}")

    for step in range(start_step, args.steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        x, y = sample_batch(train_buf, args.batch, args.ctx, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not all_ranks_finite(loss):
            if handle_nonfinite("LOSS", step, loss):
                continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if use_fsdp:
            grad_norm = model.clip_grad_norm_(args.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not all_ranks_finite(grad_norm):
            if handle_nonfinite("GRAD", step, grad_norm):
                continue
        opt.step()
        last_step = step

        if step % args.log_every == 0 or step == args.steps - 1:
            tok_s = (step - t0_step + 1) * args.batch * world * args.ctx / (time.time() - t0)
            log(f"step {step:6d} loss {float(loss):.3f} lr {lr_at(step):.2e} "
                f"{tok_s / 1e3:.0f}K tok/s")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            if distributed:
                dist.barrier()
            if is_main:
                log(f"  eval at step {step}...")
            eval_model = model if use_fsdp else raw
            max_win = args.fsdp_val_windows if use_fsdp else 100
            ppl = val_ppl(eval_model, valid_buf, args.ctx, device,
                          max_windows=max_win, rank=rank, world=world)
            if is_main:
                log(f"  val PPL {ppl:.2f}")
                if ppl < best:
                    best = ppl
                best_t = torch.tensor([best], device=device, dtype=torch.float64)
            elif distributed:
                best_t = torch.tensor([0.0], device=device, dtype=torch.float64)
            if distributed:
                dist.broadcast(best_t, src=0)
                best = float(best_t.item())
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            if distributed:
                dist.barrier()
            if is_main:
                log(f"  checkpoint at step {step}...")
            runtime_state = (
                capture_runtime_state(gen, device) if args.stateful_checkpoints else None)
            meta = {"size": args.size, "cfg": cfg, "vocab": args.vocab,
                    "val_ppl": best, "step": step, "best": best,
                    "checkpoint_schema": (STATE_COMPLETE_SCHEMA
                                          if args.stateful_checkpoints else "model-only-v1"),
                    "optimizer_state_saved": bool(args.stateful_checkpoints),
                    "runtime_state_saved": bool(args.stateful_checkpoints)}
            if use_fsdp:
                saved_ckpt = save_fsdp_sharded_checkpoint(
                    model, opt, out_dir, step, meta, rank, world,
                    save_opt=not args.no_save_opt, runtime_state=runtime_state,
                    log=log if is_main else lambda _m: None)
                if not saved_ckpt:
                    raise RuntimeError(f"failed to save sharded ckpt_step{step + 1}")
                if is_main:
                    if args.no_prune_checkpoints:
                        log("checkpoint pruning disabled")
                    else:
                        prune_sharded_checkpoints(out_dir, keep=2, log=log)
            else:
                saved_ckpt = True
                if is_main:
                    ckpt_payload = {"model": raw.state_dict(), **meta}
                    if not args.no_save_opt:
                        opt_state = opt.state_dict()
                        if args.stateful_checkpoints and not opt_state.get("state"):
                            raise RuntimeError("refusing to write an empty optimizer state")
                        ckpt_payload["opt"] = opt_state
                    if args.stateful_checkpoints:
                        ckpt_payload["runtime_state"] = runtime_state
                    saved_ckpt = safe_torch_save(
                        ckpt_payload,
                        out_dir / f"ckpt_step{step + 1}.pt",
                        log,
                        f"ckpt_step{step + 1}.pt")
                if distributed:
                    ok_t = torch.tensor([1 if saved_ckpt else 0], device=device, dtype=torch.int32)
                    dist.broadcast(ok_t, src=0)
                    saved_ckpt = bool(ok_t.item())
                if not saved_ckpt:
                    raise RuntimeError(f"failed to save ckpt_step{step + 1}.pt")
                if is_main:
                    log(f"  saved ckpt_step{step + 1}.pt")
                    if args.no_prune_checkpoints:
                        log("checkpoint pruning disabled")
                    else:
                        prune_step_checkpoints(out_dir, keep=2, log=log)
            if distributed:
                dist.barrier()

    if is_main:
        summary = {"best_val_ppl": best, "params": n_params, "steps": args.steps,
                   "last_step": last_step, "completed": last_step >= args.steps - 1,
                   "eff_batch": args.batch * world, "ctx": args.ctx}
        if use_fsdp:
            summary["fsdp_sharded"] = True
            summary["world_size"] = world
        summary["stateful_checkpoints"] = bool(args.stateful_checkpoints)
        summary["optimizer_state_restored"] = bool(optimizer_state_restored)
        summary["runtime_state_restored"] = bool(runtime_state_restored)
        if last_step >= args.steps - 1:
            (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
            log(f"done. best val PPL {best:.2f} -> {out_dir}")
        else:
            (out_dir / "train_incomplete.json").write_text(json.dumps(summary, indent=2))
            log(f"incomplete at step {last_step}; no train_summary.json written -> {out_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
