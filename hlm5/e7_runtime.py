"""Runtime helpers for the pinned frozen-3B E7 experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .e7_contract import (
    MODEL_FORWARD_DTYPE,
    MODEL_HIDDEN_SIZE,
    MODEL_LAYERS,
    MODEL_PARAMETER_COUNT,
    MODEL_VOCAB_SIZE,
)


def load_pinned_tokenizer(snapshot: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("pinned tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pinned_model(snapshot: Path, device: torch.device) -> Any:
    from transformers import AutoModelForCausalLM

    if device.type != "cuda":
        raise RuntimeError("E7 model forward is registered only on CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def assert_model_invariants(model: Any) -> dict[str, Any]:
    head = model.get_output_embeddings()
    embedding = model.get_input_embeddings()
    config = model.config
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    floating_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    checks = {
        "class_name": type(model).__name__,
        "parameter_count": parameter_count,
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "layers": int(config.num_hidden_layers),
        "tied_embeddings": bool(
            head.weight.data_ptr() == embedding.weight.data_ptr()
        ),
        "head_bias_is_none": getattr(head, "bias", None) is None,
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "floating_parameter_dtypes": floating_dtypes,
    }
    expected = {
        "class_name": "SmolLM3ForCausalLM",
        "parameter_count": MODEL_PARAMETER_COUNT,
        "hidden_size": MODEL_HIDDEN_SIZE,
        "vocab_size": MODEL_VOCAB_SIZE,
        "layers": MODEL_LAYERS,
        "tied_embeddings": True,
        "head_bias_is_none": True,
        "all_parameters_frozen": True,
        "floating_parameter_dtypes": [MODEL_FORWARD_DTYPE],
    }
    if checks != expected:
        raise RuntimeError(f"pinned E7 model invariant mismatch: {checks} != {expected}")
    return checks


def encode_batch(
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] == 0:
        raise ValueError("E7 prompt encoded to an empty sequence")
    return {name: tensor.to(device) for name, tensor in encoded.items()}


@torch.inference_mode()
def final_hidden_batch(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, MODEL_HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("registered SmolLM3 CausalLM has no .model backbone")
    rows: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = encode_batch(tokenizer, texts[start : start + batch_size], device)
        outputs = backbone(**batch, use_cache=False, return_dict=True)
        last_indices = batch["attention_mask"].sum(dim=1) - 1
        hidden_batch_size = batch["input_ids"].shape[0]
        row_indices = torch.arange(hidden_batch_size, device=device)
        hidden = outputs.last_hidden_state[row_indices, last_indices]
        if hidden.shape[0] != hidden_batch_size:
            raise RuntimeError("failed to select one final hidden state per prompt")
        if hidden.dtype != torch.bfloat16:
            raise RuntimeError(f"unexpected native hidden dtype: {hidden.dtype}")
        rows.append(hidden)
    return torch.cat(rows, dim=0)


@torch.inference_mode()
def native_head_logits(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dtype != torch.bfloat16:
        raise RuntimeError(f"ordinary E7 head requires bfloat16 hidden, got {hidden.dtype}")
    logits = model.get_output_embeddings()(hidden)
    if logits.dtype != torch.bfloat16:
        raise RuntimeError(f"ordinary E7 head returned {logits.dtype}, expected bfloat16")
    return logits
