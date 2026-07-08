"""Checkpoint/tokenizer loading for the HLM5 release bundle."""
from __future__ import annotations

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"

_CKPT = {
    "baseline": "hlm5_lm_baseline_fineweb_g3_final.pt",
    "hybrid": "hlm5_lm_hybrid_fineweb_g3_final.pt",
}


def checkpoint_path(which: str = "baseline") -> Path:
    p = MODELS_DIR / _CKPT[which]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download the checkpoint from "
            "https://huggingface.co/qriton/hlm5-1b-trunk into models/ "
            "(see models/README.md)."
        )
    return p


def load_tokenizer(path: Path | None = None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path or MODELS_DIR / "tokenizers" / "fineweb-65536-compat.json"))


def load_trunk(which: str = "baseline", device: str | None = None, path: Path | None = None):
    from .model import HLM5LM
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path or checkpoint_path(which), map_location=dev, weights_only=True)
    model = HLM5LM(vocab=ck["vocab"], **ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck
