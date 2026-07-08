import pytest
import torch

from hlm5 import io
from hlm5.model import HLM5LM

requires_weights = pytest.mark.skipif(
    not (io.MODELS_DIR / "hlm5_lm_baseline_fineweb_g3_final.pt").exists(),
    reason="checkpoint not downloaded")


def test_tokenizer_loads():
    tok = io.load_tokenizer()
    assert tok.get_vocab_size() == 65536


@requires_weights
def test_checkpoint_meta_weights_only():
    ck = torch.load(io.checkpoint_path("baseline"), weights_only=True, map_location="cpu")
    assert ck["vocab"] == 65536
    assert "model" in ck and "cfg" in ck and "val_ppl" in ck


def test_tiny_forward_random():
    m = HLM5LM(vocab=128, dim=32, n_layers=1, n_heads=2, max_len=16)
    ids = torch.randint(0, 128, (1, 8))
    logits, audit = m(ids)          # forward returns (logits, audit)
    assert logits.shape == (1, 8, 128)
    assert audit is None
