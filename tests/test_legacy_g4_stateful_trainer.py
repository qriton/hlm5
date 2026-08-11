"""CPU and static contracts for the legacy G4 state-complete trainer."""

from __future__ import annotations

import importlib.util
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "research"
    / "legacy_g4_recovery"
    / "train_hlm5_lm_fineweb_stateful.py"
)


def load_trainer_module():
    stub = types.ModuleType("hlm5_lm")
    stub.HLM5LM = object
    stub.lm_config = lambda _size: {}
    previous = sys.modules.get("hlm5_lm")
    sys.modules["hlm5_lm"] = stub
    try:
        spec = importlib.util.spec_from_file_location("legacy_g4_stateful_test", TRAINER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("hlm5_lm", None)
        else:
            sys.modules["hlm5_lm"] = previous


def test_sampling_rng_replay_matches_repeated_legacy_draws() -> None:
    trainer = load_trainer_module()
    actual = torch.Generator().manual_seed(1234)
    expected = torch.Generator().manual_seed(1234)

    trainer.advance_sampling_generator(actual, draws=11, batch=3, upper_bound=997)
    for _ in range(11):
        torch.randint(0, 997, (3,), generator=expected)

    assert torch.equal(actual.get_state(), expected.get_state())
    assert torch.equal(
        torch.randint(0, 997, (3,), generator=actual),
        torch.randint(0, 997, (3,), generator=expected),
    )


def test_cpu_runtime_state_round_trip_is_exact() -> None:
    trainer = load_trainer_module()
    generator = torch.Generator().manual_seed(4567)
    torch.manual_seed(7654)
    state = trainer.capture_runtime_state(generator, "cpu")

    expected_sample = torch.randint(0, 1000, (8,), generator=generator)
    expected_global = torch.randint(0, 1000, (8,))
    torch.manual_seed(1)
    generator.manual_seed(2)

    trainer.restore_runtime_state(state, generator, "cpu")
    assert torch.equal(torch.randint(0, 1000, (8,), generator=generator), expected_sample)
    assert torch.equal(torch.randint(0, 1000, (8,)), expected_global)


def test_restart_lr_multiplier_has_locked_endpoints() -> None:
    trainer = load_trainer_module()
    assert trainer.restart_lr_multiplier(0, 2000, 0.1) == 0.1
    assert trainer.restart_lr_multiplier(1999, 2000, 0.1) == 1.0
    assert trainer.restart_lr_multiplier(2000, 2000, 0.1) == 1.0
    assert trainer.restart_lr_multiplier(-1, 2000, 0.1) == 1.0
    assert trainer.restart_lr_multiplier(0, 0, 0.1) == 1.0


def test_state_complete_contract_is_fail_closed() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "hlm5-g4-state-complete-v1" in text
    assert "--stateful-checkpoints" in text
    assert "--require-complete-state" in text
    assert "--expected-resume-step" in text
    assert "--legacy-sampling-rng-offset" in text
    assert "--restart-warmup-steps" in text
    assert "--stop-after-resume" in text
    assert "--no-prune-checkpoints" in text
    assert 'payload["opt"] = opt_state' in text
    assert 'payload["runtime_state"] = runtime_state' in text
    assert "refusing to write an empty optimizer state" in text
    assert "state-complete resume required" in text
    assert "STATEFUL_RESUME_VALIDATED" in text
    assert text.count("checkpoint pruning disabled") == 2


def test_legacy_default_remains_model_only_without_explicit_stateful_flag() -> None:
    text = TRAINER.read_text(encoding="utf-8")
    assert "if args.fsdp and not args.stateful_checkpoints and not args.no_save_opt:" in text
    assert "args.no_save_opt = True" in text
    assert "--stateful-checkpoints is incompatible with --no-save-opt" in text
    assert (
        "f74bf513719f8c3a110cbf45296889ad5914b3ca1cf7b5ba8538195d123d42f9"
        in text
    )


def test_tiny_state_complete_save_and_resume_round_trip(tmp_path: Path) -> None:
    fake_module = tmp_path / "hlm5_lm.py"
    fake_module.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class HLM5LM(nn.Module):\n"
        "    def __init__(self, vocab, **kwargs):\n"
        "        super().__init__()\n"
        "        self.emb = nn.Embedding(vocab, 8)\n"
        "        self.head = nn.Linear(8, vocab)\n"
        "    def forward(self, ids):\n"
        "        return self.head(self.emb(ids)), None\n"
        "def lm_config(name):\n"
        "    return {}\n",
        encoding="utf-8",
    )

    tokens = (np.arange(256, dtype=np.uint16) % 32).tobytes()
    for name in ("train.bin", "val.bin"):
        path = tmp_path / name
        path.write_bytes(b"HLM3" + struct.pack("<IQ", 1, 256) + tokens)

    out_dir = tmp_path / "run"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    common = [
        sys.executable,
        str(TRAINER),
        "--train-bin",
        str(tmp_path / "train.bin"),
        "--val-bin",
        str(tmp_path / "val.bin"),
        "--size",
        "tiny",
        "--vocab",
        "32",
        "--ctx",
        "8",
        "--batch",
        "2",
        "--steps",
        "2",
        "--warmup",
        "1",
        "--eval-every",
        "2",
        "--save-every",
        "2",
        "--out-dir",
        str(out_dir),
        "--stateful-checkpoints",
        "--no-prune-checkpoints",
    ]
    subprocess.run(common + ["--resume", "none"], check=True, env=environment)

    checkpoint = torch.load(out_dir / "ckpt_step2.pt", map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema"] == "hlm5-g4-state-complete-v1"
    assert checkpoint["optimizer_state_saved"] is True
    assert checkpoint["runtime_state_saved"] is True
    assert checkpoint["opt"]["state"]
    assert checkpoint["runtime_state"]["schema"] == "hlm5-g4-state-complete-v1"

    completed = subprocess.run(
        common
        + [
            "--resume",
            "auto",
            "--require-complete-state",
            "--expected-resume-step",
            "2",
            "--stop-after-resume",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "STATEFUL_RESUME_VALIDATED step=2 optimizer=True runtime=True" in completed.stdout
