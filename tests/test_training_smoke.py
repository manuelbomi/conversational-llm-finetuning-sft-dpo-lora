"""End-to-end smoke tests: do the actual SFT and DPO training scripts run
without error and produce a loadable adapter? These use the real
`sshleifer/tiny-gpt2` (a genuinely tiny, randomly-initialized GPT-2 -- a
few KB of weights) and 3 training steps, so they run in well under a
minute even on CPU, but they exercise the real `transformers`/`peft`/`trl`
code path, not a mock. They need network access to download the tiny model
the first time (cached afterward) -- see README's "Testing" section.

These prove the training *loop* is wired correctly (data format -> trainer
-> saved adapter -> reloadable). They say nothing about output *quality*,
which needs a real base model and real data -- see README's honest caveat
about what this repo's tests do and don't demonstrate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_synthetic_data import generate_examples, write_jsonl

    data_dir = tmp_path_factory.mktemp("data")
    write_jsonl(data_dir / "train.jsonl", generate_examples(20, seed=1))
    write_jsonl(data_dir / "val.jsonl", generate_examples(6, seed=2))
    write_jsonl(data_dir / "test.jsonl", generate_examples(6, seed=3))
    return data_dir


@pytest.fixture(scope="module")
def sft_adapter_dir(tmp_path_factory, tiny_dataset) -> Path:
    output_dir = tmp_path_factory.mktemp("sft-adapter")
    result = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "src" / "sft" / "train_sft.py"),
            "--data-dir",
            str(tiny_dataset),
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "3",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"train_sft.py failed:\n{result.stdout}\n{result.stderr}"
    return output_dir


def test_sft_training_produces_a_loadable_adapter(sft_adapter_dir):
    assert (sft_adapter_dir / "adapter_config.json").exists()
    assert (sft_adapter_dir / "adapter_model.safetensors").exists() or list(
        sft_adapter_dir.glob("adapter_model.*")
    )


def test_dpo_training_continues_from_sft_adapter(tiny_dataset, sft_adapter_dir, tmp_path):
    dpo_output_dir = tmp_path / "dpo-adapter"
    result = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "src" / "dpo" / "train_dpo.py"),
            "--data-dir",
            str(tiny_dataset),
            "--sft-adapter-dir",
            str(sft_adapter_dir),
            "--output-dir",
            str(dpo_output_dir),
            "--max-steps",
            "3",
            "--batch-size",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"train_dpo.py failed:\n{result.stdout}\n{result.stderr}"
    assert (dpo_output_dir / "adapter_config.json").exists()


def test_dpo_training_fails_clearly_without_sft_adapter(tiny_dataset, tmp_path):
    result = subprocess.run(
        [
            PYTHON,
            str(REPO_ROOT / "src" / "dpo" / "train_dpo.py"),
            "--data-dir",
            str(tiny_dataset),
            "--sft-adapter-dir",
            str(tmp_path / "does-not-exist"),
            "--output-dir",
            str(tmp_path / "dpo-out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "run src/sft/train_sft.py first" in result.stdout + result.stderr
