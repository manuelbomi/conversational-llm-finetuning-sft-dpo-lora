#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base model's weights and export a
plain (non-PEFT) model directory.

Why this exists alongside `AdapterRegistry` (src/serve/model_registry.py),
which serves adapters *unmerged*: these are two different production
tradeoffs, not competing implementations of the same thing.

- Unmerged (AdapterRegistry): many adapters can share one base model in
  memory, hot-swap is instant (just changes which small adapter matrix is
  active), and rollback is free. This is what you want while you're still
  iterating on adapters, or running several task-specific adapters side by
  side. See docs/adr/0001-lora-vs-full-finetune.md.
- Merged (this script): the LoRA delta is folded into the base weights, so
  the result loads and runs with plain `AutoModelForCausalLM.from_pretrained`
  -- no `peft` dependency at inference time, one fewer moving part, and it's
  the shape most high-throughput serving stacks (vLLM, TGI, TensorRT-LLM)
  expect. This is what you want once an adapter is *finalized* and about to
  go into a low-latency serving path (see the sibling
  `low-latency-llm-serving-benchmark` repo).

Usage:
    python src/merge_and_export.py --adapter-dir artifacts/dpo-adapter --out-dir artifacts/merged-model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def merge_and_export(model_name: str, adapter_dir: Path, out_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    # `merge_and_unload()` folds each LoRA delta (B @ A * scaling) into its
    # target linear layer's weight matrix in place, then strips the peft
    # wrapper -- what comes back is a plain `AutoModelForCausalLM`-compatible
    # module, not a `PeftModel`.
    t0 = time.perf_counter()
    merged_model = peft_model.merge_and_unload()
    merge_seconds = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    adapter_size = sum(f.stat().st_size for f in adapter_dir.glob("*.safetensors"))
    merged_size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))

    return {
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(out_dir),
        "merge_seconds": round(merge_seconds, 4),
        "adapter_weights_bytes": adapter_size,
        "merged_weights_bytes": merged_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "merged-model")
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"{args.adapter_dir} not found -- train and save an adapter first.")

    report = merge_and_export(args.model_name, args.adapter_dir, args.out_dir)
    print(json.dumps(report, indent=2))
    print(
        f"\nMerged model exported to {args.out_dir} -- loadable with plain "
        "AutoModelForCausalLM.from_pretrained(...), no peft import required."
    )


if __name__ == "__main__":
    main()
