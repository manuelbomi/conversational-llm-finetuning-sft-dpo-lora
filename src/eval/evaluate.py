#!/usr/bin/env python
"""Evaluate a trained adapter against the held-out test split.

Reports two independent things (see `src/eval/metrics.py` docstrings for
why they're kept separate): (1) label classification quality (precision/
recall/F1 per signal label, parsed from generated completions), and (2)
DPO preference win rate (does the model score the compliant response
higher than the over-promising one, by log-probability).

Usage:
    python src/eval/evaluate.py --adapter-dir artifacts/dpo-adapter
    python src/eval/evaluate.py --adapter-dir artifacts/sft-adapter  # compare stages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.data.format import build_prompt, build_sft_target  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    compute_label_metrics,
    compute_preference_win_rate,
    parse_completion,
)
from src.taxonomy import SIGNAL_LABELS  # noqa: E402


def load_examples(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@torch.no_grad()
def generate_label_predictions(
    model, tokenizer, examples: list[dict], max_new_tokens: int = 24
) -> list[str | None]:
    predictions = []
    for ex in examples:
        prompt = build_prompt(ex["utterance"])
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append(parse_completion(completion))
    return predictions


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, target: str) -> float:
    """Sum of log P(target_token | prompt + prior target tokens) -- the
    quantity DPO's loss is actually built from, computed here directly
    (not via DPOTrainer) so this eval script has no training-loop
    dependency and can score any adapter checkpoint standalone.
    """
    full_text = prompt + "\n\n" + target
    prompt_ids = tokenizer(prompt + "\n\n", return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    logits = model(full_ids).logits  # (1, seq_len, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)

    target_start = prompt_ids.shape[1]
    total_logprob = 0.0
    for i in range(target_start, full_ids.shape[1]):
        token_id = full_ids[0, i]
        total_logprob += log_probs[0, i - 1, token_id].item()
    return total_logprob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "synthetic")
    parser.add_argument("--max-examples", type=int, default=None, help="cap for a fast smoke run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    examples = load_examples(args.data_dir / "test.jsonl")
    if args.max_examples:
        examples = examples[: args.max_examples]

    predictions = generate_label_predictions(model, tokenizer, examples)
    golds = [ex["label"] for ex in examples]
    label_metrics = compute_label_metrics(predictions, golds, SIGNAL_LABELS)

    chosen_logps, rejected_logps = [], []
    for ex in examples:
        prompt = build_prompt(ex["utterance"])
        chosen_logps.append(
            sequence_logprob(model, tokenizer, prompt, build_sft_target(ex["label"], ex["chosen"]))
        )
        rejected_logps.append(
            sequence_logprob(
                model, tokenizer, prompt, build_sft_target(ex["label"], ex["rejected"])
            )
        )
    win_rate = compute_preference_win_rate(chosen_logps, rejected_logps)

    report = {
        "adapter_dir": str(args.adapter_dir),
        "n_examples": len(examples),
        "label_metrics": label_metrics,
        "preference_win_rate": win_rate,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
