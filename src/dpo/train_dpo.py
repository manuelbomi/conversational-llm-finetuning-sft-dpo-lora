#!/usr/bin/env python
"""Stage 2: preference alignment with DPO, starting from the SFT adapter.

SFT (stage 1) taught the model the output *format*. DPO teaches it a
*preference*: given the same prompt, prefer the compliant response
(`chosen`) over the over-promising one (`rejected`) it could otherwise
produce with similar likelihood. This is what closes the gap between "the
model can produce a well-formatted answer" and "the model reliably produces
the answer we actually want."

Continues training the SAME LoRA adapter SFT produced (`--sft-adapter-dir`)
rather than starting a fresh one -- DPO is a refinement of the SFT policy,
not a separate model. `DPOTrainer` gets the reference-model log-probs for
free by temporarily disabling the adapter on this same model (standard
practice with peft + DPO), so no second copy of the base model needs to be
loaded.

See docs/adr/0001-lora-vs-full-finetune.md and
docs/adr/0002-sft-then-dpo-vs-single-stage.md for why this two-stage,
adapter-based approach was chosen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import DPOConfig, DPOTrainer  # noqa: E402

from src.data.format import format_dpo_example  # noqa: E402


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument(
        "--sft-adapter-dir", type=Path, default=REPO_ROOT / "artifacts" / "sft-adapter"
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "synthetic")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "dpo-adapter")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature: how sharply to prefer chosen over rejected",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if not args.sft_adapter_dir.exists():
        raise SystemExit(
            f"{args.sft_adapter_dir} not found -- run src/sft/train_sft.py first, DPO continues from the SFT adapter."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # `is_trainable=True` is what lets DPOTrainer continue updating this
    # adapter's weights, rather than treating it as a frozen starting point.
    model = PeftModel.from_pretrained(base_model, args.sft_adapter_dir, is_trainable=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "val.jsonl"),
        },
    )
    dataset = dataset.map(format_dpo_example, remove_columns=dataset["train"].column_names)

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        max_length=256,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        use_cpu=not _cuda_available(),
        bf16=False,
        fp16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None + a PeftModel => trl disables the adapter to get reference log-probs, see module docstring
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"DPO training complete. final loss: {train_result.training_loss:.4f}")
    print(f"adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
