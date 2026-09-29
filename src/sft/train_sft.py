#!/usr/bin/env python
"""Stage 1: supervised fine-tuning (SFT) with LoRA.

Teaches the base model the *format* this tutorial wants (predict a signal
label, then a suggested response) via next-token prediction on
prompt+target pairs. This is the stage that gets the model to reliably
produce parseable output at all -- DPO (stage 2) then nudges it to prefer
the *compliant* response style over the over-promising one, starting from
this SFT checkpoint rather than the raw base model.

Runs on CPU with a tiny model by default (`--model-name sshleifer/tiny-gpt2`,
a real but tiny randomly-initialized GPT-2 -- useless for output quality,
useful for proving the training loop is correct and fast to iterate on). For
a real model, see the `--model-name` recommendation in README.md and the
QLoRA note below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from src.data.format import format_sft_example  # noqa: E402


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # "all-linear" auto-targets every linear projection layer peft can
        # find on the loaded model, so this config works unmodified across
        # architectures (GPT-2, Llama, Qwen, ...) instead of hardcoding
        # per-model module names like ["q_proj", "v_proj"].
        target_modules="all-linear",
    )


def load_model_and_tokenizer(model_name: str, use_qlora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_qlora:
        try:
            import torch
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            print(
                "WARNING: --use-qlora was set but bitsandbytes is not usable in this "
                "environment (no CUDA / not installed). Falling back to plain LoRA. "
                "QLoRA needs a real GPU -- see docs/PRODUCTION.md.",
                file=sys.stderr,
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "synthetic")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "sft-adapter")
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 = use --epochs instead")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.data_dir / "train.jsonl"),
            "validation": str(args.data_dir / "val.jsonl"),
        },
    )
    dataset = dataset.map(format_sft_example, remove_columns=dataset["train"].column_names)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_qlora)
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=False,  # tiny model/tiny data -- not needed, and it slows CPU runs down
        logging_steps=1,
        report_to="none",
        dataset_text_field="text",
        max_length=256,
        save_strategy="no",  # we save the adapter explicitly below
        use_cpu=not _cuda_available(),
        bf16=False,
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"SFT training complete. final loss: {train_result.training_loss:.4f}")
    print(f"adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
