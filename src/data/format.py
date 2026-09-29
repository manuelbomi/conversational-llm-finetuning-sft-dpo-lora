"""Turns a raw `{utterance, label, chosen, rejected}` example into the two
shapes training actually needs: a single instruction/response string for
SFT, and a prompt/chosen/rejected triple for DPO. Both training scripts and
the eval script import from here so the prompt format used at train time
and at eval/serve time can never silently drift apart.
"""

from __future__ import annotations

from src.taxonomy import INSTRUCTION_TEMPLATE, SIGNAL_LABELS


def build_prompt(utterance: str) -> str:
    return INSTRUCTION_TEMPLATE.format(labels=", ".join(SIGNAL_LABELS), utterance=utterance)


def build_sft_target(label: str, response: str) -> str:
    """The response format the model is trained to produce: a labeled line
    followed by the suggested response, both machine-parseable (see
    `src/eval/evaluate.py::parse_completion`) and human-readable.
    """
    return f"Label: {label}\nResponse: {response}"


def format_sft_example(example: dict) -> dict:
    """Shape trl's `SFTTrainer` expects: a single `text` field with the full
    prompt+response, since we're doing plain causal-LM SFT rather than a
    chat-template-based format (kept simple on purpose for a tutorial --
    see README for the chat-template variant note).
    """
    prompt = build_prompt(example["utterance"])
    target = build_sft_target(example["label"], example["chosen"])
    return {"text": f"{prompt}\n\n{target}"}


def format_dpo_example(example: dict) -> dict:
    """Shape trl's `DPOTrainer` expects: `prompt`, `chosen`, `rejected`."""
    prompt = build_prompt(example["utterance"])
    return {
        "prompt": prompt,
        "chosen": build_sft_target(example["label"], example["chosen"]),
        "rejected": build_sft_target(example["label"], example["rejected"]),
    }
