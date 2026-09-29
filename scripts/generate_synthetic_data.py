#!/usr/bin/env python
"""Generate a synthetic labeled dataset for the SFT + DPO pipeline.

Fully offline, template-based, no external API -- deliberately so, because a
tutorial repo people clone and run immediately shouldn't need an LLM API key
just to produce its own training data. See docs/PRODUCTION.md for why real
projects should NOT stop at synthetic data: this generator's templates are
narrow and repetitive by construction, which is exactly why a model trained
only on this data won't generalize well -- the companion
`conversational-ai-eval-golden-dataset-framework` repo is where real,
SME-annotated data would come from in production.

Produces three JSONL files under `data/synthetic/`:
  - train.jsonl / val.jsonl / test.jsonl: each line is
    {"utterance": str, "label": str, "chosen": str, "rejected": str}
    -- `chosen`/`rejected` are a compliant vs. over-promising response,
    used directly as DPO preference pairs and as the SFT target (`chosen`).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

from src.taxonomy import SIGNAL_LABELS  # noqa: E402

# Each entry: (utterance_template, label, compliant_response_template,
# overpromising_response_template). `{product}` is filled from PRODUCTS.
_TEMPLATES: list[tuple[str, str, str, str]] = [
    (
        "Honestly the {product} plan feels too expensive for what I get.",
        "OBJECTION",
        "I understand the cost concern -- let's look at exactly what's included at this tier before deciding.",
        "Don't worry, I'll guarantee you a discount no matter what.",
    ),
    (
        "I'm hesitant about switching to {product} right now.",
        "OBJECTION",
        "That's a fair hesitation -- what specifically is giving you pause?",
        "You'd be crazy not to switch, everyone regrets waiting.",
    ),
    (
        "Okay, I'm ready to sign up for {product}, how do I start?",
        "BUYING_SIGNAL",
        "Great -- I'll confirm a couple of eligibility details, then walk you through account opening.",
        "Perfect, I'll guarantee this is the best decision you'll ever make.",
    ),
    (
        "Let's do it, sign me up for {product} today.",
        "BUYING_SIGNAL",
        "Happy to help -- first let's confirm the terms so you know exactly what you're agreeing to.",
        "This will definitely make you rich, let's go.",
    ),
    (
        "If I don't fix this {product} issue today, could I lose my account?",
        "ESCALATION_RISK",
        "I want to get this right for you -- let me connect you with a specialist who can review your account directly.",
        "Don't worry about it, nothing bad will ever happen to your account.",
    ),
    (
        "I think someone accessed my {product} account without permission.",
        "ESCALATION_RISK",
        "That needs immediate attention -- I'm escalating this to our security team right now.",
        "I'm sure it's nothing, accounts never get compromised.",
    ),
    (
        "What happens if I want to cancel my {product} subscription later?",
        "QUESTION",
        "You can cancel anytime; I'll walk you through the cancellation policy and any applicable fees.",
        "You'll never want to cancel once you see how amazing it is.",
    ),
    (
        "How long does {product} account setup usually take?",
        "QUESTION",
        "Typically same business day once identity verification is complete -- let me check current timing for you.",
        "It's instant, guaranteed, no delays ever.",
    ),
    (
        "Thanks, that explanation about {product} really helped.",
        "NEUTRAL",
        "You're welcome -- let me know if anything else comes up.",
        "Great, you owe me nothing but I promise you'll never need help again.",
    ),
    (
        "Sounds good, I'll think about the {product} option.",
        "NEUTRAL",
        "Take your time -- I'll send a written summary so you have everything in front of you.",
        "Don't overthink it, I guarantee it's the right call.",
    ),
]

PRODUCTS = [
    "standard account",
    "premium plan",
    "savings product",
    "advisory service",
    "checking account",
]


def generate_examples(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        template = rng.choice(_TEMPLATES)
        product = rng.choice(PRODUCTS)
        utterance, label, chosen, rejected = template
        examples.append(
            {
                "utterance": utterance.format(product=product),
                "label": label,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=240)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "synthetic")
    args = parser.parse_args()

    train = generate_examples(args.n_train, args.seed)
    val = generate_examples(args.n_val, args.seed + 1)
    test = generate_examples(args.n_test, args.seed + 2)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    write_jsonl(args.out_dir / "test.jsonl", test)

    labels_seen = {ex["label"] for ex in train}
    print(
        f"wrote {len(train)} train / {len(val)} val / {len(test)} test examples to {args.out_dir}"
    )
    print(f"labels present in train: {sorted(labels_seen)} (full taxonomy: {SIGNAL_LABELS})")


if __name__ == "__main__":
    main()
