"""Pure, framework-free metric functions -- unit-tested against hand-picked
cases so a bug here can't silently produce a plausible-looking wrong number
in the eval report.
"""

from __future__ import annotations

import re

_LABEL_LINE_RE = re.compile(r"Label:\s*([A-Z_]+)")


def parse_completion(text: str) -> str | None:
    """Extract the predicted label from a model completion shaped like
    "Label: OBJECTION\\nResponse: ...". Returns None if the model didn't
    produce a parseable label -- this is itself a useful signal (a base
    model that hasn't been through SFT will usually fail to parse; a
    well-trained one reliably won't), surfaced in the report as a
    `parse_failure_rate`.
    """
    match = _LABEL_LINE_RE.search(text)
    return match.group(1) if match else None


def compute_label_metrics(
    predictions: list[str | None], golds: list[str], labels: list[str]
) -> dict:
    """Per-label precision/recall/F1 plus macro/micro averages and accuracy.
    An unparseable prediction (`None`) counts as wrong for every label, not
    excluded from the denominator -- a model that fails to produce a
    parseable answer should be penalized, not given a pass.
    """
    per_label = {}
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, golds, strict=True) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, golds, strict=True) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, golds, strict=True) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    accuracy = (
        sum(1 for p, g in zip(predictions, golds, strict=True) if p == g) / len(golds)
        if golds
        else 0.0
    )
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(per_label) if per_label else 0.0
    parse_failure_rate = (
        sum(1 for p in predictions if p is None) / len(predictions) if predictions else 0.0
    )

    return {
        "per_label": per_label,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "parse_failure_rate": parse_failure_rate,
        "n_examples": len(golds),
    }


def compute_preference_win_rate(chosen_logps: list[float], rejected_logps: list[float]) -> float:
    """Fraction of examples where the model assigns higher log-probability
    to the compliant (`chosen`) response than the over-promising
    (`rejected`) one. This is the DPO-relevant question: "does the model
    prefer the response we want," independent of whether it also gets the
    *label* right -- the two are evaluated separately on purpose (see
    README's evaluation section).
    """
    if not chosen_logps:
        return 0.0
    wins = sum(1 for c, r in zip(chosen_logps, rejected_logps, strict=True) if c > r)
    return wins / len(chosen_logps)
