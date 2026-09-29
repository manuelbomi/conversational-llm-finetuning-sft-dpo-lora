"""The fixed label vocabulary this tutorial trains a model to predict.

Kept as a plain list (not an enum tied to a specific framework) because this
module is imported by the data generator, the SFT/DPO training scripts, and
the eval script -- it's the one place that defines "what a correct answer
looks like" for all of them.
"""

from __future__ import annotations

SIGNAL_LABELS: list[str] = [
    "OBJECTION",
    "BUYING_SIGNAL",
    "ESCALATION_RISK",
    "QUESTION",
    "NEUTRAL",
]

INSTRUCTION_TEMPLATE = (
    "Classify the conversational signal in this customer utterance as exactly "
    "one of {labels}, then suggest a one-sentence compliant response.\n\n"
    "Utterance: {utterance}"
)
