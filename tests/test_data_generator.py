import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_synthetic_data import generate_examples
from src.taxonomy import SIGNAL_LABELS


def test_generates_requested_count():
    examples = generate_examples(50, seed=1)
    assert len(examples) == 50


def test_every_example_has_required_fields():
    examples = generate_examples(20, seed=2)
    for ex in examples:
        assert set(ex.keys()) == {"utterance", "label", "chosen", "rejected"}
        assert ex["label"] in SIGNAL_LABELS
        assert ex["utterance"]
        assert ex["chosen"] != ex["rejected"]


def test_deterministic_given_same_seed():
    a = generate_examples(30, seed=7)
    b = generate_examples(30, seed=7)
    assert a == b


def test_different_seeds_produce_different_data():
    a = generate_examples(30, seed=7)
    b = generate_examples(30, seed=8)
    assert a != b


def test_all_labels_appear_with_enough_examples():
    examples = generate_examples(500, seed=3)
    labels_seen = {ex["label"] for ex in examples}
    assert labels_seen == set(SIGNAL_LABELS)
