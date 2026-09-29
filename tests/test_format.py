from src.data.format import build_prompt, build_sft_target, format_dpo_example, format_sft_example


def test_build_prompt_includes_utterance_and_labels():
    prompt = build_prompt("this is too expensive")
    assert "this is too expensive" in prompt
    assert "OBJECTION" in prompt


def test_build_sft_target_shape():
    target = build_sft_target("OBJECTION", "Let's talk about the cost.")
    assert target == "Label: OBJECTION\nResponse: Let's talk about the cost."


def test_format_sft_example_produces_single_text_field():
    example = {
        "utterance": "how do I start?",
        "label": "QUESTION",
        "chosen": "Happy to help.",
        "rejected": "bad",
    }
    formatted = format_sft_example(example)
    assert set(formatted.keys()) == {"text"}
    assert "how do I start?" in formatted["text"]
    assert "Label: QUESTION" in formatted["text"]
    assert "Happy to help." in formatted["text"]


def test_format_dpo_example_produces_prompt_chosen_rejected():
    example = {
        "utterance": "too expensive",
        "label": "OBJECTION",
        "chosen": "good response",
        "rejected": "bad response",
    }
    formatted = format_dpo_example(example)
    assert set(formatted.keys()) == {"prompt", "chosen", "rejected"}
    assert "too expensive" in formatted["prompt"]
    assert "good response" in formatted["chosen"]
    assert "bad response" in formatted["rejected"]
    assert "Label: OBJECTION" in formatted["chosen"]
    assert "Label: OBJECTION" in formatted["rejected"]
