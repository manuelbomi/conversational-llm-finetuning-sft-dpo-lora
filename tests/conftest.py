import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_adapter_dir(tmp_path_factory) -> Path:
    """A real (untrained, freshly-initialized) LoRA adapter attached to the
    tiny model and saved to disk -- built directly via peft, no training
    run, so serve-layer tests (hot-swap, generation plumbing) stay fast and
    independent of the SFT/DPO training scripts under test elsewhere.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    peft_model = get_peft_model(
        model, LoraConfig(r=2, target_modules="all-linear", task_type="CAUSAL_LM")
    )

    out_dir = tmp_path_factory.mktemp("tiny-adapter")
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir
