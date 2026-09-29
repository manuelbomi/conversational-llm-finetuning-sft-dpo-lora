"""Loads a base model once and lets multiple LoRA adapters be attached,
swapped, and queried by name -- without reloading the (much larger) base
model weights for every version. This is the mechanism a real deployment's
"adapter versioning" (see docs/PRODUCTION.md) is built on: promote a new
adapter by calling `load_adapter(name, path)`, roll back by calling
`set_active(previous_name)`, no base-model reload either way.
"""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


class AdapterRegistry:
    def __init__(self, base_model_name: str) -> None:
        self._base_model_name = base_model_name
        self._base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PeftModel | None = None
        self._active_adapter: str | None = None
        self._known_adapters: dict[str, Path] = {}

    @property
    def active_adapter(self) -> str | None:
        return self._active_adapter

    @property
    def loaded_adapters(self) -> list[str]:
        return list(self._known_adapters.keys())

    def load_adapter(self, name: str, path: Path, *, activate: bool = True) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"adapter path does not exist: {path}")
        self._known_adapters[name] = path

        if self._model is None:
            self._model = PeftModel.from_pretrained(self._base_model, path, adapter_name=name)
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        else:
            self._model.load_adapter(str(path), adapter_name=name)

        if activate:
            self.set_active(name)

    def set_active(self, name: str) -> None:
        if name not in self._known_adapters:
            raise KeyError(f"adapter '{name}' is not loaded; call load_adapter() first")
        self._model.set_adapter(name)
        self._active_adapter = name

    @property
    def model(self) -> PeftModel:
        if self._model is None:
            raise RuntimeError("no adapter loaded yet")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("no adapter loaded yet")
        return self._tokenizer
