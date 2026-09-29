"""Minimal FastAPI inference server exposing the trained adapter(s).

`POST /classify` runs generation and parses out the label; `POST
/suggest_response` returns the full completion (label + suggested
response). `GET /adapters` / `POST /adapters/{name}/activate` expose the
hot-swap mechanism in `model_registry.py` over HTTP, which is what a
canary/rollback deploy (docs/PRODUCTION.md) would call.

Loading real model weights at import time would make this module slow to
merely import (e.g. in tests that don't need a loaded model) -- so the
registry is built lazily in the FastAPI `lifespan`, matching the pattern
used in the real-time-conversation-signal-copilot repo's `app/main.py`.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.format import build_prompt
from src.eval.metrics import parse_completion
from src.serve.model_registry import AdapterRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Read env vars at startup time (not module-import time) so tests can
    # `monkeypatch.setenv(...)` before constructing a `TestClient`, which
    # re-runs this lifespan on each `with TestClient(app) as client:` block.
    model_name = os.getenv("BASE_MODEL_NAME", "sshleifer/tiny-gpt2")
    default_adapter_dir = Path(os.getenv("DEFAULT_ADAPTER_DIR", "artifacts/dpo-adapter"))

    registry = AdapterRegistry(model_name)
    if default_adapter_dir.exists():
        registry.load_adapter("default", default_adapter_dir)
    app.state.registry = registry
    yield


app = FastAPI(title="Conversational Signal Classifier", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    utterance: str
    max_new_tokens: int = 24


class ClassifyResponse(BaseModel):
    label: str | None
    raw_completion: str
    adapter: str


class SuggestResponse(BaseModel):
    completion: str
    adapter: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/adapters")
async def list_adapters() -> dict[str, object]:
    registry: AdapterRegistry = app.state.registry
    return {"loaded": registry.loaded_adapters, "active": registry.active_adapter}


@app.post("/adapters/{name}/activate")
async def activate_adapter(name: str) -> dict[str, str]:
    registry: AdapterRegistry = app.state.registry
    try:
        registry.set_active(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"active": name}


def _generate(registry: AdapterRegistry, utterance: str, max_new_tokens: int) -> str:
    prompt = build_prompt(utterance)
    inputs = registry.tokenizer(prompt, return_tensors="pt")
    output = registry.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=registry.tokenizer.pad_token_id or registry.tokenizer.eos_token_id,
    )
    return registry.tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    registry: AdapterRegistry = app.state.registry
    completion = _generate(registry, request.utterance, request.max_new_tokens)
    return ClassifyResponse(
        label=parse_completion(completion),
        raw_completion=completion,
        adapter=registry.active_adapter or "",
    )


@app.post("/suggest_response", response_model=SuggestResponse)
async def suggest_response(request: ClassifyRequest) -> SuggestResponse:
    registry: AdapterRegistry = app.state.registry
    completion = _generate(registry, request.utterance, request.max_new_tokens)
    return SuggestResponse(completion=completion, adapter=registry.active_adapter or "")
