# ADR 0001: LoRA/QLoRA over full fine-tuning

## Status
Accepted

## Context
Fine-tuning a base model for this task (classify a conversational signal,
suggest a compliant response) can update either every weight in the model
(full fine-tuning) or a small set of low-rank adapter matrices injected
alongside the frozen base weights (LoRA), optionally with the base weights
held in 4-bit (QLoRA).

## Decision
Train with LoRA by default (`src/sft/train_sft.py --lora-r 8 --lora-alpha
16`), with a `--use-qlora` flag for 4-bit base weights when a real GPU is
available.

See real GPU examples from same Author here:

https://github.com/manuelbomi/Enterprise-Open-Weight-LLM-Fine-Tuning-GPU-Optimization 

and here:

https://github.com/manuelbomi/EconoLLM-Tracer-Affordable-LLM-Fine-Tuning-for-Enterprise-Traceability-Data 

---

## Rationale
- **Memory.** Full fine-tuning needs optimizer state (Adam: 2 extra copies of
  every trainable parameter) sized to the *whole* model. LoRA's trainable
  parameters are typically under 1% of the base model's, so optimizer state
  shrinks by roughly the same factor -- this is what makes fine-tuning a
  multi-billion-parameter model fit on a single consumer or single
  data-center GPU at all, rather than requiring multi-GPU sharding.
- **Storage and deployment.** A LoRA adapter for a small model is a few MB
  (see the measured `adapter_model.safetensors` size in this repo's
  `artifacts/`), not a full copy of the base model. `src/serve/model_registry.py`
  exploits this directly: many task-specific adapters share one base model
  in memory, and swapping the active one is instant.
- **Rollback.** Because the base weights never change, reverting to the
  previous adapter (or to zero fine-tuning at all) is just switching which
  small file is loaded -- no risk of a corrupted full-weight checkpoint.
- **Cost of the tradeoff.** LoRA's effective capacity is bounded by its rank;
  for a task this narrow (5-way classification + short templated response)
  that ceiling isn't a practical constraint, but it would matter more for,
  say, teaching a model an entirely new domain of open-ended generation.
  QLoRA adds a second tradeoff on top: 4-bit base weights save VRAM but add
  a small amount of quantization noise and dequantize-on-the-fly compute
  overhead versus a full-precision base model.

## When to revisit
If the task grows well beyond signal classification + short templated
responses -- e.g. open-ended multi-turn dialogue generation in the target
domain -- re-evaluate whether LoRA's rank ceiling is still comfortably above
what the task needs, or whether full fine-tuning (or a higher rank) earns
back its cost.
