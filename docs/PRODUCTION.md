# Production deployment notes

This repo trains a LoRA/QLoRA adapter through SFT then DPO on CPU-feasible
synthetic data, purely to make the *pipeline mechanics* runnable end-to-end
by anyone who clones it. Everything below is what changes to run this for
real, at production scale.

## Dataset versioning

`scripts/generate_synthetic_data.py` is deterministic given a seed, which is
fine for a tutorial but not a substitute for tracking a real dataset's
lineage. In production:

- Version `data/` with [DVC](https://dvc.org/) (or an equivalent — LakeFS,
  a versioned object-store prefix) pointing at object storage, so a training
  run's `dvc.lock` records the exact data hash used, not just a seed.
- Replace the synthetic generator's narrow templates with real, redacted
  conversation excerpts and SME-reviewed labels. The companion
  `golden-dataset-eval-framework` repo's annotation pipeline (multi-annotator
  agreement via Cohen's/Fleiss' kappa, adjudication, versioned "golden"
  export) is exactly the upstream this data should come from — this repo's
  `data/make_dataset.py`-equivalent is a stand-in for that pipeline's output
  format, not a replacement for it.

## Experiment tracking

Wire an [MLflow](https://mlflow.org/) run around each `train_sft.py` /
`train_dpo.py` invocation: log the CLI args, the LoRA config, `training_loss`
per logging step, and the final adapter as an MLflow artifact. This turns
"what hyperparameters produced adapter X" from a question you answer by
reading shell history into one MLflow answers directly, and is a prerequisite
for the promotion gate below (which needs to reference a specific tracked
run, not just a directory on disk).

## Model registry and promotion gate

`src/serve/model_registry.py`'s `AdapterRegistry` already supports the
production shape: load a new adapter under a name, keep the previous one
loaded, and flip `active_adapter` with `set_active()` — no base-model reload,
no downtime. What a production pipeline adds on top:

1. A new adapter is a *candidate*, not automatically active.
2. Run `python src/eval/evaluate.py --adapter-dir <candidate>` (or the
   real-data equivalent) and feed the report into an eval-gate script —
   compare directly to the companion `golden-dataset-eval-framework` repo's
   `eval_gate.py`, which does exactly this: computes per-class
   precision/recall/F1 and calibration against a golden set and exits
   non-zero if any configured threshold regresses.
3. Only a CI job whose eval-gate step passed is allowed to call
   `POST /adapters/{name}/activate` against the serving registry — wire this
   as a required check in the promotion pipeline, the same
   evaluate-then-gate-then-promote pattern as this account's other
   conversational-AI repos.
4. Roll back by calling `POST /adapters/{previous_name}/activate` — since
   the previous adapter is still loaded (not unloaded on promotion), rollback
   is a single API call, not a redeploy.

## Reproducibility

`train_sft.py` and `train_dpo.py` don't currently pin a random seed for
`transformers`/`torch`/`datasets` — for tutorial-scale runs the variance
doesn't matter, but a production training script should call
`transformers.set_seed(seed)` once at the top of `main()` and log the seed
alongside the MLflow run, so a flagged model can be exactly reproduced for
debugging or audit.

## Safety review before promotion

Before a candidate adapter is promoted, run it against the guardrail
regression suite from the companion `real-time-conversation-signal-copilot`
repo's `guardrails/` tests (compliance blocklist + LLM-as-judge check) — a
model that classifies signals well but has started generating a
policy-violating response style should fail promotion even if its label F1
looks fine. Label quality and response-style compliance are evaluated
separately in `src/eval/metrics.py` for exactly this reason (see that
module's docstrings).

## Canary rollout and automatic rollback

Once an adapter passes the gate, activate it for a small percentage of
traffic before activating it globally — e.g. route by a hash of the
conversation ID so the same conversation stays on one adapter for its
duration, with N% of new conversation IDs routed to the candidate. Monitor
the live `parse_failure_rate` and label distribution (via the drift check in
the companion `golden-dataset-eval-framework` repo's `app/drift.py`) on the
canary slice; auto-revert to the previous adapter (a single
`set_active(previous_name)` call) if either regresses past a threshold
before ramping to 100%.

## Sizing: LoRA rank vs. VRAM

Actual numbers depend heavily on base model size, sequence length, batch
size, and optimizer choice — but as a rule of thumb, LoRA/QLoRA's *added*
memory over just holding the base model in memory scales with `r` (rank)
and the number of target modules, not with the base model's full parameter
count, which is the entire point of the method (see ADR 0001). Concretely,
for a ~1-3B parameter instruct model:

| Setup | Base weights | Typical single-GPU fit |
|---|---|---|
| Full fine-tune, fp16 | ~2-6 GB | Needs a large GPU (24-80 GB) once optimizer state (Adam ≈ 2x params) is added |
| LoRA, fp16/bf16 base, r=8-16 | ~2-6 GB | Fits comfortably on a 16-24 GB GPU; adapter optimizer state is small |
| QLoRA, 4-bit base, r=8-16 | ~1-3 GB | Fits on a 8-16 GB GPU |

(Reasoning follows the PEFT/QLoRA papers' parameter-count argument, not a
benchmark run in this repo — no GPU is available in this sandbox. Validate
against your actual base model and sequence length before committing to
hardware.)

## From adapter to serving

`src/merge_and_export.py` merges a finalized adapter into the base weights
and exports a plain (non-PEFT) model directory — the shape the sibling
`low-latency-llm-serving-benchmark` repo's serving stack (and vLLM/TGI/
TensorRT-LLM generally) expects for highest-throughput serving. Keep
unmerged adapters (`AdapterRegistry`) for iteration and instant rollback;
merge only once an adapter is finalized and headed into a high-throughput
serving path where the extra `peft` indirection isn't worth its (small)
runtime cost.
