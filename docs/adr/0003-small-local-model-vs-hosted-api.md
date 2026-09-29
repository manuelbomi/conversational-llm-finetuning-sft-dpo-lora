# ADR 0003: A small, locally fine-tuned model over a large hosted-API-only model

## Status
Accepted

## Context
The classification + short-response task this repo trains for (5-way
signal label + one templated-style compliant sentence) could instead be
solved by prompting a large hosted API model (few-shot or zero-shot, no
fine-tuning) at inference time.

## Decision
Fine-tune a small, locally-hosted open-weight model (this repo defaults to
`sshleifer/tiny-gpt2` for fast CPU iteration in tests/tutorials; see
README.md for the realistic small-production-model recommendation, e.g. a
~0.5B-1.5B instruct model) rather than relying solely on a large hosted API
model at inference time.

## Rationale
- **Latency and cost at call-center scale.** A signal classifier that runs
  once per conversational turn, potentially across thousands of concurrent
  live conversations (see the sibling `real-time-conversation-signal-copilot`
  and `low-latency-llm-serving-benchmark` repos), pays its inference cost
  and latency on every single turn. A small model served locally with
  batching (see `low-latency-llm-serving-benchmark`) has a fundamentally
  different cost and p99-latency profile than a per-turn call to a large
  hosted API -- both in dollars and in round-trip network latency, which
  matters directly for a "live" user-facing budget.
- **Data control.** Fine-tuning data here is synthetic and generic on
  purpose (see `scripts/generate_synthetic_data.py`'s docstring), but in a
  real deployment this pipeline's input would be real (redacted) customer
  conversation excerpts. Keeping training and inference on infrastructure
  you control avoids sending that data to a third-party API as a matter of
  course, which is a meaningfully simpler data-governance story.
- **A narrow, well-specified task doesn't need a frontier model's breadth.**
  This task is a fixed 5-way label plus a short, templated-style response --
  not open-ended reasoning or broad world knowledge. A small model, fine-
  tuned specifically for this narrow task, can match a much larger
  general-purpose model's quality *on this task specifically*, at a fraction
  of the serving cost. This is a bet worth validating empirically per task
  (see `src/eval/evaluate.py` and the companion `golden-dataset-eval-framework`
  repo's evaluation-gate pattern) rather than assumed universally.
- **Cost of the tradeoff.** A small fine-tuned model is worse than a large
  hosted model at anything *outside* its narrow training distribution --
  it will not gracefully handle a wildly out-of-scope utterance the way a
  frontier model's broader training might. This repo's `parse_failure_rate`
  metric (`src/eval/metrics.py`) is one honest signal for catching that
  failure mode in evaluation.

## When to revisit
If the task's scope grows beyond fixed-taxonomy classification and short
templated responses -- e.g. open-ended free-text advice generation needing
broad world knowledge -- re-evaluate whether a small fine-tuned model's
narrow competence still covers the real distribution of inputs, or whether
a hosted frontier model (possibly *guiding* a smaller model via distillation
into a future fine-tuning round) is the better fit.
