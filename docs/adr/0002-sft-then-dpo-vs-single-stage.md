# ADR 0002: Two-stage SFT → DPO over single-stage DPO, and DPO over PPO/RLHF

## Status
Accepted

## Context
There are several ways to get from a general-purpose base model to one that
reliably classifies a signal and prefers a compliant response: supervised
fine-tuning (SFT) alone, preference optimization alone, or SFT followed by
preference optimization. Preference optimization itself can be done with
PPO-based RLHF (a reward model + reinforcement learning loop) or with Direct
Preference Optimization (DPO, a single supervised-style loss over
chosen/rejected pairs, no separate reward model or RL loop).

## Decision
Run SFT first (`src/sft/train_sft.py`), then continue the *same* LoRA
adapter with DPO (`src/dpo/train_dpo.py --sft-adapter-dir artifacts/sft-adapter`).
Use DPO, not PPO/RLHF, for the preference stage.

## Rationale
- **Why SFT before preference optimization at all.** DPO's loss compares the
  model's log-probability of a `chosen` vs. `rejected` response *given the
  same prompt*. A base model that has never seen this repo's `Label:
  ...\nResponse: ...` output format assigns near-random, noisy probability
  to both -- there's no reliable signal to sharpen yet. SFT's job is
  narrower and easier: teach the format and the task via next-token
  prediction, so DPO then has a policy that's already *close* to producing
  parseable, on-task output, and only needs to shift a preference between
  two plausible completions.
- **Why DPO over PPO/RLHF.** PPO-based RLHF needs a separately trained
  reward model, then an online RL loop sampling from the policy, scoring
  with the reward model, and updating with a policy-gradient method --
  meaningfully more infrastructure, more hyperparameters (KL penalty
  coefficient, reward normalization, rollout batch size, ...), and more
  ways for training to silently destabilize. DPO reformulates the same
  chosen-vs-rejected objective as a closed-form loss computed directly from
  the policy's own log-probabilities (see `sequence_logprob` in
  `src/eval/evaluate.py` for the same quantity computed standalone for
  eval) -- no reward model, no rollout loop, no RL-specific tuning. For a
  well-scoped preference (favor a specific compliant response style over a
  specific over-promising one, both already known at data-authoring time --
  see `scripts/generate_synthetic_data.py`), DPO gets the same practical
  outcome with a fraction of the moving parts.
- **Why continue the same adapter rather than train two separate ones.**
  `src/dpo/train_dpo.py` loads the SFT adapter with `is_trainable=True` and
  keeps training it, rather than starting a fresh LoRA adapter for DPO. DPO
  is a refinement of the SFT policy's preferences, not an unrelated task --
  starting from the SFT adapter (instead of the raw base model) means DPO
  only has to *nudge* an already-reasonable policy, not rediscover the
  output format from scratch under a very different loss.

## When to revisit
If preference data volume grows large enough that a learned reward model
would generalize meaningfully better than DPO's fixed chosen/rejected pairs
(DPO has no mechanism to interpolate a "how much better" reward signal
beyond the pairwise comparison), or if the task needs online exploration
(sampling novel responses and scoring them) rather than optimizing against a
fixed offline preference dataset, PPO-based RLHF becomes worth its added
complexity.
