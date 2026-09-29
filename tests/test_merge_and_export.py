"""Exercises the real merge path end-to-end on the tiny model / tiny
untrained adapter fixture -- proves `merge_and_unload()` produces a model
directory that loads with plain `AutoModelForCausalLM`, and that the
reported sizes/timings are real numbers, not placeholders.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM

from src.merge_and_export import merge_and_export


def test_merge_produces_a_plain_loadable_model(tiny_adapter_dir, tmp_path):
    out_dir = tmp_path / "merged"
    report = merge_and_export("sshleifer/tiny-gpt2", tiny_adapter_dir, out_dir)

    assert (out_dir / "config.json").exists()
    assert report["merge_seconds"] >= 0
    assert report["merged_weights_bytes"] > 0

    # No `peft` import needed to load the result -- that's the point.
    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir))
    assert reloaded is not None
