# Week 12 paired update run

This directory is separate from the original Week 8–9 evidence. The Colab notebook rebuilds a v1 adapter and continues training it on v2. Use the *same session's* v1 and v2 files for the before/after comparison. The original audited baselines in the parent directory retain their historical provenance.

- `lora_v1.json`, `lora_v2.json`: per-answer output and condition summaries.
- `training_v1.json`, `training_v2.json`: elapsed training time, peak allocated memory, and settings.
- `runtime.json`: package versions and adapter weight hashes.
- `colab_baselines.json`: automatic baseline scores from the rerun; audit raw answers before interpreting.
- `pairs.json`: both answers, version-two reference and automatic core-fact scores keyed by question ID.

Run `python -m experiments.update_audit results/week12` to regenerate the pairs. For the separate local retrieval experiment, run `python -m experiments.rank_audit`. Its 176 ranking records reuse the same 11 facts and are not independent answer trials.

The November 9 run is complete. Reviewing all eleven answer pairs against the v2 core fact changes one automatic judgment: the updated answer to changed_2 calls two-factor authentication voluntary now, while the handbook already requires it. See `pairs_audited.json`. Stale vs. updated: changed policies 1/6 → 2/6; stable policies 3/5 → 2/5; overall 4/11 → 4/11. Raw automatic updated accuracy is 5/11. The update fixed password reset and backup retention but lost the Pro Plan capacity and the core 2FA requirement. Core-fact credit does not endorse extra details in an answer.
