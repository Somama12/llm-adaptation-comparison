# Week 8–9 completed experiments

The Colab runs completed October 18, 2026 (local time); this evidence note was finalized October 19. The Weeks 1–7 midterm remains a historical record.

## Completed work

Week 8: ran TF-IDF RAG, MiniLM embedding RAG, 13-example few-shot prompting, and zero-shot generation using Qwen/Qwen2.5-3B-Instruct with the same Hugging Face NF4 evaluation backend on a Tesla T4. Saved 77 generated answers and 121 scoring rows.

Week 9: trained a Version 1 QLoRA adapter on 13 independently worded handbook examples, then evaluated 11 held-out question wordings against both original and updated truth (22 scoring rows). Training completed 20 epochs with seed 42 in 124.21 seconds. Peak allocated CUDA memory was 3,612,005,888 bytes (3.36 GiB); average training loss was 0.8343. Rank 16, alpha 32, dropout 0.05, q_proj/v_proj targets, completion-only labels, batch size 1, gradient accumulation 4, and learning rate 0.0002 were fixed before evaluation. Training used NF4 double quantization; all evaluation conditions used the same NF4 backend without double quantization.

## Audited factual accuracy

Counts are correct answers out of 11 independent synthetic facts (5 stable, 6 changed).

| Method | Original v1 | Stale v1 against v2 | Refreshed v2 |
|---|---:|---:|---:|
| TF-IDF RAG | 10/11 | 6/11 | 11/11 |
| Embedding RAG | 10/11 | 6/11 | 11/11 |
| Few-shot | 10/11 | 5/11 | 10/11 |
| Zero-shot | 1/11 | 1/11 | Not applicable |
| LoRA v1 | 5/11 | 4/11 | Week 10 pending |

LoRA scored 3/5 stable and 2/6 changed facts against v1; against v2 it scored 3/5 stable and 1/6 changed facts. Its accidental match to the new 2FA requirement is not evidence of an update: it also invents an SMS-only requirement and produces unrelated multilingual continuation. Both RAG methods answer “Yes” to the v1 2FA question even though standard accounts have optional 2FA; their stale 1/6 changed-fact match is similarly accidental. Few-shot v2 retains a 30-day refund window instead of 45 days.

## Scoring audit and evidence

All 77 baseline and 11 LoRA generated records were inspected by the assistant against the relevant facts. This is not an independent or blinded annotation study. Raw outputs are preserved unchanged. The audited baseline corrects three scoring rows: zero-shot “Daily” is accepted as every 24 hours under both truths; the stale few-shot answer limited to accounts “with administrative privileges” is rejected for the all-standard-account requirement. LoRA scoring is unchanged and measures the requested core fact; unsupported extra claims are noted above.

- `results/colab_baselines.json`: original baseline output.
- `results/colab_baselines_audited.json`: corrected scores with explicit audit overrides and source checksum.
- `results/lora_v1.json`: completed adapter evaluation.
- `results/training_v1.json`: actual training measurements.
- `results/week9_provenance.json`: runtime package versions and original Colab file/adapter SHA-256 hashes.
- `docs/week8_9_results.md`: detailed tables generated from audited evidence.

The baseline and LoRA evaluations used code commit a7d60e7d9ec517ae4bffe8801bf0c9194aa156b4. The saved LoRA evaluation is a rerun of the same trained adapter; the earlier evaluation was overwritten and is not counted as a replication. Its dirty-working-tree flag includes generated result files. Baseline and LoRA source hashes agree. Original result checksums were verified after transfer; GitHub’s added terminal newline was removed to restore the original bytes.

## Model change and limitations

Hugging Face login succeeded, but the Llama download was denied for the authenticated account. No additional user permission was needed to continue: the Colab comparison was moved to accessible Qwen 2.5 3B consistently across methods. Local Llama/Ollama results remain a separate pilot. Do not compare their latency or accuracy as a controlled same-model trial against Qwen.

This small synthetic memorization pilot holds out wording, not facts. There is one retained Colab evaluation per condition, deterministic decoding, fixed condition order, different context budgets, and no independent statistical replication. Time, tokens, and allocated memory are compute proxies; energy and dollar cost were not measured. These results do not establish a general ranking of methods.

## Remaining work and checkpoint handoff

Week 9 training and initial/stale evaluation are complete. Week 10 remains: continue the v1 adapter on v2 and measure recovery. Week 11 remains: finalize failure analysis, reproducibility, and final presentation/report. The Week 8–9 research update is available as a [PDF](reports/week8_9_research_update.pdf) and editable [LaTeX source](reports/week8_9_research_update.tex), with architecture, measured results, the scoring audit, and reproducibility appendices.

Adapter weights remain in the Colab runtime at `/content/llm-adaptation-comparison/artifacts/lora-v1`. A 16.73 MiB backup exists at `/content/week9_adapter_backup.zip`; browser downloads failed, so this archive is not yet backed up locally or in GitHub. Download it before the runtime expires. If it is lost, retrain using the fixed recipe before Week 10; the exact recorded weight hash may depend on runtime reproducibility. Raw measurements and training/provenance records are durably saved in this repository. Adapter weights are intentionally ignored by Git.
