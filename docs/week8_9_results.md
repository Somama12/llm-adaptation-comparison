# Experiment results

Accuracy uses conservative task-specific scoring; inspect raw answers for ambiguous cases.

| File | Method | Knowledge → truth | Group | Correct / n | Mean seconds |
|---|---|---|---|---:|---:|
| colab_baselines_audited.json | tfidf | v1 → v1 | stable | 5/5 | 0.521 |
| colab_baselines_audited.json | tfidf | v1 → v2 | stable | 5/5 | 0.521 |
| colab_baselines_audited.json | tfidf | v1 → v1 | changed | 5/6 | 0.340 |
| colab_baselines_audited.json | tfidf | v1 → v2 | changed | 1/6 | 0.340 |
| colab_baselines_audited.json | tfidf | v2 → v2 | stable | 5/5 | 0.510 |
| colab_baselines_audited.json | tfidf | v2 → v2 | changed | 6/6 | 0.337 |
| colab_baselines_audited.json | embedding | v1 → v1 | stable | 5/5 | 0.514 |
| colab_baselines_audited.json | embedding | v1 → v2 | stable | 5/5 | 0.514 |
| colab_baselines_audited.json | embedding | v1 → v1 | changed | 5/6 | 0.354 |
| colab_baselines_audited.json | embedding | v1 → v2 | changed | 1/6 | 0.354 |
| colab_baselines_audited.json | embedding | v2 → v2 | stable | 5/5 | 0.525 |
| colab_baselines_audited.json | embedding | v2 → v2 | changed | 6/6 | 0.365 |
| colab_baselines_audited.json | fewshot | v1 → v1 | stable | 5/5 | 1.298 |
| colab_baselines_audited.json | fewshot | v1 → v2 | stable | 5/5 | 1.298 |
| colab_baselines_audited.json | fewshot | v1 → v1 | changed | 5/6 | 1.325 |
| colab_baselines_audited.json | fewshot | v1 → v2 | changed | 0/6 | 1.325 |
| colab_baselines_audited.json | fewshot | v2 → v2 | stable | 5/5 | 1.334 |
| colab_baselines_audited.json | fewshot | v2 → v2 | changed | 5/6 | 1.400 |
| colab_baselines_audited.json | zero | none → v1 | stable | 1/5 | 0.296 |
| colab_baselines_audited.json | zero | none → v2 | stable | 1/5 | 0.296 |
| colab_baselines_audited.json | zero | none → v1 | changed | 0/6 | 0.326 |
| colab_baselines_audited.json | zero | none → v2 | changed | 0/6 | 0.326 |
| lora_v1.json | lora | v1 → v1 | stable | 3/5 | 1.819 |
| lora_v1.json | lora | v1 → v2 | stable | 3/5 | 1.819 |
| lora_v1.json | lora | v1 → v1 | changed | 2/6 | 2.772 |
| lora_v1.json | lora | v1 → v2 | changed | 1/6 | 2.772 |

Repeated timings are not independent test facts. With only 11 questions (5 stable, 6 changed), these results are a pilot.
Wall time and token counts are compute proxies. Dollar cost and energy were not measured.
Compare timings only on the same hardware/backend. Retrieval-only metrics are not generated-answer accuracy.
