# Experiment results

Accuracy uses conservative task-specific scoring; inspect raw answers for ambiguous cases.

| File | Method | Knowledge → truth | Group | Correct / n | Mean seconds |
|---|---|---|---|---:|---:|
| retrieval_corrected.json | tfidf | v1 → v1 | stable | 5/5 | 0.002 |
| retrieval_corrected.json | tfidf | v1 → v2 | stable | 5/5 | 0.002 |
| retrieval_corrected.json | tfidf | v1 → v1 | changed | 6/6 | 0.001 |
| retrieval_corrected.json | tfidf | v1 → v2 | changed | 0/6 | 0.001 |
| retrieval_corrected.json | tfidf | v2 → v2 | stable | 5/5 | 0.002 |
| retrieval_corrected.json | tfidf | v2 → v2 | changed | 6/6 | 0.002 |
| retrieval_corrected.json | embedding | v1 → v1 | stable | 5/5 | 0.016 |
| retrieval_corrected.json | embedding | v1 → v2 | stable | 5/5 | 0.016 |
| retrieval_corrected.json | embedding | v1 → v1 | changed | 6/6 | 0.018 |
| retrieval_corrected.json | embedding | v1 → v2 | changed | 0/6 | 0.018 |
| retrieval_corrected.json | embedding | v2 → v2 | stable | 5/5 | 0.039 |
| retrieval_corrected.json | embedding | v2 → v2 | changed | 6/6 | 0.016 |
| generation_local_scored.json | tfidf | v1 → v1 | stable | 15/15 | 1.269 |
| generation_local_scored.json | tfidf | v1 → v2 | stable | 15/15 | 1.269 |
| generation_local_scored.json | tfidf | v1 → v1 | changed | 18/18 | 1.088 |
| generation_local_scored.json | tfidf | v1 → v2 | changed | 0/18 | 1.088 |
| generation_local_scored.json | tfidf | v2 → v2 | stable | 15/15 | 1.465 |
| generation_local_scored.json | tfidf | v2 → v2 | changed | 18/18 | 1.136 |
| generation_local_scored.json | embedding | v1 → v1 | stable | 15/15 | 1.523 |
| generation_local_scored.json | embedding | v1 → v2 | stable | 15/15 | 1.523 |
| generation_local_scored.json | embedding | v1 → v1 | changed | 18/18 | 1.334 |
| generation_local_scored.json | embedding | v1 → v2 | changed | 0/18 | 1.334 |
| generation_local_scored.json | embedding | v2 → v2 | stable | 15/15 | 1.299 |
| generation_local_scored.json | embedding | v2 → v2 | changed | 18/18 | 0.843 |
| generation_local_scored.json | fewshot | v1 → v1 | stable | 9/15 | 2.412 |
| generation_local_scored.json | fewshot | v1 → v2 | stable | 9/15 | 2.412 |
| generation_local_scored.json | fewshot | v1 → v1 | changed | 15/18 | 2.118 |
| generation_local_scored.json | fewshot | v1 → v2 | changed | 0/18 | 2.118 |
| generation_local_scored.json | fewshot | v2 → v2 | stable | 12/15 | 3.433 |
| generation_local_scored.json | fewshot | v2 → v2 | changed | 12/18 | 3.463 |
| generation_local_scored.json | zero | none → v1 | stable | 0/15 | 1.380 |
| generation_local_scored.json | zero | none → v2 | stable | 0/15 | 1.380 |
| generation_local_scored.json | zero | none → v1 | changed | 0/18 | 1.103 |
| generation_local_scored.json | zero | none → v2 | changed | 0/18 | 1.103 |
| retrieval_paraphrases.json | tfidf | v1 → v1 | stable | 3/5 | 0.002 |
| retrieval_paraphrases.json | tfidf | v1 → v2 | stable | 3/5 | 0.002 |
| retrieval_paraphrases.json | tfidf | v1 → v1 | changed | 3/6 | 0.001 |
| retrieval_paraphrases.json | tfidf | v1 → v2 | changed | 0/6 | 0.001 |
| retrieval_paraphrases.json | tfidf | v2 → v2 | stable | 3/5 | 0.001 |
| retrieval_paraphrases.json | tfidf | v2 → v2 | changed | 4/6 | 0.001 |
| retrieval_paraphrases.json | embedding | v1 → v1 | stable | 4/5 | 0.013 |
| retrieval_paraphrases.json | embedding | v1 → v2 | stable | 4/5 | 0.013 |
| retrieval_paraphrases.json | embedding | v1 → v1 | changed | 4/6 | 0.013 |
| retrieval_paraphrases.json | embedding | v1 → v2 | changed | 0/6 | 0.013 |
| retrieval_paraphrases.json | embedding | v2 → v2 | stable | 4/5 | 0.017 |
| retrieval_paraphrases.json | embedding | v2 → v2 | changed | 3/6 | 0.019 |

Repeated timings are not independent test facts. With only 11 questions (5 stable, 6 changed), these results are a pilot.
Wall time and token counts are compute proxies. Dollar cost and energy were not measured.
Compare timings only on the same hardware/backend. Retrieval-only metrics are not generated-answer accuracy.
