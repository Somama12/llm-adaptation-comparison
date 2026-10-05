# Local timing and token use

Warm runs; CPU inference on an 8 GiB Mac; three deterministic repeats. Prompt caching and fixed order limit causal interpretations.

| Method | Knowledge | Mean end-to-end seconds | Mean prompt tokens | Mean output tokens |
|---|---|---:|---:|---:|
| tfidf | v1 | 1.170 | 108.1 | 4.5 |
| tfidf | v2 | 1.285 | 108.9 | 4.4 |
| embedding | v1 | 1.420 | 108.1 | 4.5 |
| embedding | v2 | 1.050 | 108.9 | 4.4 |
| fewshot | v1 | 2.252 | 640.5 | 17.6 |
| fewshot | v2 | 3.449 | 649.5 | 19.1 |
| zero | none | 1.229 | 78.5 | 6.0 |
