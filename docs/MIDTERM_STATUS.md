# Midterm implementation and evidence

Prepared October 5, 2026. This is an evidence brief for drafting the midterm, not a claim that the three-way study is finished.

## Work completed after Report 2

The repository now contains a working MiniLM embedding retriever and a complete Llama 3.2 3B RAG pipeline served through Ollama. The full initial/stale/updated evaluation has been run for TF-IDF RAG, embedding RAG, and a 13-demonstration few-shot condition. A zero-shot control and a separate exploratory paraphrase retrieval test are also included. Each local generation condition was repeated three times; all repeated answer strings were identical. Thus the independent sample remains 11 questions, not 33.

The evaluation saves generated answers, reference answers, factual scores, retrieval matches, retrieval similarity, retrieval and total latency, token counts, index setup time, model identity, and software versions. Local measurements used an 8 GiB Apple Silicon Mac. Ollama 0.35.1 fell back to CPU inference after GPU discovery timed out. These measurements should not be described as GPU timings.

## Local generated-answer results

Each entry below counts unique questions; the three repeats gave the same answers.

| Method | Initial v1: stable / changed | Stale v1→v2: stable / changed | Updated v2: stable / changed |
|---|---|---|---|
| TF-IDF RAG | 5/5; 6/6 | 5/5; 0/6 | 5/5; 6/6 |
| Embedding RAG | 5/5; 6/6 | 5/5; 0/6 | 5/5; 6/6 |
| Few-shot | 3/5; 5/6 | 3/5; 0/6 | 4/5; 4/6 |
| Zero-shot | 0/5; 0/6 | 0/5; 0/6 | No knowledge update applies |

Zero-shot abstained on every question. Few-shot made concrete factual errors despite demonstrations containing the correct facts: for example, it gave 2 business days for v1 refund processing instead of 5, and retained the 30-day refund window in the v2 condition instead of 45. Updating demonstrations did not guarantee recovery on every fact. RAG recovered all six changed facts after re-indexing on this particular dataset.

The current evidence supports RAG's staleness-and-recovery pattern under controlled conditions. It does not establish that RAG is universally superior, that embeddings outperform lexical retrieval, or that LoRA will behave in a particular way.

## Scoring correction and audit

Report 1's stale changed-fact result of 1/6 (17%) was caused by the original scorer matching the word “mandatory” in a sentence about admin accounts while evaluating standard employee accounts. Corrected scoring gives 0/6. The historical JSON is retained unchanged.

During generated-answer validation, the scorer was extended to recognize a concise “No” as a correct v1 response to the mandatory-2FA question. Saved answers were re-scored without regenerating them. `generation_local.json` retains the original scores; `generation_local_scored.json` contains the corrected scores and provenance. All distinct local generated answers were inspected against the relevant facts; the corrected scores agree with that inspection. This was a developer audit, not an independent or blinded annotation study.

## Embeddings versus TF-IDF

Both retrievers score perfectly on the original questions with matching handbook versions. The separately authored paraphrase set is more difficult. Neither retriever is uniformly better across versions. Treat the paraphrase results as exploratory, and inspect both section retrieval and factual-content scores: these measure different things.

## LoRA implementation and outstanding evidence

The CUDA QLoRA code uses the Llama 3.2 3B Instruct base model, NF4 weights, rank-16 adapters on q_proj/v_proj, completion-only loss, a fixed 20-epoch budget, and seed 42. Training examples are independently worded questions paired with handbook section text; the evaluation question strings are not used in training. The notebook trains v1, evaluates initial and stale performance, continues the adapter on v2, and evaluates updated performance. It records training time and peak allocated CUDA memory.

The training dataset masking and adapter gradients passed a tiny random-model CPU smoke test. This does not validate CUDA execution or demonstrate task learning. The Colab T4 session and dependencies have been set up; actual model access, training, and same-backend comparison remain pending Hugging Face authentication. No LoRA accuracy, training time, or comparative conclusion is available yet.

## Limitations to retain in the midterm

- Eleven synthetic facts, with only six changing; no broad-domain or statistical superiority claim.
- Held-out question wording but deliberately overlapping facts across adaptation and evaluation.
- Few-shot includes all 13 sections, with longer context than single-chunk RAG.
- Fixed method order, deterministic decoding, and potential prompt-cache effects on timing.
- Task-specific scoring rather than a general semantic judge; audit raw answers.
- Local Ollama and Colab NF4 differ in hardware and quantization; use Colab baselines for the main three-way comparison.
- Token counts, wall time, and memory are compute proxies. No energy or dollar cost was measured.

## Files to use

`docs/local_results.md` contains detailed result tables. `results/generation_local_scored.json` is the authoritative local generated-answer result. `results/retrieval_corrected.json` and `results/retrieval_paraphrases.json` contain retrieval trials. `notebooks/colab_lora.ipynb` is the GPU workflow. `README.md` contains reproduction commands and design notes.
