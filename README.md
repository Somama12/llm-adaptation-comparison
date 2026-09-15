# LLM Adaptation Comparison: Fine-Tuning vs. Few-Shot Prompting vs. RAG

A comparative evaluation of three LLM adaptation strategies — fine-tuning, few-shot prompting, and retrieval-augmented generation (RAG) — for domain-specific question answering, with a specific focus on **staleness-robustness**: how each approach handles information that changes after initial deployment.

## Research Question
For a narrow, well-defined QA task, how do fine-tuning, few-shot prompting, and RAG compare in accuracy, latency, and computational cost — and, more importantly, how does each approach handle information that changes after initial setup?

## Experimental Design
To directly test staleness-robustness under controlled conditions, this project uses a **synthetic domain** rather than real-world documentation: a fictional company's ("Nimbus Cloud Storage") internal IT/support policy handbook.

- `data/handbook_v1.md` — the handbook as it exists at initial system deployment
- `data/handbook_v2.md` — the same handbook 2.5 months later, with 6 specific facts deliberately changed (password policy, refund window, backup retention, support SLA, storage plan, 2FA requirement) and 5 facts left unchanged as controls

This gives full control over exactly which facts are "stale" vs "current" at evaluation time — something real-world docs can't offer on a course project timeline.

## Repository Structure
```
data/           synthetic handbook v1 (initial) and v2 (post-update)
eval/           held-out QA pairs (qa_pairs.json) + evaluation harness (run_harness.py)
rag/            RAG retriever implementation (retriever.py)
results/        saved evaluation outputs (JSON)
docs/           progress reports, notes
```

## Current Status: RAG Retrieval Baseline (Working)

Implemented a TF-IDF-based retriever (`rag/retriever.py`) that:
1. Chunks the handbook markdown by subsection
2. Builds a TF-IDF index over chunks
3. Retrieves the top-matching chunk for a given question via cosine similarity

### Staleness Experiment Results (`eval/run_harness.py`)

Three conditions were run against 11 held-out QA pairs (5 stable facts, 6 changed facts):

| Condition | Stable Accuracy | Changed-Fact Accuracy |
|---|---|---|
| v1-indexed RAG vs. v1 ground truth (sanity check) | 5/5 (100%) | 6/6 (100%) |
| **v1-indexed RAG vs. v2 ground truth (staleness exposure)** | 5/5 (100%) | **1/6 (17%)** |
| v2-indexed RAG vs. v2 ground truth (post re-index) | 5/5 (100%) | 6/6 (100%) |

**Interpretation:** When the underlying policy changes but the RAG index is not refreshed, retrieval on stable facts remains perfect, but retrieval on changed facts collapses to 17% — the retriever confidently returns outdated section text. Simply re-indexing on the updated document (no retraining, no fine-tuning) restores 100% accuracy immediately. This is the core mechanic this project will now compare against fine-tuning and few-shot prompting, where "fixing" staleness requires either retraining or manually updating prompt examples.

## Next Steps
- [ ] Add embedding-based retrieval (sentence-transformers) as a stronger RAG variant
- [ ] Wire retrieved context into an actual local LLM (Llama 3.2 3B / Phi-3-mini via Ollama) for full generation, not just retrieval
- [ ] Build the few-shot prompting variant using the same QA pairs
- [ ] Build the fine-tuning (LoRA) variant on Google Colab
- [ ] Run all three variants against the same staleness eval and compare accuracy/latency/cost
