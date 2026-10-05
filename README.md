# LLM adaptation comparison

A controlled pilot comparing **RAG, few-shot prompting, and LoRA fine-tuning** on the synthetic Nimbus Cloud Storage handbook. The central variable is factual staleness: five evaluated facts stay constant and six change between two handbook versions.

## Implementation status

- TF-IDF and MiniLM embedding retrieval, with the same retrieval interface.
- End-to-end Llama 3.2 3B generation through a local Ollama API.
- Few-shot prompting using 13 independently worded, handbook-derived demonstrations.
- Zero-shot baseline.
- CUDA QLoRA training and evaluation scripts, including continued training on v2.
- Colab notebook for GPU execution, with same-backend baselines.
- Per-answer accuracy, retrieval matches, latency, token counts, index-build time, training time and peak CUDA allocation.

**Implemented does not mean experimentally completed.** Completed local results are in `results/` and summarized in `docs/`. LoRA training results must be produced by running the Colab notebook; no LoRA scores are invented or assumed.

## Setup and local experiments

Python 3.11 is recommended. Install Ollama from https://ollama.com/download and start its server.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m experiments.run --retrieval-only --methods tfidf embedding --output results/retrieval_corrected.json
python -m experiments.run --model llama3.2:3b --repeats 3 --output results/generation_local.json
python -m experiments.run --retrieval-only --methods tfidf embedding --qa eval/qa_paraphrases.json --output results/retrieval_paraphrases.json
python -m experiments.report --inputs results/retrieval_corrected.json results/generation_local.json results/retrieval_paraphrases.json --output docs/local_results.md
```

The first embedding run downloads `sentence-transformers/all-MiniLM-L6-v2`. The default run includes TF-IDF RAG, embedding RAG, few-shot, and zero-shot. A warm-up generation is excluded from timing. Raw answers are saved after each question; a run is marked complete only after all trials finish. Rerunning a command replaces its output, so use another filename to preserve independent runs.

For this workspace, a portable Ollama binary is in ignored `.tools/`. Start it with `OLLAMA_MODELS="$PWD/.cache/ollama" .tools/ollama serve`. Models, environments, and adapter weights are intentionally excluded from Git.

## Colab: LoRA and comparable baselines

[Open the GPU notebook](https://colab.research.google.com/github/Somama12/llm-adaptation-comparison/blob/codex/midterm-experiments/notebooks/colab_lora.ipynb).

Choose a T4 GPU runtime. Sign in to Hugging Face through the notebook and obtain access to `meta-llama/Llama-3.2-3B-Instruct` if needed. The notebook:

1. Runs TF-IDF RAG, embedding RAG, few-shot, and zero-shot using the same NF4 Hugging Face backend as LoRA.
2. Trains a v1 adapter with completion-only loss and a fixed training budget.
3. Evaluates that adapter against v1 and v2 truth using the same generated answers.
4. Continues training the v1 adapter on v2, then evaluates recovery.
5. Downloads raw evaluation results and training measurements.

Do not compare Mac/Ollama latency directly with Colab/NF4 latency, or attribute backend differences to adaptation strategy. The main three-way comparison should use the Colab baselines and LoRA results together. The notebook requires a user-authenticated GPU session; its training recipe is not locally GPU-validated on the 8 GB Mac.

## Experimental design and limits

For RAG, “knowledge version” is the indexed handbook. For few-shot, it is the demonstration version. For LoRA, it is the training/update data version. We score deployment knowledge against v1 truth, the same answers against v2 truth (stale), and updated knowledge against v2 truth (recovery). The zero-shot control has no handbook knowledge.

Adaptation questions do not duplicate the 11 evaluation question strings. The facts deliberately overlap: this measures factual acquisition and retention with held-out wording, not generalization to unseen facts. Few-shot includes all 13 handbook sections, so it is a full-coverage in-context baseline. Its context cost is recorded. The extra 11 paraphrases probe the same facts and are not 11 new independent observations. They are exploratory, authored after the original baseline.

Three deterministic repeats provide timing observations; they do not increase the independent sample size beyond 11 facts. Runs use a fixed condition order and may benefit from prompt caching. Small denominators and a synthetic domain limit external validity. Energy and dollar cost are not measured; time, tokens, and memory are compute proxies.

Scoring is a conservative, task-specific factual heuristic, not a semantic LLM judge. It requires full numeric tokens and units, rejects competing old/new values, and handles the scope of the 2FA requirement. Raw outputs should be audited for paraphrases, negations, and ambiguity. Numerical scoring alone cannot establish entailment.

### Correction to the Report 1 score

The historical `results/retrieval_eval_results.json` remains unchanged for traceability. Its 1/6 stale changed-fact score was a false positive: the old checker matched “mandatory” in the v1 passage about **admin** accounts while evaluating **standard** accounts. Corrected scoring gives **0/6**. Both old and new code retrieve the relevant section; this was an answer-scoring error. State this correction explicitly in the midterm rather than presenting the old 17% as a real success rate.

## Repository map

- `data/`: v1/v2 source handbooks
- `eval/`: original and exploratory paraphrased questions; compatibility entry point
- `rag/`: TF-IDF and embedding retrievers
- `experiments/`: prompts, model backends, scoring, runner, result tables
- `finetune/`: CUDA QLoRA training
- `notebooks/`: Colab workflow
- `results/`: actual raw measurements and historical results
- `tests/`: scoring regressions, retrieval, prompt separation, error handling
- `docs/`: evidence and midterm notes

Run tests with `pip install pytest` then `python -m pytest -q`. `requirements-local.lock.txt` records the local environment; `requirements-train.txt` adds the CUDA training dependencies. The Mac lock is not a CUDA lock.
