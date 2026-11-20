# Reproduce and hand off the Nimbus pilot

Python 3.11 is supported locally. Create a fresh environment, then install the training requirements and pytest to run every test (including the small CPU adapter reload test):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt pytest
python -m pip check
python -m pytest -q
python -m experiments.verify_evidence
python -m experiments.demo compare
python -m experiments.demo inspect --id changed_2
```

The evidence check recomputes the published comparison from saved raw answers and explicit audit overrides, verifies input hashes, validates unique question coverage and summaries, and reproduces all 176 passage rankings. It does not perform fresh inference or training. Historical automatic scores remain unchanged even when the current heuristic fixes a known false positive.

## Fresh local execution

Use a new output filename each time. The runner refuses to replace existing evidence. Install and start Ollama, then pull `llama3.2:3b`.

```sh
python -m experiments.run --retrieval-only --methods tfidf embedding --output artifacts/check/retrieval.json
python -m experiments.run --model llama3.2:3b --repeats 1 --output artifacts/check/generation.json
python -m experiments.report --inputs artifacts/check/retrieval.json artifacts/check/generation.json --output artifacts/check/results.md
python -m experiments.rank_audit --output artifacts/check/rank_audit.json
python -m experiments.demo ask
```

For a non-default Ollama port, add `--url http://127.0.0.1:PORT` to `experiments.run` or `experiments.demo ask`. MiniLM is downloaded on its first use. The local Llama results are separate from the controlled Qwen/T4 study. Model loading and generation may take several minutes on an 8 GB Mac.

## GPU training, continued training, reload, and export

Open `notebooks/colab_lora.ipynb` in Colab and select a T4 GPU. Run cells in order. The public Qwen checkpoint does not require a token. The notebook calls:

```sh
python -m experiments.paired_run --run-dir artifacts/runs/my-new-run
```

The directory must not exist. Each run trains v1, saves it, reloads it in a separate evaluation process, continues from that v1 adapter on v2, and reloads v2 in a separate evaluation process. Baselines use the same Qwen model and NF4 inference backend. `--skip-baselines` validates the adapter sequence without generating another baseline comparison. The fixed recipe is 20 epochs per stage, 13 training examples, seed 42, rank 16, alpha 32, dropout 0.05, learning rate 0.0002, q/v projection adapters, NF4 training, and FP16 compute.

The adjacent ZIP includes **both adapters and tokenizers**, raw responses, training metadata, runtime package versions, automatic result tables, and a checksum manifest. Download the ZIP before Colab expires. Verify the downloaded copy locally:

```sh
python -m experiments.export_run --verify /path/to/my-new-run.zip
```

After verification, extract the archive. On a CUDA machine with the training dependencies installed, reload either adapter using the same base model:

```sh
python -m experiments.run --backend hf --model Qwen/Qwen2.5-3B-Instruct --methods lora --adapter /path/to/extracted/adapters/v2 --knowledge-versions v2 --output artifacts/reloaded-v2.json
```

The base model is not included in the ZIP. Adapter weights are not a standalone model. Do not load the Qwen adapter into Llama.

## Historical evidence versus reproduction

The published comparison uses audited Week 8 baseline answers and the matched Week 12 LoRA sequence. Its original/stale/updated counts are 10/6/11 for each RAG variant, 10/5/10 for few-shot, and 5/4/4 for LoRA, all out of 11. The raw updated LoRA score was 5/11; the audit corrects a voluntary-now/required-later answer to 4/11.

The original Week 12 weight hashes are recorded, but those original weights have no verified durable backup. A newly trained adapter must be labeled a reproduction, never represented as the historical checkpoint. Fixed seeds do not guarantee identical weights, outputs, or timings across package/runtime changes. Each new run records its own provenance. Review its answers before making new accuracy claims.

The project is a small synthetic-domain research pilot, not a production chatbot. Eleven facts, one retained training sequence, overlapping training/evaluation facts, and an assistant-assisted unblinded audit limit generalization. No claim of universal superiority, statistically established differences, energy savings, or dollar cost is made.
