"""Run with python -m experiments.run; all paths resolve from the repository."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from experiments.data import ROOT, messages
from experiments.models import OllamaModel, HFModel
from experiments.scoring import score_answer
from rag.retriever import chunk_markdown, TfidfRetriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", choices=["tfidf", "embedding", "fewshot", "zero", "lora"], default=["tfidf", "embedding", "fewshot", "zero"])
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--adapter")
    p.add_argument("--knowledge-versions", nargs="+", choices=["v1", "v2"], default=["v1", "v2"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--qa", type=Path, default=ROOT / "eval/qa_pairs.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    if args.retrieval_only and any(m not in ["tfidf", "embedding"] for m in args.methods):
        p.error("retrieval-only supports tfidf and embedding")
    if "lora" in args.methods and (not args.adapter or args.backend != "hf" or len(args.methods) != 1 or len(args.knowledge_versions) != 1):
        p.error("LoRA requires --backend hf --adapter PATH and a single method and knowledge version")
    if args.adapter and args.methods != ["lora"]:
        p.error("An adapter can only be used with --methods lora")
    qa_pairs = json.loads(args.qa.read_text())
    model = None if args.retrieval_only else (OllamaModel(args.model) if args.backend == "ollama" else HFModel(args.model, args.adapter))
    metadata = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for folder in ["experiments", "rag", "finetune"] for f in sorted((ROOT / folder).glob("*.py"))},
        "qa_sha256": hashlib.sha256(args.qa.read_bytes()).hexdigest(),
        "handbook_sha256": {v: hashlib.sha256((ROOT / f"data/handbook_{v}.md").read_bytes()).hexdigest() for v in args.knowledge_versions},
        "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scikit-learn", "sentence-transformers"]},
        "model": model.metadata if model else None, "repeats": args.repeats,
        "cost_note": "Local wall time and token counts are compute proxies; energy and dollar cost were not measured.",
        "warmup": "One untimed generation before trials" if model else None}
    if model:
        model.generate(messages("Say ready.", "zero", "v1"))
    rows, setup = [], []
    encoder = None
    if "embedding" in args.methods:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        metadata["embedding_model_load_seconds"] = time.perf_counter()-start
        metadata["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps({"metadata": metadata, "setup": setup, "rows": rows, "summary": summarize(rows)}, indent=2))
    for method in args.methods:
        for version in args.knowledge_versions:
            if method == "zero" and version != args.knowledge_versions[0]:
                continue
            start = time.perf_counter()
            retriever = None
            if method in ["tfidf", "embedding"]:
                chunks = chunk_markdown(str(ROOT / f"data/handbook_{version}.md"))
                if method == "tfidf":
                    retriever = TfidfRetriever(chunks)
                else:
                    from rag.embedding import EmbeddingRetriever
                    retriever = EmbeddingRetriever(chunks, encoder=encoder)
            setup.append({"method": method, "knowledge_version": version, "index_seconds": time.perf_counter()-start})
            for repeat in range(args.repeats):
                for qa in qa_pairs:
                    start = time.perf_counter()
                    context, section, similarity = None, None, None
                    if retriever:
                        chunk, similarity = retriever.retrieve(qa["question"])[0]
                        context, section = chunk.text, chunk.section_id
                    retrieval_seconds = time.perf_counter()-start
                    output = {"answer": context} if args.retrieval_only else model.generate(messages(qa["question"], method, version, context))
                    elapsed = time.perf_counter()-start
                    # Score the SAME answer against both truths; do not regenerate for stale scoring.
                    for truth in ["v1", "v2"]:
                        if version == "v2" and truth == "v1" and method != "zero":
                            continue
                        rows.append({"method": method, "knowledge_version": version if method != "zero" else "none",
                            "truth_version": truth, "repeat": repeat, "id": qa["id"], "category": qa["category"],
                            "question": qa["question"], "expected": qa[f"answer_{truth}"],
                            "correct": score_answer(qa, output["answer"], truth), "retrieved_section": section,
                            "section_match": section == qa["section"] if section else None,
                            "similarity": similarity, "retrieval_seconds": retrieval_seconds,
                            "end_to_end_seconds": elapsed, **output})
                    save()
                print(f"Completed {method} {version} repeat {repeat+1}", flush=True)
    metadata["status"] = "complete"
    save()


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["knowledge_version"], row["truth_version"], row["category"])
        groups.setdefault(key, []).append(row)
    return [{"method": k[0], "knowledge_version": k[1], "truth_version": k[2], "category": k[3],
        "n": len(v), "correct": sum(r["correct"] for r in v),
        "accuracy": statistics.mean(r["correct"] for r in v),
        "mean_seconds": statistics.mean(r["end_to_end_seconds"] for r in v),
        "median_seconds": statistics.median(r["end_to_end_seconds"] for r in v),
        "unique_questions": len({r["id"] for r in v}),
        "mean_prompt_tokens": statistics.mean(r.get("prompt_eval_count") or 0 for r in v),
        "mean_output_tokens": statistics.mean(r.get("eval_count") or 0 for r in v)} for k,v in groups.items()]


if __name__ == "__main__":
    main()
