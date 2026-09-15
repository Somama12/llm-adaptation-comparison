"""
RAG Staleness Evaluation Harness
==================================
Runs the held-out QA eval set against two versions of the retriever:
  - v1_retriever: indexed on handbook_v1.md (the "deployment-time" knowledge)
  - v2_retriever: indexed on handbook_v2.md (the "post-update" knowledge)

For each question, we check whether the retrieved chunk contains the
*correct* answer for that index version. The key experiment: for "changed"
questions, does re-indexing on v2 correctly surface the updated fact?
This simulates RAG's core claimed advantage -- swap the index, get fresh
answers, no retraining needed.

Usage: python eval/run_harness.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import chunk_markdown, TfidfRetriever


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


import re


def answer_in_chunk(answer: str, chunk_text: str) -> bool:
    """Loose containment check: does the numeric/key fact in `answer`
    appear in the retrieved chunk text?"""
    # Extract the core distinguishing token (number/unit) from the answer
    # e.g. "90 days" -> "90", "2 hours" -> "2", "150 GB" -> "150"
    nums = re.findall(r"\d+", answer)
    if nums:
        return all(n in chunk_text for n in nums)
    # Fallback for non-numeric answers (e.g. yes/no 2FA question)
    key_terms = ["mandatory", "optional"]
    hits = [t for t in key_terms if t in answer.lower() and t in chunk_text.lower()]
    return len(hits) > 0


def run_eval(handbook_path: str, qa_pairs: list[dict], answer_key: str) -> list[dict]:
    chunks = chunk_markdown(handbook_path)
    retriever = TfidfRetriever(chunks)
    results = []
    for qa in qa_pairs:
        retrieved = retriever.retrieve(qa["question"], top_k=1)
        chunk, score = retrieved[0]
        correct = answer_in_chunk(qa[answer_key], chunk.text)
        results.append({
            "id": qa["id"],
            "category": qa["category"],
            "question": qa["question"],
            "expected_section": qa["section"],
            "retrieved_section": chunk.title,
            "section_match": qa["section"] in chunk.title,
            "retrieval_score": round(score, 3),
            "expected_answer": qa[answer_key],
            "answer_found_in_chunk": correct,
        })
    return results


def summarize(results: list[dict], label: str):
    total = len(results)
    section_correct = sum(r["section_match"] for r in results)
    answer_correct = sum(r["answer_found_in_chunk"] for r in results)
    print(f"\n=== {label} ===")
    print(f"Section retrieval accuracy: {section_correct}/{total} ({100*section_correct/total:.0f}%)")
    print(f"Answer-present accuracy:    {answer_correct}/{total} ({100*answer_correct/total:.0f}%)")

    for cat in ["stable", "changed"]:
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(r["answer_found_in_chunk"] for r in cat_results)
        print(f"  [{cat}] {cat_correct}/{len(cat_results)} correct")


if __name__ == "__main__":
    with open("eval/qa_pairs.json") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs "
          f"({sum(q['category']=='stable' for q in qa_pairs)} stable, "
          f"{sum(q['category']=='changed' for q in qa_pairs)} changed)")

    # Experiment 1: v1-indexed RAG answering against v1 ground truth (baseline sanity check)
    results_v1_on_v1 = run_eval("data/handbook_v1.md", qa_pairs, answer_key="answer_v1")
    summarize(results_v1_on_v1, "v1-indexed RAG vs v1 ground truth (sanity check)")

    # Experiment 2: v1-indexed RAG answering against v2 ground truth (THIS IS THE STALENESS TEST)
    # If RAG is never re-indexed after the policy update, does it still give the OLD (wrong) answer?
    results_v1_on_v2 = run_eval("data/handbook_v1.md", qa_pairs, answer_key="answer_v2")
    summarize(results_v1_on_v2, "v1-indexed RAG vs v2 ground truth (staleness exposure)")

    # Experiment 3: v2-indexed RAG answering against v2 ground truth (RAG's claimed advantage:
    # just re-index, no retraining, and you're immediately correct on updated facts)
    results_v2_on_v2 = run_eval("data/handbook_v2.md", qa_pairs, answer_key="answer_v2")
    summarize(results_v2_on_v2, "v2-indexed RAG vs v2 ground truth (post re-index)")

    # Save all results for the report
    output = {
        "v1_indexed_vs_v1_truth": results_v1_on_v1,
        "v1_indexed_vs_v2_truth": results_v1_on_v2,
        "v2_indexed_vs_v2_truth": results_v2_on_v2,
    }
    with open("results/retrieval_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to results/retrieval_eval_results.json")
