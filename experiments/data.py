"""Adaptation examples are authored independently of evaluation questions."""
import json
from pathlib import Path
from rag.retriever import chunk_markdown
ROOT = Path(__file__).resolve().parents[1]


def examples(version):
    # Section-to-question mapping; never reads evaluation questions or answers.
    questions = [
        "State the password rotation interval for Nimbus staff.",
        "Describe the authentication requirement for an ordinary Nimbus employee.",
        "What is the maximum duration of a remote VPN login?",
        "State Nimbus's deadline for requesting a purchase refund.",
        "When should an authorized reimbursement be processed?",
        "For what period are deleted customers' backup copies kept?",
        "State the paid-account backup schedule.",
        "What response target applies to standard support requests?",
        "State the response target for priority support requests.",
        "When does an unanswered support request move to the second tier?",
        "Describe the entry-level storage subscription.",
        "Describe the professional storage subscription.",
        "Describe the shared team storage subscription."]
    chunks = chunk_markdown(str(ROOT / "data" / f"handbook_{version}.md"))
    if len(chunks) != len(questions):
        raise ValueError("Handbook sections changed; review adaptation question mapping")
    return [{"question": q, "answer": c.text, "section": c.section_id} for q,c in zip(questions,chunks)]


def messages(question, method, version, context=None):
    from experiments.models import SYSTEM
    out = [{"role": "system", "content": SYSTEM}]
    if method == "fewshot":
        for item in examples(version):
            out += [{"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}]
    prompt = question if context is None else f"Policy context:\n{context}\n\nQuestion: {question}"
    return out + [{"role": "user", "content": prompt}]
