import json
import numpy as np
import pytest
from experiments.data import ROOT, examples, messages
from experiments.scoring import score_answer
from rag.retriever import chunk_markdown, TfidfRetriever, Chunk
from rag.embedding import EmbeddingRetriever
QA = json.loads((ROOT / 'eval/qa_pairs.json').read_text())


def test_stale_two_factor_false_positive_fixed():
    qa = next(q for q in QA if q['id'] == 'changed_2')
    text = '2FA is optional for standard employee accounts but mandatory for accounts with administrative privileges.'
    assert score_answer(qa, text, 'v1')
    assert not score_answer(qa, text, 'v2')


def test_numeric_substrings_and_contradictions():
    qa = next(q for q in QA if q['id'] == 'changed_5')
    assert not score_answer(qa, '24 hours', 'v2')
    assert not score_answer(qa, '2 hours or 4 hours', 'v2')
    assert not score_answer(qa, '2 days', 'v2')
    assert score_answer(qa, '2 hours', 'v2')


@pytest.mark.parametrize('version', ['v1', 'v2'])
def test_reference_answers_and_retrieval(version):
    retriever = TfidfRetriever(chunk_markdown(str(ROOT / f'data/handbook_{version}.md')))
    for qa in QA:
        assert score_answer(qa, qa[f'answer_{version}'], version)
        chunk, _ = retriever.retrieve(qa['question'])[0]
        assert chunk.section_id == qa['section']
        assert score_answer(qa, chunk.text, version)


def test_no_exact_eval_questions_in_adaptation_data():
    heldout = {q['question'].lower() for q in QA}
    assert not heldout.intersection(e['question'].lower() for e in examples('v1'))
    assert len(examples('v1')) == 13
    assert len(messages('test', 'fewshot', 'v1')) == 28


def test_embedding_ranking_and_interface():
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[0., 1.], [1., 0.]]) if len(texts) == 2 else np.array([[1., 0.]])
    chunks = [Chunk('a', 'a', 'a'), Chunk('b', 'b', 'b')]
    retriever = EmbeddingRetriever(chunks, encoder=Encoder())
    assert retriever.retrieve('query') == [(chunks[1], 1.)]
    with pytest.raises(ValueError):
        retriever.retrieve('query', 0)


def test_ollama_errors_do_not_become_answers(monkeypatch):
    from experiments.models import OllamaModel
    class Response:
        def raise_for_status(self):
            raise RuntimeError('server failed')
    monkeypatch.setattr('experiments.models.requests.get', lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match='server failed'):
        OllamaModel()


def test_summary_does_not_count_repeats_as_unique_questions():
    from experiments.run import summarize
    row = {'method':'tfidf', 'knowledge_version':'v1', 'truth_version':'v1',
           'category':'stable', 'correct':True, 'end_to_end_seconds':1., 'id':'stable_1'}
    result = summarize([row,row])[0]
    assert result['n'] == 2
    assert result['unique_questions'] == 1


def test_concise_yes_no_answers():
    qa = next(q for q in QA if q['id'] == 'changed_2')
    assert score_answer(qa, 'No.', 'v1')
    assert not score_answer(qa, 'No.', 'v2')
    assert score_answer(qa, 'Yes.', 'v2')
    assert not score_answer(qa, 'Yes.', 'v1')
