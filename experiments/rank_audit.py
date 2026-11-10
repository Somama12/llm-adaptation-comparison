"""Rank relevant handbook sections under four lexical indexing choices."""
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rag.retriever import chunk_markdown
ROOT = Path(__file__).resolve().parents[1]


def run():
    rows, summary = [], []
    for version in ('v1', 'v2'):
        chunks = chunk_markdown(ROOT / f'data/handbook_{version}.md')
        for form, file in [('original', 'qa_pairs'), ('paraphrase', 'qa_paraphrases')]:
            questions = json.loads((ROOT / f'eval/{file}.json').read_text())
            for mode in ('word/body', 'word/title+body', 'char/body', 'char/title+body'):
                text = [(c.section_id + ' ' if 'title' in mode else '') + c.text for c in chunks]
                vec = (TfidfVectorizer(stop_words='english') if mode.startswith('word')
                       else TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)))
                matrix = vec.fit_transform(text)
                ranks = []
                for q in questions:
                    scores = (vec.transform([q['question']]) @ matrix.T).toarray()[0]
                    order = np.argsort(-scores, kind='stable')
                    ids = [chunks[i].section_id for i in order]
                    rank = ids.index(q['section']) + 1
                    ranks.append(rank)
                    rows.append(dict(version=version, form=form, mode=mode, question=q['question'],
                                     id=q['id'], target=q['section'], rank=rank, first=ids[0],
                                     top_score=float(scores[order[0]])))
                summary.append(dict(version=version, form=form, mode=mode,
                                    top1=ranks.count(1), top3=sum(r<=3 for r in ranks), n=len(ranks)))
    return dict(summary=summary, rows=rows)


if __name__ == '__main__':
    data = run()
    (ROOT/'results/rank_audit.json').write_text(json.dumps(data, indent=2)+'\n')
    for r in data['summary']:
        print(r['version'], r['form'], r['mode'], r['top1'], r['top3'], sep=' | ')
