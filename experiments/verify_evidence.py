"""Validate the published pilot against raw files without rerunning or rescoring it."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from experiments.data import ROOT
from experiments.update_audit import compare


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    def read(name):
        return json.loads((root / name).read_text())

    qa = {q['id']: q for q in read('eval/qa_pairs.json')}
    require(len(qa) == 11, 'Expected eleven evaluation facts')
    raw = read('results/colab_baselines.json')
    audited = read('results/colab_baselines_audited.json')
    audit = audited['metadata']['score_audit']
    require(digest(root / audit['source']) == audit['source_sha256'], 'Baseline audit source hash differs')
    expected_rows = copy.deepcopy(raw['rows'])
    for change in audit['overrides']:
        matches = [r for r in expected_rows if all(r[k] == change[k] for k in
                   ('method', 'knowledge_version', 'truth_version', 'id'))]
        require(len(matches) == 1, 'Audit override must identify exactly one row')
        matches[0]['correct'] = change['correct']
    require(expected_rows == audited['rows'], 'Baseline audit differs beyond documented overrides')

    files = ['results/colab_baselines.json', 'results/colab_baselines_audited.json',
             'results/week12/lora_v1.json', 'results/week12/lora_v2.json']
    for name in files:
        data = read(name)
        meta = data['metadata']
        require(meta['status'] == 'complete', f'Incomplete file: {name}')
        require(meta['qa_sha256'] == digest(root / 'eval/qa_pairs.json'), f'Question provenance: {name}')
        for version, sha in meta['handbook_sha256'].items():
            require(sha == digest(root / f'data/handbook_{version}.md'), f'Handbook provenance: {name}')
        require(meta['model']['model'] == 'Qwen/Qwen2.5-3B-Instruct', f'Model mismatch: {name}')
        groups = {}
        for row in data['rows']:
            q = qa[row['id']]
            require(row['question'] == q['question'] and row['expected'] == q['answer_' + row['truth_version']],
                    f'Question/reference mismatch: {name}')
            require(row['category'] == q['category'], f'Category mismatch: {name}')
            key = (row['method'], row['knowledge_version'], row['truth_version'], row['repeat'])
            groups.setdefault(key, []).append(row['id'])
        require(all(len(ids) == 11 and set(ids) == set(qa) for ids in groups.values()), f'Missing/duplicate facts: {name}')
        for summary in data['summary']:
            rows = [r for r in data['rows'] if all(r[k] == summary[k] for k in
                    ('method', 'knowledge_version', 'truth_version', 'category'))]
            require(summary['n'] == len(rows) and summary['correct'] == sum(r['correct'] for r in rows),
                    f'Summary mismatch: {name}')
            require(summary['accuracy'] == sum(r['correct'] for r in rows) / len(rows), f'Accuracy mismatch: {name}')

    folder = root / 'results/week12'
    paired = compare(folder)
    require(paired == read('results/week12/pairs.json'), 'Saved automatic pairs differ from raw outputs')
    pairs = read('results/week12/pairs_audited.json')
    require(len(pairs) == 11 and {r['id'] for r in pairs} == set(qa), 'Audited pairs missing/duplicated')
    by_id = {r['id']: r for r in paired}
    for row in pairs:
        require(all(row[k] == v for k, v in by_id[row['id']].items()), 'Audited pair changed raw evidence')
        for old, new in [('stale', 'audited_stale'), ('updated', 'audited_updated')]:
            require(type(row[new]) is bool, 'Audit judgment must be boolean')
            if row[old] != row[new]:
                require(bool(row.get('note')), 'Undocumented score override')
    table = []
    for method in ('tfidf', 'embedding', 'fewshot'):
        scores = [sum(r['correct'] for r in audited['rows'] if r['method'] == method and
                  (r['knowledge_version'], r['truth_version']) == c)
                  for c in [('v1', 'v1'), ('v1', 'v2'), ('v2', 'v2')]]
        table.append(dict(method=method, original=scores[0], stale=scores[1], updated=scores[2], n=11))
    initial = read('results/week12/lora_v1.json')['rows']
    table.append(dict(method='lora', original=sum(r['correct'] for r in initial if r['truth_version'] == 'v1'),
                      stale=sum(r['audited_stale'] for r in pairs), updated=sum(r['audited_updated'] for r in pairs), n=11))
    require([(r['original'], r['stale'], r['updated']) for r in table] ==
            [(10, 6, 11), (10, 6, 11), (10, 5, 10), (5, 4, 4)], 'Published comparison does not match evidence')
    from experiments.rank_audit import run
    fresh, saved = run(), read('results/rank_audit.json')
    require(fresh['summary'] == saved['summary'], 'Retrieval summary changed')
    require(len(saved['rows']) == 176, 'Expected 176 retrieval rankings')
    require([{k:v for k,v in r.items() if k != 'top_score'} for r in fresh['rows']] ==
            [{k:v for k,v in r.items() if k != 'top_score'} for r in saved['rows']], 'Retrieval ranks changed')
    provenance = read('results/week9_provenance.json')
    for name, sha in provenance['files'].items():
        if name.startswith('results/'):
            require(digest(root / name) == sha, f'Historical evidence altered: {name}')
    return dict(status='passed', facts=11, retrieval_records=176, comparison=table,
                note='Historical audit preserved. Fresh generation/training can differ; this check does not train a model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = verify()
    print(json.dumps(data, indent=2))
    if args.output:
        if args.output.exists():
            parser.error('Output already exists')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + '\n')
