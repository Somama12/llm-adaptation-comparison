"""Interactive examples for the final demonstration (local inference or saved evaluation)."""
import argparse
import json
from textwrap import fill

from experiments.data import ROOT, messages
from experiments.models import OllamaModel
from rag.retriever import TfidfRetriever, chunk_markdown


def ask(question, url):
    model = OllamaModel(url=url)
    print('Local inference   |   llama3.2:3b   |   Ollama', flush=True)
    for version in ('v1', 'v2'):
        index = TfidfRetriever(chunk_markdown(ROOT / f'data/handbook_{version}.md'))
        chunk, score = index.retrieve(question)[0]
        print(f'\n{version.upper()}  {chunk.title}   [cosine {score:.3f}]', flush=True)
        print(fill(chunk.text.replace('**',''), 86), flush=True)
        output = model.generate(messages(question, 'tfidf', version, chunk.text))
        print('MODEL ANSWER:', output['answer'], flush=True)


def results():
    rows = json.loads((ROOT/'results/week12/pairs_audited.json').read_text())
    print('Qwen2.5-3B-Instruct   |   recorded T4 experiment   |   audited')
    print(f"{'Question':<14} {'Stale':<8} Updated")
    for r in rows:
        values = ['correct' if r[k] else 'miss' for k in ['audited_stale','audited_updated']]
        print(f"{r['id']:<14} {values[0]:<8} {values[1]}")
    for cat in ['changed','stable']:
        group = [r for r in rows if r['category']==cat]
        print(f"{cat}: {sum(r['audited_stale'] for r in group)}/{len(group)} -> "
              f"{sum(r['audited_updated'] for r in group)}/{len(group)}")


def compare():
    baselines = json.loads((ROOT/'results/colab_baselines_audited.json').read_text())['rows']
    pairs = json.loads((ROOT/'results/week12/pairs_audited.json').read_text())
    initial = json.loads((ROOT/'results/week12/lora_v1.json').read_text())['rows']
    print('Qwen2.5-3B-Instruct   |   audited core facts   |   11 questions')
    print('Method             Original     Stale     Updated')
    for method in ['tfidf', 'embedding', 'fewshot', 'lora']:
        if method == 'lora':
            scores = [sum(r['correct'] for r in initial if r['truth_version']=='v1'),
                      sum(r['audited_stale'] for r in pairs), sum(r['audited_updated'] for r in pairs)]
        else:
            scores = [sum(r['correct'] for r in baselines if r['method']==method and
                          (r['knowledge_version'],r['truth_version'])==condition)
                      for condition in [('v1','v1'),('v1','v2'),('v2','v2')]]
        print(f'{method:<17}'+'  '.join(f'{k:>5}/11' for k in scores))


def inspect_fact(qid):
    rows = json.loads((ROOT/'results/week12/pairs_audited.json').read_text())
    r = next((r for r in rows if r['id']==qid),None)
    if r is None:
        raise SystemExit('Unknown question ID. Run results to list available IDs.')
    for label,key in [('QUESTION','question'),('V2 POLICY','expected'),('UPDATED ANSWER','after'),
                      ('AUDIT NOTE','note')]:
        if key in r: print(f'{label}: '+fill(r[key],85))
    print('Automatic:',r['updated'],'  Audited:',r['audited_updated'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['ask','results','compare','inspect'])
    parser.add_argument('question', nargs='?',default='Within how many days of purchase can a customer request a full refund?')
    parser.add_argument('--id',default='changed_2')
    parser.add_argument('--url',default='http://127.0.0.1:11434')
    args = parser.parse_intermixed_args()
    if args.mode == 'ask':
        ask(args.question,args.url)
    elif args.mode=='compare':
        compare()
    elif args.mode=='inspect':
        inspect_fact(args.id)
    else:
        results()
