"""Re-score saved answers without rerunning inference; preserve original correctness."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.data import ROOT
from experiments.run import summarize
from experiments.scoring import score_answer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qa', type=Path, default=ROOT/'eval/qa_pairs.json')
    args=p.parse_args()
    data=json.loads(args.input.read_text())
    qa={q['id']:q for q in json.loads(args.qa.read_text())}
    for r in data['rows']:
        r.setdefault('original_correct', r['correct'])
        r['correct']=score_answer(qa[r['id']],r['answer'],r['truth_version'])
    data['metadata']['rescore']={'source':str(args.input), 'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'scoring_sha256':hashlib.sha256((ROOT/'experiments/scoring.py').read_bytes()).hexdigest(),
        'reason':'Recognize concise yes/no answers as well as explicit policy wording; no inference was rerun.'}
    data['summary']=summarize(data['rows'])
    args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__ == '__main__':
    main()
