"""Create an evidence table only from completed experiment output files."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    lines=['# Experiment results', '', 'Accuracy uses conservative task-specific scoring; inspect raw answers for ambiguous cases.', '',
           '| File | Method | Knowledge → truth | Group | Correct / n | Mean seconds |',
           '|---|---|---|---|---:|---:|']
    for path in args.inputs:
        data=json.loads(path.read_text())
        if data['metadata'].get('status') != 'complete':
            raise ValueError(f'{path} is not marked complete; do not report partial runs')
        for row in data['summary']:
            lines.append(f"| {path.name} | {row['method']} | {row['knowledge_version']} → {row['truth_version']} | {row['category']} | {row['correct']}/{row['n']} | {row['mean_seconds']:.3f} |")
    lines += ['', 'Repeated timings are not independent test facts. With only 11 questions (5 stable, 6 changed), these results are a pilot.',
              'Wall time and token counts are compute proxies. Dollar cost and energy were not measured.',
              'Compare timings only on the same hardware/backend. Retrieval-only metrics are not generated-answer accuracy.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
