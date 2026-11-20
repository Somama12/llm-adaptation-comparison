"""Fresh GPU reproduction in a new directory; export both adapters before runtime expiry."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from experiments.data import ROOT
from experiments.export_run import export_run
from experiments.update_audit import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--skip-baselines', action='store_true', help='Rebuild/reload adapters only; no new baseline claims')
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    if folder.exists():
        parser.error('Run directory already exists; choose a new name')
    import torch
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required (Colab T4 recommended)')
    folder.mkdir(parents=True)

    def run(module, *argv):
        command = [sys.executable, '-m', module, *map(str, argv)]
        print('RUN:', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    result_files = []
    if not args.skip_baselines:
        baseline = folder / 'colab_baselines.json'
        run('experiments.run', '--backend', 'hf', '--model', args.model,
            '--methods', 'tfidf', 'embedding', 'fewshot', 'zero', '--output', baseline)
        result_files.append(baseline)
    for version in ('v1', 'v2'):
        adapter = folder / 'adapters' / version
        resume = ['--resume-adapter', folder / 'adapters/v1'] if version == 'v2' else []
        run('finetune.train', '--model', args.model, '--version', version, '--output', adapter, *resume)
        result = folder / f'lora_{version}.json'
        # Separate process reloads the saved adapter: this is also the deployment check.
        run('experiments.run', '--backend', 'hf', '--model', args.model, '--methods', 'lora',
            '--adapter', adapter, '--knowledge-versions', version, '--output', result)
        shutil.copyfile(adapter / 'training_metadata.json', folder / f'training_{version}.json')
        result_files.append(result)
    (folder / 'pairs.json').write_text(json.dumps(compare(folder), indent=2) + '\n')
    run('experiments.report', '--inputs', *result_files, '--output', folder / 'automatic_results.md')
    metadata = {'versions': {p: importlib.metadata.version(p) for p in
                ['torch', 'peft', 'transformers', 'bitsandbytes', 'accelerate', 'sentence-transformers', 'numpy', 'scikit-learn']},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'weights_sha256': {v: hashlib.sha256((folder / 'adapters' / v / 'adapter_model.safetensors').read_bytes()).hexdigest()
                                   for v in ('v1', 'v2')},
                'note': 'New reproduction; not the historical Week 12 weights or audited scores.'}
    (folder / 'runtime.json').write_text(json.dumps(metadata, indent=2) + '\n')
    archive = folder.with_suffix('.zip')
    print(json.dumps(export_run(folder, archive), indent=2), flush=True)
    print('DOWNLOAD AND KEEP:', archive, flush=True)


if __name__ == '__main__':
    main()
