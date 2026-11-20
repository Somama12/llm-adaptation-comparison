"""Export both adapters and their measured results with a verifiable SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def export_run(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise ValueError('Archive exists; choose a new filename')
    from safetensors import safe_open
    members = {}
    for version in ('v1', 'v2'):
        adapter = folder / 'adapters' / version
        for required in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json',
                         'tokenizer.json', 'training_metadata.json'):
            if not (adapter / required).is_file():
                raise ValueError(f'Missing adapter file: {version}/{required}')
        config = json.loads((adapter / 'adapter_config.json').read_text())
        if config.get('peft_type') != 'LORA':
            raise ValueError('Expected a LoRA adapter')
        with safe_open(adapter / 'adapter_model.safetensors', framework='pt', device='cpu') as weights:
            if not list(weights.keys()):
                raise ValueError('Empty adapter weights')
        for path in sorted(adapter.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt', '.jinja', '.model'):
                if path.is_symlink():
                    raise ValueError('Do not export symlinks')
                members[str(path.relative_to(folder))] = path.read_bytes()
        for name in (f'lora_{version}.json', f'training_{version}.json'):
            path = folder / name
            data = json.loads(path.read_text())
            if name.startswith('lora_') and data['metadata']['status'] != 'complete':
                raise ValueError('Cannot export incomplete evaluation')
            members[name] = path.read_bytes()
    for name in ('runtime.json', 'pairs.json', 'automatic_results.md', 'colab_baselines.json'):
        path = folder / name
        if path.is_file():
            members[name] = path.read_bytes()
    manifest = {'schema': 1, 'files': {name: {'sha256': sha(data), 'bytes': len(data)}
                                     for name, data in members.items()},
                'note': 'Adapter weights require the base model named in adapter_config.json. Scores are automatic unless explicitly audited.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2) + '\n')
    return verify_archive(output)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive entries')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Unsafe archive paths')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('schema') != 1 or set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Archive contents do not match manifest')
        for version in ('v1', 'v2'):
            for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json', 'tokenizer_config.json', 'training_metadata.json'):
                if f'adapters/{version}/{name}' not in names:
                    raise ValueError('Missing required adapter member')
            for name in (f'lora_{version}.json', f'training_{version}.json'):
                if name not in names:
                    raise ValueError('Missing evaluation/training evidence')
        for name, entry in manifest['files'].items():
            data = archive.read(name)
            if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
    return dict(status='verified', files=len(manifest['files']), archive_sha256=sha(Path(path).read_bytes()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_archive(args.verify), indent=2))
    elif args.run_dir and args.output:
        print(json.dumps(export_run(args.run_dir, args.output), indent=2))
    else:
        parser.error('Use --verify ZIP or --run-dir DIRECTORY --output ZIP')
