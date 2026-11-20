"""Acceptance regressions: preserve evidence, reject corrupt bundles, reload real adapters."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from experiments.data import ROOT
from experiments.export_run import export_run, verify_archive
from experiments.scoring import score_answer
from experiments.verify_evidence import verify


def test_published_evidence_is_consistent():
    assert verify()['status'] == 'passed'


def test_voluntary_now_required_later_is_not_current_mandate():
    qa = next(q for q in json.loads((ROOT / 'eval/qa_pairs.json').read_text()) if q['id'] == 'changed_2')
    assert not score_answer(qa, '2FA is voluntary for standard accounts but will be required next year.', 'v2')


def test_runner_does_not_overwrite_evidence(tmp_path):
    target = tmp_path / 'result.json'
    target.write_text('preserve me')
    result = subprocess.run([sys.executable, '-m', 'experiments.run', '--retrieval-only',
                             '--methods', 'tfidf', '--output', str(target)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'already exists' in result.stderr
    assert target.read_text() == 'preserve me'


def test_export_round_trip_reloads_real_peft_weights(tmp_path):
    # Tiny random CPU model tests serialization/deployment, NOT Qwen task accuracy or CUDA.
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from peft import LoraConfig, get_peft_model, PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    config = LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(42)
    base = LlamaForCausalLM(config)
    base_state = {k:v.clone() for k,v in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    token_ids = torch.tensor([[1,2,3]])
    loss = model(input_ids=token_ids, labels=token_ids).loss
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad, alpha=-0.01)
    model.eval()
    expected = model(input_ids=token_ids).logits.detach()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'a':1}, unk_token='[UNK]')), unk_token='[UNK]')
    folder = tmp_path / 'run'
    for version in ('v1', 'v2'):
        dest = folder / 'adapters' / version
        model.save_pretrained(dest)
        tokenizer.save_pretrained(dest)
        (dest / 'training_metadata.json').write_text(json.dumps({'test_only': True}))
        (folder / f'training_{version}.json').write_text(json.dumps({'test_only': True}))
        (folder / f'lora_{version}.json').write_text(json.dumps({'metadata': {'status': 'complete', 'test_only': True}}))
    archive = tmp_path / 'run.zip'
    assert export_run(folder, archive)['status'] == 'verified'
    restored = tmp_path / 'restored'
    with zipfile.ZipFile(archive) as z:
        z.extractall(restored)  # This archive was created here and verified above.
    fresh = LlamaForCausalLM(config)
    fresh.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(fresh, restored / 'adapters/v2').eval()
    torch.testing.assert_close(loaded(input_ids=token_ids).logits, expected)
    damaged = tmp_path / 'damaged.zip'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(damaged, 'w') as dst:
        for name in src.namelist():
            dst.writestr(name, b'corrupt' if name == 'adapters/v2/adapter_model.safetensors' else src.read(name))
    with pytest.raises(ValueError, match='Checksum'):
        verify_archive(damaged)
    with pytest.raises(ValueError, match='exists'):
        export_run(folder, archive)
