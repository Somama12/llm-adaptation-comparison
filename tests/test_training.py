"""CPU smoke test of adapter gradients; not evidence of trained task accuracy."""
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('peft')
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from finetune.train import PolicyDataset


def test_completion_mask_and_lora_backward():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
            prefix = [1, 2, 3]
            return prefix if add_generation_prompt else prefix + [4, 5, 6]
    dataset = PolicyDataset(Tokenizer(), 'v1')
    assert len(dataset) == 13
    row = dataset[0]
    assert row['labels'].tolist() == [-100, -100, -100, 4, 5, 6]
    model = get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)),
        LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    loss = model(**{k:v.unsqueeze(0) for k,v in row.items()}).loss
    assert torch.isfinite(loss)
    loss.backward()
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n for n,p in trainable)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in trainable)
