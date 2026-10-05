"""Colab CUDA QLoRA training. No evaluation questions are used for training."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from experiments.data import examples
from experiments.models import SYSTEM


class PolicyDataset(Dataset):
    def __init__(self, tokenizer, version):
        self.rows = []
        for example in examples(version):
            prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example['question']}]
            prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": example['answer']}], tokenize=True)
            if full[:len(prefix)] != prefix:
                raise ValueError('Chat template prefix mismatch; review completion masking')
            self.rows.append({"input_ids": full, "attention_mask": [1]*len(full),
                              "labels": [-100]*len(prefix) + full[len(prefix):]})
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return {k: torch.tensor(v) for k,v in self.rows[index].items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--version', choices=['v1', 'v2'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume-adapter', help='Continue adapting the v1 adapter on v2 content')
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('This training recipe requires a CUDA GPU. Select a GPU runtime in Colab.')
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    model = prepare_model_for_kbit_training(model)
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    model.config.use_cache = False
    dataset = PolicyDataset(tokenizer, args.version)
    trainer = Trainer(model=model, train_dataset=dataset,
        args=TrainingArguments(output_dir=args.output, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, num_train_epochs=args.epochs, learning_rate=2e-4,
            fp16=True, logging_steps=5, save_strategy='no', report_to='none', seed=42,
            gradient_checkpointing=True, optim='adamw_torch'))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    metadata = {**vars(args), 'seed':42, 'training_examples':len(dataset), 'train_seconds':elapsed,
        'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(), 'gpu':torch.cuda.get_device_name(),
        'metrics':result.metrics, 'note':'Fixed epoch budget; no tuning on held-out evaluation answers.'}
    Path(args.output, 'training_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
