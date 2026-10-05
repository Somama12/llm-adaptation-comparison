import time
import requests

SYSTEM = "Answer the Nimbus Cloud Storage policy question concisely. Use the supplied policy information when available. If you do not know, say 'I do not know'. Give only the answer, without explaining your reasoning."


class OllamaModel:
    def __init__(self, model="llama3.2:3b", url="http://127.0.0.1:11434", seed=42):
        self.model, self.url, self.seed = model, url.rstrip("/"), seed
        response = requests.get(self.url + "/api/tags", timeout=10)
        response.raise_for_status()
        self.metadata = {"backend": "ollama", "model": model, "seed": seed,
                         "installed_models": response.json()["models"],
                         "temperature": 0, "num_predict": 96, "num_ctx": 4096}

    def generate(self, messages):
        start = time.perf_counter()
        response = requests.post(self.url + "/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "keep_alive": "30m", "options": {"temperature": 0, "seed": self.seed,
            "num_predict": 96, "num_ctx": 4096}}, timeout=300)
        response.raise_for_status()
        result = response.json()
        return {"answer": result["message"]["content"], "generation_seconds": time.perf_counter()-start,
                **{k: result.get(k) for k in ["prompt_eval_count", "eval_count", "load_duration",
                   "prompt_eval_duration", "eval_duration", "total_duration"]}}


class HFModel:
    """Same-base evaluation for Colab; optionally loads a trained PEFT adapter."""
    def __init__(self, model, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.metadata = {"backend": "huggingface", "model": model, "adapter": adapter,
                         "quantization": "nf4", "do_sample": False, "max_new_tokens": 96}

    def generate(self, messages):
        import torch
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_tensors="pt").to(self.model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(inputs, max_new_tokens=96, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        answer_ids = output[0, inputs.shape[1]:]
        return {"answer": self.tokenizer.decode(answer_ids, skip_special_tokens=True),
                "generation_seconds": time.perf_counter()-start,
                "prompt_eval_count": inputs.shape[1], "eval_count": len(answer_ids)}
