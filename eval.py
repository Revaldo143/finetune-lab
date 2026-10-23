"""Quick eval: base model vs fine-tuned adapter on 5 test prompts.

Usage (Colab, after train.py):
    python eval.py
Compares side-by-side so you can SEE what the fine-tune changed.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./bro-qwen-adapter"

TESTS = [
    "unakku peru enna",
    "handoff na enna",
    "look roulette deadline eppo",
    "brief ah sollu",
    "who is Revaldo",
]


def gen(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=120, do_sample=False)
    full = tok.decode(out[0], skip_special_tokens=True)
    return full.split("assistant")[-1].strip()


def main():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)  # T4 = sm75: no bf16 gemm
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    tuned = PeftModel.from_pretrained(base, ADAPTER)

    for t in TESTS:
        print(f"\n{'='*60}\nPROMPT: {t}")
        print(f"\n[BASE]\n{gen(base, tok, t)}")
        print(f"\n[TUNED]\n{gen(tuned, tok, t)}")


if __name__ == "__main__":
    main()
