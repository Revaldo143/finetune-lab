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
    "india oda national animal enna",
    "coffee vs tea, unakku ethu pudikkum",
    "naan next month la 1 lakh earn panniduvena",
    "github la pudhu repo create panna step by step sollu",
    "BROK coin contract address enna",
]
# Old TESTS (unakku peru enna / handoff na enna / look roulette deadline
# eppo / brief ah sollu / who is Revaldo) were all IN the training data,
# so they were memorised and proved nothing. The 5 fresh prompts above
# are unseen — BROK is the key v3 re-test: it must REFUSE, never invent
# an address (v2 TUNED hallucinated 0x5FbD... — never use that address).


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
        trust_remote_code=True,
        dtype=torch.float16)  # NOT torch_dtype (deprecated/ignored) — T4 has no bf16
    tuned = PeftModel.from_pretrained(base, ADAPTER)

    for t in TESTS:
        print(f"\n{'='*60}\nPROMPT: {t}")
        # FIX (2026-11-05, verified in Colab): PeftModel.from_pretrained
        # modifies `base` in place, so gen(base) above would really be
        # TUNED — v1 + v2 evals compared tuned vs tuned and proved
        # nothing. disable_adapter() runs the same model as true BASE.
        with tuned.disable_adapter():
            print(f"\n[BASE]\n{gen(tuned, tok, t)}")
        print(f"\n[TUNED]\n{gen(tuned, tok, t)}")


if __name__ == "__main__":
    main()
