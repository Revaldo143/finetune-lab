"""Merge the bro LoRA adapter into Qwen2.5-7B-Instruct and save the full merged model.

Usage:
    python merge_lora.py --adapter ./bro-qwen-adapter --out ./bro-qwen-merged

Downloads Qwen/Qwen2.5-7B-Instruct from Hugging Face on first run (~15GB).
Needs: pip install torch transformers peft huggingface_hub   (CPU torch is fine)
Colab notes (verified 2026-11-06): device_map="auto" splits the model across
GPU+CPU (plain "cpu" dies at 68% on free Colab's ~12.7GB RAM); offload_dir
lets PEFT dispatch disk-offloaded layers. If PEFT crashes on torchao
("only versions above 0.16.0 are supported"), run: pip uninstall -y torchao
Run: python merge_lora.py
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./bro-qwen-adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="./bro-qwen-merged")
    args = ap.parse_args()

    print("Loading base model (fp16, GPU+CPU) — takes a few minutes...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print("Applying adapter...")
    model = PeftModel.from_pretrained(base, args.adapter, offload_dir="./offload")
    print("Merging LoRA into base (a few minutes)...")
    merged = model.merge_and_unload()
    print(f"Saving merged model to {args.out} ...")
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Done — merged model saved.")


if __name__ == "__main__":
    main()
