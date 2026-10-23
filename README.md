# Fine-tune Lab — "bro" (Tanglish assistant)

A real, honest fine-tuning experiment: teach Qwen2.5-7B-Instruct to talk like
bro — Tanglish, calls him sir, knows his projects.

## What this teaches (and doesn't)

- ✅ STYLE: Tanglish, "sir", brief direct answers
- ✅ FACTS: his projects, preferences, deadlines
- ❌ NOT intelligence — the tuned model won't be smarter than base Qwen.
  That's the honest limit of fine-tuning.

## Files

- `data/part*.py` — 249 synthetic examples (I authored them; the "AI trains AI" part)
- `build_dataset.py` — combines + shuffles + 95/5 train/val split → `dataset_*.jsonl`
- `train.py` — QLoRA (4-bit + LoRA r=16), ~20-40 min on Colab free T4
- `eval.py` — base vs tuned side-by-side on 5 prompts
- `finetune_bro_qwen.ipynb` — the Colab notebook (upload to colab.research.google.com)

## Run it (Colab, free)

1. Colab → Runtime → Change runtime type → **T4 GPU**
2. Accept the license: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
3. Upload the notebook + the two `dataset_*.jsonl` files
4. Run cells top to bottom → adapter lands in `./bro-qwen-adapter/`

## Build the dataset locally

```
python build_dataset.py
```

## Honest notes

- Colab free sessions can time out — keep the tab open during training.
- 249 examples is small; style transfer works, deep knowledge doesn't.
- The adapter (~100MB) can later plug into Ollama for local chat.
