"""Merge a LoRA adapter into Qwen2.5-7B-Instruct — by hand, one shard at a time.

No transformers, no accelerate, no PEFT. Only torch + safetensors +
huggingface_hub. That is the whole point: the normal merge script loads the
full 15GB model into RAM and dies on free Colab (RAM ceiling, PEFT dispatch
errors, silent deaths in the merge phase). This one loads ONE shard (~2GB),
merges it, saves it, frees it, and moves on. Peak RAM is ~3-4GB.

Usage (Colab):
    !python merge_manual.py --adapter ./bro-qwen-adapter --out ./bro-qwen-merged
    # --base defaults to Qwen/Qwen2.5-7B-Instruct on the Hugging Face hub.
    # The base is downloaded shard by shard (cached, so re-runs are fast).

How it works:
    1. Read r / lora_alpha / target_modules from the adapter's
       adapter_config.json (falls back to train.py's values if missing).
    2. Read the base model's index (model.safetensors.index.json).
    3. For each shard, in order: load it, and for every target weight that
       has a LoRA pair, do  W_new = W + (B @ A) * (alpha / r)  in float32,
       then cast back to the weight's original dtype (fp16).
    4. Save the shard under the SAME filename in --out, and copy the index,
       config, and tokenizer files, so --out is a normal loadable HF model.

Tied weights: Qwen2.5-7B ties lm_head to embed_tokens and the adapter has no
embedding targets, so lm_head / embed_tokens are copied as-is, never merged.
"""

import argparse
import gc
import json
import os
import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

# Fallback LoRA settings (from train.py). Used only if adapter_config.json
# is missing or does not list them — the adapter's own file always wins.
FALLBACK_R = 16
FALLBACK_ALPHA = 32
FALLBACK_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

# Files copied from the base model so the output is a loadable HF model.
COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
]

# Weights that are tied in Qwen2.5-7B — never merged, only copied as-is.
TIED_KEYS = {"lm_head.weight", "model.embed_tokens.weight"}


def load_adapter_config(adapter_dir):
    """Read r / alpha / targets from adapter_config.json (with fallbacks)."""
    path = os.path.join(adapter_dir, "adapter_config.json")
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    r = int(cfg.get("r", FALLBACK_R))
    alpha = float(cfg.get("lora_alpha", FALLBACK_ALPHA))
    targets = cfg.get("target_modules", FALLBACK_TARGETS)
    if isinstance(targets, str):
        targets = [targets]
    # rsLoRA uses alpha / sqrt(r) instead of alpha / r.
    if cfg.get("use_rslora", False):
        scaling = alpha / (r ** 0.5)
    else:
        scaling = alpha / r
    return r, alpha, list(targets), scaling


def load_adapter_weights(adapter_dir):
    """Return ({base_key: (A, B)}, {base_key: full_weight}) from the adapter.

    PEFT saves LoRA weights as e.g.
        base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
    We strip the wrapper prefix and the .lora_A/.lora_B suffix to recover the
    base weight name: model.layers.0.self_attn.q_proj.weight
    Keys that are full weights (modules_to_save) are kept as overrides.
    """
    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(sf_path):
        state = load_file(sf_path)
    else:
        bin_path = os.path.join(adapter_dir, "adapter_model.bin")
        state = torch.load(bin_path, map_location="cpu", weights_only=True)

    pairs = {}
    overrides = {}
    for key, tensor in state.items():
        core = key
        if core.startswith("base_model.model."):
            core = core[len("base_model.model."):]
        if core.endswith(".lora_A.default.weight"):
            base_key = core[:-len(".lora_A.default.weight")] + ".weight"
            pairs.setdefault(base_key, {})["A"] = tensor
        elif core.endswith(".lora_A.weight"):
            base_key = core[:-len(".lora_A.weight")] + ".weight"
            pairs.setdefault(base_key, {})["A"] = tensor
        elif core.endswith(".lora_B.default.weight"):
            base_key = core[:-len(".lora_B.default.weight")] + ".weight"
            pairs.setdefault(base_key, {})["B"] = tensor
        elif core.endswith(".lora_B.weight"):
            base_key = core[:-len(".lora_B.weight")] + ".weight"
            pairs.setdefault(base_key, {})["B"] = tensor
        elif core.endswith(".weight"):
            overrides[core] = tensor

    lora = {}
    for base_key, ab in pairs.items():
        if "A" in ab and "B" in ab:  # keep only complete pairs
            lora[base_key] = (ab["A"], ab["B"])
    return lora, overrides


def load_base_index(base):
    """Return the base model's index dict, or None for a single-file model."""
    if os.path.isdir(base):
        path = os.path.join(base, "model.safetensors.index.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    try:
        path = hf_hub_download(repo_id=base,
                               filename="model.safetensors.index.json")
    except Exception:
        return None
    with open(path) as f:
        return json.load(f)


def shard_names(index):
    """Ordered shard filenames (sorted: Qwen shards are zero-padded)."""
    if index is None:
        return ["model.safetensors"]
    return sorted(set(index["weight_map"].values()))


def get_file(base, is_local, filename):
    """Local path for a base file, downloading it from the hub if needed."""
    if is_local:
        return os.path.join(base, filename)
    return hf_hub_download(repo_id=base, filename=filename)


def copy_helper_files(base, is_local, out_dir, index):
    """Copy index + config + tokenizer files so --out loads as an HF model."""
    names = list(COPY_FILES)
    if index is not None:
        names.append("model.safetensors.index.json")
    for name in names:
        try:
            src = get_file(base, is_local, name)
            if is_local and not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(out_dir, name))
        except Exception as e:
            print(f"  (skip {name}: {e})")


def main():
    ap = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a base model, one shard at a time.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF hub id or local dir of the base model")
    ap.add_argument("--adapter", default="./bro-qwen-adapter",
                    help="dir with adapter_model.safetensors + adapter_config.json")
    ap.add_argument("--out", default="./bro-qwen-merged",
                    help="dir to write the merged model to")
    args = ap.parse_args()

    r, alpha, targets, scaling = load_adapter_config(args.adapter)
    print(f"LoRA: r={r}, alpha={alpha}, scaling={scaling:.4f}, targets={targets}")
    lora, overrides = load_adapter_weights(args.adapter)
    print(f"Adapter: {len(lora)} LoRA pairs, {len(overrides)} full-weight overrides")

    is_local = os.path.isdir(args.base)
    index = load_base_index(args.base)
    shards = shard_names(index)
    print(f"Base: {args.base} ({len(shards)} shard(s))")
    os.makedirs(args.out, exist_ok=True)

    total_merged = 0
    total_copied = 0
    total_skipped = 0
    for i, shard in enumerate(shards, 1):
        path = get_file(args.base, is_local, shard)
        tensors = load_file(path)
        out_tensors = {}
        seen_storage = set()
        merged_here = 0
        copied_here = 0
        for key, w in tensors.items():
            if key in lora and key not in TIED_KEYS:
                A, B = lora[key]
                # W + (B @ A) * scaling, in float32, back to the original dtype.
                delta = (B.float() @ A.float()) * scaling
                new_w = (w.float() + delta).to(w.dtype)
                merged_here += 1
            elif key in overrides:
                new_w = overrides[key].to(w.dtype)
                merged_here += 1
            else:
                if key.endswith(".weight") and key.split(".")[-2] in targets:
                    total_skipped += 1  # a target weight with no adapter entry
                new_w = w
                copied_here += 1
            # safetensors refuses tensors that share storage; clone if needed.
            storage_ptr = new_w.untyped_storage().data_ptr()
            if storage_ptr in seen_storage:
                new_w = new_w.clone()
                storage_ptr = new_w.untyped_storage().data_ptr()
            seen_storage.add(storage_ptr)
            out_tensors[key] = new_w.contiguous()

        save_file(out_tensors, os.path.join(args.out, shard),
                  metadata={"format": "pt"})
        total_merged += merged_here
        total_copied += copied_here
        print(f"Shard {i}/{len(shards)}: {shard} — "
              f"{merged_here} merged, {copied_here} copied")
        del tensors, out_tensors
        gc.collect()

    copy_helper_files(args.base, is_local, args.out, index)
    if total_skipped:
        print(f"Note: {total_skipped} target weight(s) had no adapter entry "
              f"(copied as-is).")
    print(f"Done — merged model saved to {args.out} "
          f"({len(shards)} shard(s), {total_merged} weights merged, "
          f"{total_copied} copied)")


if __name__ == "__main__":
    main()
