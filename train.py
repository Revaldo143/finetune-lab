"""QLoRA fine-tune of Qwen2.5-7B-Instruct on the bro dataset.

Run on Colab with a free T4 GPU:
    1. Runtime -> Change runtime type -> T4 GPU
    2. Upload dataset_train.jsonl / dataset_val.jsonl (or git clone this repo)
    3. pip install -r requirements-colab.txt
    4. huggingface-cli login   (accept https://huggingface.co/Qwen/Qwen2.5-7B-Instruct first)
    5. python train.py

Time: ~20-40 min for 3 epochs on ~237 examples. Output: ./bro-qwen-adapter/
What it learns: STYLE (Tanglish, calls him sir, brief) + FACTS (his projects).
It will NOT become smarter than the base model — honest expectations.
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT = "./bro-qwen-adapter"


def formatting_func(example):
    # Alpaca -> simple chat text; Qwen chat template applied by tokenizer later
    return (
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 = sm75: no bf16 gemm support
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,  # NOT torch_dtype (deprecated/ignored) — non-quantized layers (lm_head etc) must be fp16 on T4
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # Safety: peft builds LoRA adapters in the base model's orig dtype (bf16);
    # T4 (sm75) has no bf16 kernels, so cast every non-quantized bf16 param to fp16.
    # Params4bit (quantized weights) is skipped via its quant_state attribute.
    n_cast = 0
    for p in model.parameters():
        if p.dtype == torch.bfloat16 and not hasattr(p, "quant_state"):
            p.data = p.data.to(torch.float16)
            n_cast += 1
    print(f"LoRA-wrapped; cast {n_cast} bf16 params to fp16")
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files="dataset_train.jsonl")["train"]
    val_ds = load_dataset("json", data_files="dataset_val.jsonl")["train"]

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,   # T4 = sm75: bf16 gemm unsupported, use fp16
        bf16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,  # already LoRA-wrapped above; do NOT pass peft_config again
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        args=args,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(OUT)
    print(f"Done — adapter saved to {OUT}")


if __name__ == "__main__":
    main()
