"""fine_tune_llama2_mss4j.py

Fine‑tune LLaMA‑2 7B on the pre‑processed ManySStuBs4J dataset using
QLoRA (4‑bit) + LoRA adaptation.

Dataset layout expected (output of prepare_mss4j.py):
    dataset/
      ├─ train.jsonl
      ├─ val.jsonl
      └─ test.jsonl

Each JSONL line should contain a single key "text" with the entire
sequence <s>[INST] ... </s>.

Quick start (single‑GPU RTX 4090):
    python fine_tune_llama2_mss4j.py \
        --model_dir models/llama2-7b-hf \
        --data_dir dataset \
        --output_dir checkpoints/llama2-7b-mss4j \
        --epochs 1

Main dependencies:
    pip install "transformers>=5.0" "peft>=0.10" bitsandbytes accelerate datasets tqdm
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import bitsandbytes as bnb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fine‑tune LLaMA‑2 7B con QLoRA en ManySStuBs4J")
    p.add_argument("--model_dir", required=True, help="Ruta de los pesos HF de LLaMA‑2 7B (base)")
    p.add_argument("--data_dir", required=True, help="Directorio con train/val/test.jsonl")
    p.add_argument("--output_dir", required=True, help="Carpeta de salida checkpoints")
    p.add_argument("--epochs", type=int, default=1, help="Épocas de entrenamiento (def. 1)")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate LoRA (def. 2e‑4)")
    p.add_argument("--batch_size", type=int, default=2, help="Micro‑batch por GPU (def. 2)")
    p.add_argument("--max_len", type=int, default=1024, help="Longitud máx. tokens (def. 1024)")
    p.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    p.add_argument("--r", type=int, default=64, help="Rank LoRA (def. 64)")
    p.add_argument("--alpha", type=int, default=16, help="Alpha LoRA (def. 16)")
    p.add_argument("--fp16", action="store_true", help="Usar FP16 (en vez de BF16) para LoRA")
    return p.parse_args()

def get_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    tokenizer.pad_token_id = 0  # llama no tiene PAD; se usa <unk> (id 0)
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(ds, tokenizer, max_len):
    def _tok(ex):

        prompt          = f"<BUGGY>\n{ex['buggy']}\n\n<FIXED>\n"
        completion      = ex["fixed"]
        full_example    = prompt + completion

        enc_full  = tokenizer(
            full_example,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

        enc_labels      = [-100] * len(enc_full["input_ids"])
        prompt_len      = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        enc_labels[prompt_len:] = enc_full["input_ids"][prompt_len:]

        enc_full["labels"] = enc_labels
        return enc_full

    return ds.map(_tok, batched=False, remove_columns=ds.column_names)

def load_model(model_dir: str, lora_r: int, lora_alpha: int):
    """Carga el modelo base en 4‑bit NF4.

    * Intenta primero usar los shards **safetensors** (más rápidos).
    * Si éstos no existen, vuelve a intentarlo ignorando safetensors para
      que Transformers cargue los `.bin` sin que el usuario tenga que
      borrar ficheros ni añadir nuevas dependencias.
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    def _load(use_safetensors: bool | None):
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=use_safetensors,
        )

    try:
        model = _load(use_safetensors=True)  # primero safetensors
    except (FileNotFoundError, ValueError):
        # Fallback a shards .bin si los .safetensors no existen.
        print("Shards .safetensors no encontrados, cargando .bin …")
        model = _load(use_safetensors=False)

    # LoRA en capas de atención y proyecciones FFN
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model

def main():
    args = parse_args()

    tokenizer = get_tokenizer(args.model_dir)


    ds_train = load_dataset("json", data_files=str(Path(args.data_dir) / "train.jsonl"))["train"]
    ds_val = load_dataset("json", data_files=str(Path(args.data_dir) / "val.jsonl"))["train"]

    ds_train = tokenize_dataset(ds_train, tokenizer, args.max_len)
    ds_val = tokenize_dataset(ds_val, tokenizer, args.max_len)


    model = load_model(args.model_dir, args.r, args.alpha)
    #model.gradient_checkpointing_enable()
    #model.config.use_cache = False


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Parámetros de entrenamiento
    t_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        save_total_limit=3,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=args.fp16,
        bf16=not args.fp16,
        optim="paged_adamw_8bit",
        report_to="none",
        do_eval=True,
        save_steps=5000)

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator)

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine‑tuning terminado. Adaptador guardado en: ", args.output_dir)


if __name__ == "__main__":
    main()
