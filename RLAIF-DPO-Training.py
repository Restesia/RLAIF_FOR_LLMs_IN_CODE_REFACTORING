
"""
Entrenamiento Direct Preference Optimization (DPO) para LLaMA 2‑7B
-------------------------------------------------------------------
Requisitos previos (pip):
    pip install trl transformers accelerate peft bitsandbytes datasets

Estructura dataset (JSONL):
    {"prompt": "<texto>", "better": "<respuesta preferida>", "worse": "<respuesta rechazada>"}

Uso:
    python train_dpo_llama2.py --dataset RLAIF_DPO_dset.jsonl \
                               --base_model ./Modelos/llama2-7b \
                               --lora_path checkpoints/llama2-7b-mss4j \
                               --output_dir dpo-llama2-refactor 

El script:
    1. Carga el modelo base + LoRA (opcional).
    2. Prepara Dataset → formato DPO (prompt/chosen/rejected).
    3. Lanza `DPOTrainer` de TRL con PEFT (LoRA) para eficiencia.
    4. Guarda pesos adaptador LoRA y merges opcionalmente.
"""

import argparse
from pathlib import Path
import json
from typing import Dict, Any, List
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# --------------------------------------------------
# Argumentos CLI
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, help="Ruta JSONL con pares DPO")
    p.add_argument("--base_model", type=str, required=True, help="Modelo base LLaMA2 (hf path)")
    p.add_argument("--lora_path", type=str, default=None, help="Adapter LoRA previo (opcional)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.1, help="Coeficiente beta DPO")
    return p.parse_args()

# --------------------------------------------------
# Utils
# --------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open() as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                data.append({
                    "prompt": obj["prompt"],
                    "chosen": obj["better"],
                    "rejected": obj["worse"],
                })
    return data

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()
    print("▶️  Cargando tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("▶️  Cargando modelo base…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # PEFT LoRA config (si no tienes adapter previo, lo creamos nuevo)
    if args.lora_path:
        print("▶️  Cargando adapter LoRA previo…")
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=True)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    else:
        print("▶️ Es necesario añadir un LoRA valido")
        os._exit(1)

    # Dataset
    dataset_path = Path(args.dataset)
    data = load_jsonl(dataset_path)
    dset = Dataset.from_list(data)

    # TrainingArguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
        bf16=torch.cuda.is_available(),
        # Parámetros específicos de DPO
        beta=args.beta,
        max_prompt_length=args.max_len,
        max_length=args.max_len + 256,  # prompt + generación
    )

    print("▶️  Iniciando DPOTrainer…")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # None → crea copia congelada del modelo como referencia
        args=training_args,        # <- ahora es DPOConfig
        train_dataset=dset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Entrenamiento DPO completado. Pesos guardados en", args.output_dir)

if __name__ == "__main__":
    main()


