#!/usr/bin/env python3
"""
train_missionreason.py
======================
Fine-tunes Qwen2.5-1.5B-Instruct using QLoRA on the MissionReason v2 dataset.
Designed for memory-efficient training on single GPUs.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Target lightweight model
DATASET_PATH = "./missionreason_v2_dataset/missionreason_v2_chat.jsonl"
OUTPUT_DIR = "./missionreason_qwen_1.5b_qlora"

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 80/20 Train-Validation Split
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training Parameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting QLoRA Fine-Tuning...")
    trainer.train()

    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete!")

if __name__ == "__main__":
    main()