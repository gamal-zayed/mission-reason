#!/usr/bin/env python3
"""
generate_predictions.py
=========================

Loads the fine-tuned MissionReason QLoRA adapter, runs it on the held-out
eval split (same 80/20, seed=42 split train_missionreason.py used, so this
is genuinely held-out data the model didn't train on), and writes
predictions in the exact JSONL format evaluate_missionreason.py --mode
compare expects: {"scenario_id", "predicted_action", "predicted_directive"}.

Usage:
    python generate_predictions.py \
        --adapter ./missionreason_qwen_1.5b_qlora \
        --raw-json ./missionreason_v3_dataset/missionreason_v3_raw.json \
        --chat-jsonl ./missionreason_v3_dataset/missionreason_v3_chat.jsonl \
        --out ./predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

VALID_ACTIONS = {"Observe", "Downlink", "Delay", "Skip", "Emergency"}


def parse_model_output(text: str):
    """Extract the final action and recovery directive from the model's
    generated text. Expects lines like 'Final Action: X' and optionally
    'Recovery Directive: Y', matching the format the training data used."""
    action_match = re.search(r"Final Action:\s*(\w+)", text)
    directive_match = re.search(r"Recovery Directive:\s*([A-Za-z_]+)", text)

    action = action_match.group(1).strip() if action_match else None
    if action not in VALID_ACTIONS:
        # Fallback: scan for any valid action word appearing in the text,
        # in case the model didn't follow the exact "Final Action:" format.
        for a in VALID_ACTIONS:
            if a in text:
                action = a
                break

    directive = directive_match.group(1).strip() if directive_match else None
    return action, directive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True,
                         help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--raw-json", type=str, required=True,
                         help="The raw dataset (for scenario_id lookup).")
    parser.add_argument("--chat-jsonl", type=str, required=True,
                         help="The chat-format dataset used for training "
                              "(needed to reproduce the identical eval split).")
    parser.add_argument("--out", type=str, default="./predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )

    print(f"Loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    # Reproduce the EXACT same train/test split train_missionreason.py used,
    # so we evaluate only on held-out examples the model never trained on.
    print(f"Loading dataset from {args.chat_jsonl}...")
    dataset = load_dataset("json", data_files=args.chat_jsonl, split="train")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    eval_dataset = dataset["test"]
    print(f"Eval samples (held-out): {len(eval_dataset)}")

    predictions = []
    with torch.no_grad():
        for i, example in enumerate(eval_dataset):
            messages = example["messages"][:2]  # system + user only, not the ground-truth assistant turn
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_text = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            action, directive = parse_model_output(generated_text)

            predictions.append({
                "scenario_id": example["scenario_id"],
                "predicted_action": action,
                "predicted_directive": directive,
            })

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(eval_dataset)} done")

    with open(args.out, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    n_unparsed = sum(1 for p in predictions if p["predicted_action"] is None)
    print(f"\nWrote {len(predictions)} predictions to {args.out}")
    if n_unparsed:
        print(f"WARNING: {n_unparsed} predictions had no parseable action -- "
              f"inspect these manually, they'll count as wrong in evaluation "
              f"rather than being silently dropped.")


if __name__ == "__main__":
    main()
