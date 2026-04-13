"""
QLoRA fine-tuning for Llama 3.2 3B on UTD academic data.

Usage:
    python fine_tuning/train.py --config fine_tuning/config.py
    python fine_tuning/train.py --base_model meta-llama/Llama-3.2-3B-Instruct \\
                                 --data data/utd_instructions.jsonl \\
                                 --output ./fine_tuned_model

References:
    - QLoRA: https://arxiv.org/abs/2305.14314
    - PEFT: https://github.com/huggingface/peft
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    data_path: str = "data/utd_instructions.jsonl"
    output_dir: str = "./fine_tuned_model"
    # QLoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt template (Alpaca format)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

RESPONSE_TEMPLATE = "### Response:"


def format_prompt(sample: dict) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=sample.get("instruction", ""),
        input=sample.get("input", ""),
        output=sample.get("output", ""),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: TrainingConfig) -> None:
    set_seed(cfg.seed)

    # 1. BitsAndBytes quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.use_double_quant,
    )

    # 2. Load base model
    print(f"[1/5] Loading base model: {cfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 3. LoRA config
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Tokenizer
    print("[2/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. Dataset
    print(f"[3/5] Loading dataset: {cfg.data_path}")
    dataset = load_dataset("json", data_files=cfg.data_path, split="train")
    dataset = dataset.map(lambda x: {"text": format_prompt(x)})
    dataset = dataset.train_test_split(test_size=0.05, seed=cfg.seed)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        fp16=False,
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        seed=cfg.seed,
    )

    # 7. Train
    print("[4/5] Starting QLoRA fine-tuning")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 8. Save
    print(f"[5/5] Saving model to {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("✅ Fine-tuning complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QLoRA fine-tuning — UTD AI Chatbot")
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--data", default="data/utd_instructions.jsonl")
    p.add_argument("--output", default="./fine_tuned_model")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    cfg = TrainingConfig(
        base_model=args.base_model,
        data_path=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
    )
    train(cfg)
