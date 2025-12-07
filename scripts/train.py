import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


def latest_checkpoint(output_dir: Path) -> str | None:
    """Return the latest checkpoint path in output_dir, or None if none exist."""
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
            candidates.append((step, path))
        except ValueError:
            continue
    if not candidates:
        return None
    _, ckpt_path = max(candidates, key=lambda x: x[0])
    return str(ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal LM on local data.")
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-2-13b-hf",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--data-file",
        default="data/raw/wikipedia-en-0.json",
        help="Path to a JSON array of strings for training.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints",
        help="Where to store training checkpoints.",
    )
    parser.add_argument(
        "--save-final-to",
        default="models/latest",
        help="Directory to save final model/tokenizer.",
    )
    parser.add_argument(
        "--adapter-dir",
        default="models/adapters",
        help="Directory to save LoRA adapters (when enabled).",
    )
    parser.add_argument(
        "--adapter-name",
        default="lora_adapter",
        help="Adapter name (subdirectory under adapter-dir when using LoRA).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/training",
        help="Directory for trainer logs.",
    )
    parser.add_argument(
        "--log-to",
        default="none",
        choices=["none", "tensorboard", "wandb"],
        help="Where to report metrics.",
    )
    parser.add_argument(
        "--hf-push",
        action="store_true",
        help="Push final model/adapter to Hugging Face Hub (requires token configured).",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="If set, resume from the latest checkpoint in output-dir.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapter training instead of full fine-tuning.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=2,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100, help="Number of warmup steps."
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="LR scheduler type (linear, cosine, cosine_with_restarts, etc.).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Adam beta1."
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.98, help="Adam beta2."
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping."
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience in eval steps (0 to disable).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bfloat16 mixed precision when supported (recommended for Ampere+).",
    )
    parser.add_argument(
        "--enable-flash-attn",
        action="store_true",
        help="Enable PyTorch flash/mem-efficient attention kernels (requires compatible GPU/torch).",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha (scaling)."
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout rate."
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA.",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    # Use EOS as padding to avoid padding errors and expand embeddings to include it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

    if args.enable_flash_attn and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch_dtype
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA; install with `pip install peft`.")
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Load JSON array of strings as text dataset
    raw_datasets = load_dataset("json", data_files=args.data_file, split="train")

    # Create a small validation slice to monitor training
    split_datasets = raw_datasets.train_test_split(test_size=0.01, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
        )

    tokenized_datasets = split_datasets.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    if args.log_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "zykeAI")
        os.environ.setdefault("WANDB_MODE", "online")
        report_to = ["wandb"]
    elif args.log_to == "tensorboard":
        report_to = ["tensorboard"]
    else:
        report_to = ["none"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        logging_steps=200,
        fp16=torch.cuda.is_available() and not (args.bf16 and torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_available() and args.bf16 and torch.cuda.is_bf16_supported(),
        report_to=report_to,
        logging_dir=str(log_dir),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
    )

    callbacks = []
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    resume_from = latest_checkpoint(output_dir) if args.resume_latest else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)

    if args.use_lora:
        final_dir = Path(args.adapter_dir) / args.adapter_name
        final_dir.mkdir(parents=True, exist_ok=True)
        # Save only the adapter weights and tokenizer for portability
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        # Manifest to describe compatibility
        manifest = {
            "base_model": model_name,
            "adapter_name": args.adapter_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": [m for m in args.lora_target_modules.split(",") if m.strip()],
            "data_file": args.data_file,
        }
        (final_dir / "adapter_manifest.json").write_text(
            __import__("json").dumps(manifest, indent=2)
        )
        print(f"LoRA adapter and tokenizer saved to {final_dir}")
        if args.hf_push:
            try:
                trainer.push_to_hub()
            except Exception as exc:  # pragma: no cover - diagnostic
                print(f"Hub push failed: {exc}")
    else:
        final_dir = Path(args.save_final_to)
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Model and tokenizer saved to {final_dir}")
        if args.hf_push:
            try:
                trainer.push_to_hub()
            except Exception as exc:  # pragma: no cover - diagnostic
                print(f"Hub push failed: {exc}")


if __name__ == "__main__":
    train()
