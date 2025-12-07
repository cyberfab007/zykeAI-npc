import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
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
    parser = argparse.ArgumentParser(description="Train GPT-2 on local data.")
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
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument(
        "--lora-alpha", type=int, default=16, help="LoRA alpha (scaling)."
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout rate."
    )
    parser.add_argument(
        "--lora-target-modules",
        default="c_attn,c_proj",
        help="Comma-separated target modules for LoRA.",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Use EOS as padding to avoid padding errors and expand embeddings to include it
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
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
            max_length=256,
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
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="epoch",
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=500,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        logging_dir=str(log_dir),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
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
