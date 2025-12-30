#!/usr/bin/env python3
"""
Training entrypoint for ZykeAI.

Supports:
- Full fine-tune of a HF causal LM
- LoRA adapter training (recommended)
- Resume from latest checkpoint
- Standardized output dirs:
    - checkpoints: models/checkpoints
    - final/latest: models/latest
    - adapters: models/adapters/<adapter_name>

Notes:
- Adapters are only compatible with the base model they were trained on.
- For smaller, fully open models that work well with this stack, consider:
  EleutherAI/pythia-410m-deduped or EleutherAI/pythia-1.4b-deduped
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def _default_target_modules(model_type: str) -> List[str]:
    mt = (model_type or "").lower()
    if mt in {"llama", "mistral"}:
        return ["q_proj", "v_proj"]
    if mt in {"gpt_neox"}:
        # Pythia / GPT-NeoX
        return ["query_key_value"]
    if mt in {"gpt2"}:
        return ["c_attn", "c_proj"]
    return []


def _parse_target_modules(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    modules = [m.strip() for m in value.split(",") if m.strip()]
    return modules or None


def _precision_kwargs(args) -> Dict:
    if not torch.cuda.is_available():
        return {"torch_dtype": None}
    # Prefer bf16 when supported and requested (LLaMA typically prefers bf16).
    if args.bf16 and torch.cuda.is_bf16_supported():
        return {"torch_dtype": torch.bfloat16}
    if args.fp16:
        return {"torch_dtype": torch.float16}
    # Auto: use bf16 if supported, else fp16.
    return {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}


@dataclass
class SFTExample:
    input: str
    target: str


def _iter_starter_jsonl(path: Path) -> Iterable[SFTExample]:
    for fp in sorted(path.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = str(obj.get("input", "")).strip()
                tgt = str(obj.get("target", "")).strip()
                if inp and tgt:
                    yield SFTExample(input=inp, target=tgt)


class JsonlSFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SFTExample], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = ex.input.rstrip() + "\n"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=self.max_length)
        target_ids = self.tokenizer(ex.target, add_special_tokens=False, truncation=True, max_length=self.max_length)

        input_ids = prompt_ids["input_ids"] + target_ids["input_ids"]
        input_ids = input_ids[: self.max_length]
        attention_mask = [1] * len(input_ids)

        # Mask prompt tokens so loss only applies to the completion.
        labels = ([-100] * len(prompt_ids["input_ids"])) + target_ids["input_ids"]
        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate_sft(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(x["input_ids"].shape[0]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for item in batch:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            lab = item["labels"]
            pad_len = max_len - int(ids.shape[0])
            if pad_len:
                ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros((pad_len,), dtype=torch.long)])
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

    return collate


def parse_args():
    p = argparse.ArgumentParser(description="Train a base model or LoRA adapter.")
    p.add_argument("--base-model", default=os.getenv("BASE_MODEL", "meta-llama/Llama-2-13b-hf"))
    p.add_argument("--tokenizer", default=None, help="Tokenizer name/path (defaults to --base-model).")

    data = p.add_mutually_exclusive_group(required=False)
    data.add_argument("--train-txt", default=None, help="Path to a text file for training (one document per line).")
    data.add_argument(
        "--starter-blocks",
        default=None,
        help="Directory of small JSONL files with {input,target} lines (e.g., data/starter_blocks).",
    )

    p.add_argument("--val-txt", default=None, help="Optional validation text file.")
    p.add_argument("--max-seq-len", type=int, default=512)

    p.add_argument("--output-dir", default="models/checkpoints", help="Trainer output_dir (checkpoints).")
    p.add_argument("--latest-dir", default="models/latest", help="Where to write final model/tokenizer.")
    p.add_argument("--resume-latest", action="store_true", help="Resume from the latest checkpoint in output_dir.")

    # Training hyperparameters
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--lr-scheduler", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epoch-based training when > 0.")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-to", default="none", choices=["none", "tensorboard", "wandb"])

    # Precision
    p.add_argument("--bf16", action="store_true", help="Prefer bf16 when supported (CUDA).")
    p.add_argument("--fp16", action="store_true", help="Prefer fp16 on CUDA.")

    # LoRA / adapters
    p.add_argument("--use-lora", action="store_true", help="Train a LoRA adapter instead of full fine-tune.")
    p.add_argument("--adapter-name", default=None, help="Adapter name (required when --use-lora).")
    p.add_argument("--adapter-out", default=None, help="Adapter output dir (defaults to models/adapters/<name>).")
    p.add_argument("--manifest-path", default="data/adapters/manifest.json")
    p.add_argument("--update-manifest", action="store_true", help="Write/overwrite manifest entry for adapter.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated module names. If unset, picks defaults by model type (llama/gpt_neox/gpt2).",
    )
    return p.parse_args()


def _write_manifest_entry(
    manifest_path: str,
    adapter_name: str,
    adapter_path: str,
    base_model: str,
    target_modules: List[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> None:
    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if mp.exists():
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data[adapter_name] = {
        "name": adapter_name,
        "adapter_path": adapter_path,
        "base_model": base_model,
        "target_modules": target_modules,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "version": data.get(adapter_name, {}).get("version", "0.1.0"),
    }
    mp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    base_model = args.base_model
    tokenizer_name = args.tokenizer or base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    precision = _precision_kwargs(args)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, **precision)

    data_collator = None
    train_dataset = None
    eval_dataset = None

    if args.starter_blocks:
        examples = list(_iter_starter_jsonl(Path(args.starter_blocks)))
        if not examples:
            raise SystemExit(f"No SFT examples found under {args.starter_blocks}")
        # tiny split if no eval provided
        split = max(1, int(len(examples) * 0.95))
        train_dataset = JsonlSFTDataset(examples[:split], tokenizer, max_length=args.max_seq_len)
        eval_dataset = JsonlSFTDataset(examples[split:], tokenizer, max_length=args.max_seq_len) if split < len(examples) else None
        data_collator = _collate_sft(tokenizer)
    else:
        if not args.train_txt:
            raise SystemExit("Provide --train-txt or --starter-blocks")
        files = {"train": args.train_txt}
        if args.val_txt:
            files["validation"] = args.val_txt
        raw = load_dataset("text", data_files=files)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_len)

        tokenized = raw.map(tokenize_function, batched=True, remove_columns=["text"])
        train_dataset = tokenized["train"]
        eval_dataset = tokenized.get("validation")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    adapter_out = None
    if args.use_lora:
        if not args.adapter_name:
            raise SystemExit("--adapter-name is required when --use-lora")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("peft is required for --use-lora") from exc
        target_modules = _parse_target_modules(args.lora_target_modules)
        if target_modules is None:
            target_modules = _default_target_modules(getattr(model.config, "model_type", ""))
        if not target_modules:
            raise SystemExit(
                "Unable to infer --lora-target-modules for this model; pass --lora-target-modules explicitly."
            )
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        adapter_out = args.adapter_out or str(Path("models/adapters") / args.adapter_name)
        Path(adapter_out).mkdir(parents=True, exist_ok=True)

        if args.update_manifest:
            _write_manifest_entry(
                manifest_path=args.manifest_path,
                adapter_name=args.adapter_name,
                adapter_path=adapter_out,
                base_model=base_model,
                target_modules=target_modules,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

    # TrainingArguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from = None
    if args.resume_latest:
        last_ckpt = get_last_checkpoint(str(output_dir))
        if last_ckpt:
            resume_from = last_ckpt

    # Auto mixed precision flags
    use_bf16 = bool(torch.cuda.is_available() and args.bf16 and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and (args.fp16 or (not args.bf16 and not use_bf16)))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=args.save_total_limit,
        report_to=None if args.log_to == "none" else args.log_to,
        seed=args.seed,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Save outputs
    if args.use_lora:
        assert adapter_out is not None
        model.save_pretrained(adapter_out)
        tokenizer.save_pretrained(adapter_out)
        print(f"Saved adapter to {adapter_out}")
    else:
        latest_dir = Path(args.latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(latest_dir))
        tokenizer.save_pretrained(str(latest_dir))
        print(f"Saved model to {latest_dir}")


if __name__ == "__main__":
    main()

