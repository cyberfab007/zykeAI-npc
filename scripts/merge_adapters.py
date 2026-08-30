#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model (or merge multiple adapters sequentially).
Outputs a merged model directory for export/inference without LoRA dependency.

Example:
  python scripts/merge_adapters.py \
    --base-model meta-llama/LLaMA-2-13b-hf \
    --adapter-path models/adapters/myadapter \
    --output models/merged/myadapter-merged
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Merge one or more LoRA adapters into a base model.")
    p.add_argument("--base-model", required=True, help="Base model name/path.")
    p.add_argument("--adapter-path", action="append", required=True, help="One or more adapter paths to merge.")
    p.add_argument("--output", required=True, help="Output directory for merged model.")
    return p.parse_args()


def merge_and_save(base_model: str, adapters: list[str], output: Path):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    for adapter in adapters:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to merge adapters") from exc
        model = PeftModel.from_pretrained(model, adapter)
    # Merge and drop adapter modules
    model = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Merged {len(adapters)} adapter(s) into {output}")


def main():
    args = parse_args()
    output = Path(args.output)
    merge_and_save(args.base_model, args.adapter_path, output)


if __name__ == "__main__":
    main()
