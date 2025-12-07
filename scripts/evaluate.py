import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.inference.generator import (
    ALLOWED_ACTIONS,
    ALLOWED_EMOTIONS,
    generate_npc_response,
)


REGRESSION_PROMPTS = [
    {
        "name": "greeting",
        "prompt": "NPC: Greetings, traveler. What brings you to our town?\nYou:",
        "max_new_tokens": 40,
    },
    {
        "name": "quest",
        "prompt": "Blacksmith: I can forge you a blade, but I need rare ore from the caverns.\nYou:",
        "max_new_tokens": 48,
    },
    {
        "name": "lore",
        "prompt": "Sage: Long ago, dragons ruled these skies. Tell me what you know of them.\nYou:",
        "max_new_tokens": 48,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LM checkpoint.")
    parser.add_argument(
        "--model-path",
        default="models/latest",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="models/latest",
        help="Path to tokenizer files.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional path to a LoRA adapter (if using PEFT).",
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-2-13b-hf",
        help="Base model name/path to load when applying an adapter.",
    )
    parser.add_argument(
        "--evals",
        default="wikitext2,wikitext103,c4,lambada,piqa",
        help="Comma-separated evaluations to run: wikitext2,wikitext103,c4,lambada,piqa",
    )
    parser.add_argument(
        "--piqa-samples",
        type=int,
        default=200,
        help="Number of PIQA validation examples to score.",
    )
    parser.add_argument(
        "--hellaswag-samples",
        type=int,
        default=200,
        help="Number of HellaSwag validation examples to score.",
    )
    parser.add_argument(
        "--winogrande-samples",
        type=int,
        default=200,
        help="Number of WinoGrande validation examples to score.",
    )
    parser.add_argument(
        "--arc-samples",
        type=int,
        default=200,
        help="Number of ARC examples to score.",
    )
    parser.add_argument(
        "--arc-split",
        default="ARC-Challenge",
        help="ARC split: ARC-Challenge or ARC-Easy.",
    )
    parser.add_argument(
        "--c4-split",
        default="validation[:0.1%]",
        help="C4 split for perplexity (use a small slice).",
    )
    parser.add_argument(
        "--lambada-split",
        default="validation",
        help="LAMBADA split to use for perplexity.",
    )
    parser.add_argument(
        "--npc-jsonl",
        default="data/processed/npc_val.jsonl",
        help="NPC schema dataset (JSONL) for adherence checks.",
    )
    parser.add_argument(
        "--npc-samples",
        type=int,
        default=50,
        help="Number of NPC schema samples to generate for adherence metrics.",
    )
    parser.add_argument(
        "--npc-safe-mode",
        action="store_true",
        help="Enforce safe_mode during NPC adherence eval.",
    )
    return parser.parse_args()


def compute_perplexity(model, tokenizer, dataset_name: str, config: str, split: str) -> float:
    dataset = load_dataset(dataset_name, config, split=split)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results_eval",
        per_device_eval_batch_size=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    return math.exp(eval_loss) if eval_loss is not None else float("nan")


def run_regression_prompts(model, tokenizer, device):
    model.eval()
    for item in REGRESSION_PROMPTS:
        inputs = tokenizer(
            item["prompt"], return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=item["max_new_tokens"],
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"[{item['name']}] {text}\n")


def score_option_logprob(model, tokenizer, prompt: str, option: str, device) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    option_ids = tokenizer(
        option, return_tensors="pt", add_special_tokens=False
    ).to(device)
    input_ids = torch.cat([prompt_ids.input_ids, option_ids.input_ids], dim=1)
    attention_mask = torch.cat(
        [prompt_ids.attention_mask, option_ids.attention_mask], dim=1
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    option_start = prompt_ids.input_ids.size(1)
    option_token_ids = input_ids[:, option_start:]
    option_log_probs = log_probs[:, option_start - 1 : -1, :].gather(
        2, option_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    return option_log_probs.sum().item()


def compute_piqa_accuracy(model, tokenizer, device, samples: int) -> float:
    ds = load_dataset("piqa", split=f"validation[:{samples}]")
    correct = 0
    total = 0
    for sample in ds:
        prompt = sample["goal"]
        opt1 = sample["sol1"]
        opt2 = sample["sol2"]
        score1 = score_option_logprob(model, tokenizer, prompt, opt1, device)
        score2 = score_option_logprob(model, tokenizer, prompt, opt2, device)
        pred = 0 if score1 >= score2 else 1
        if pred == sample["label"]:
            correct += 1
        total += 1
    return correct / total if total else float("nan")


def compute_hellaswag_accuracy(model, tokenizer, device, samples: int) -> float:
    ds = load_dataset("hellaswag", split=f"validation[:{samples}]")
    correct = 0
    total = 0
    for sample in ds:
        ctx = sample["ctx_a"] + " " + sample["ctx_b"]
        endings = sample["endings"]
        scores = [score_option_logprob(model, tokenizer, ctx, e, device) for e in endings]
        pred = int(torch.tensor(scores).argmax().item())
        if pred == sample["label"]:
            correct += 1
        total += 1
    return correct / total if total else float("nan")


def compute_winogrande_accuracy(model, tokenizer, device, samples: int) -> float:
    ds = load_dataset("winogrande", "winogrande_xl", split=f"validation[:{samples}]")
    correct = 0
    total = 0
    for sample in ds:
        prompt = sample["sentence"].replace("_", "")
        opts = [sample["option1"], sample["option2"]]
        scores = [score_option_logprob(model, tokenizer, prompt, o, device) for o in opts]
        pred = 0 if scores[0] >= scores[1] else 1
        label = 0 if sample["answer"] == "1" else 1
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total else float("nan")


def compute_arc_accuracy(model, tokenizer, device, samples: int, split: str) -> float:
    ds = load_dataset("ai2_arc", split=f"{split.lower()}[:{samples}]")
    correct = 0
    total = 0
    for sample in ds:
        question = sample["question"]
        choices = sample["choices"]["text"]
        scores = [score_option_logprob(model, tokenizer, question, c, device) for c in choices]
        pred_idx = int(torch.tensor(scores).argmax().item())
        pred_label = sample["choices"]["label"][pred_idx]
        if pred_label == sample["answerKey"]:
            correct += 1
        total += 1
    return correct / total if total else float("nan")


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load adapters") from exc
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evals = {e.strip() for e in args.evals.split(",") if e.strip()}
    if "wikitext2" in evals:
        try:
            ppl = compute_perplexity(
                model, tokenizer, "wikitext", "wikitext-2-raw-v1", "validation"
            )
            print(f"Perplexity (wikitext-2-raw-v1 validation): {ppl:.2f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"Wikitext-2 perplexity failed: {exc}")
    if "wikitext103" in evals:
        try:
            ppl = compute_perplexity(
                model, tokenizer, "wikitext", "wikitext-103-raw-v1", "test"
            )
            print(f"Perplexity (wikitext-103-raw-v1 test): {ppl:.2f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"Wikitext-103 perplexity failed: {exc}")
    if "c4" in evals:
        try:
            ppl = compute_perplexity(
                model, tokenizer, "c4", "en", args.c4_split
            )
            print(f"Perplexity (C4 {args.c4_split}): {ppl:.2f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"C4 perplexity failed: {exc}")
    if "lambada" in evals:
        try:
            ppl = compute_perplexity(
                model, tokenizer, "lambada", "plain_text", args.lambada_split
            )
            print(f"Perplexity (LAMBADA {args.lambada_split}): {ppl:.2f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"LAMBADA perplexity failed: {exc}")
    if "piqa" in evals:
        try:
            acc = compute_piqa_accuracy(model, tokenizer, device, args.piqa_samples)
            print(f"PIQA accuracy (first {args.piqa_samples} val): {acc:.3f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"PIQA evaluation failed: {exc}")
    if "hellaswag" in evals:
        try:
            acc = compute_hellaswag_accuracy(model, tokenizer, device, args.hellaswag_samples)
            print(f"HellaSwag accuracy (first {args.hellaswag_samples} val): {acc:.3f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"HellaSwag evaluation failed: {exc}")
    if "winogrande" in evals:
        try:
            acc = compute_winogrande_accuracy(model, tokenizer, device, args.winogrande_samples)
            print(f"WinoGrande accuracy (first {args.winogrande_samples} val): {acc:.3f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"WinoGrande evaluation failed: {exc}")
    if "arc" in evals:
        try:
            acc = compute_arc_accuracy(model, tokenizer, device, args.arc_samples, args.arc_split)
            print(f"ARC accuracy ({args.arc_split}, first {args.arc_samples}): {acc:.3f}")
        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"ARC evaluation failed: {exc}")

    # NPC adherence check
    npc_path = Path(args.npc_jsonl)
    if npc_path.exists():
        try:
            with npc_path.open("r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            samples = lines[: args.npc_samples]
            valid = 0
            total = 0
            for sample in samples:
                result = generate_npc_response(
                    base_model=args.base_model,
                    tokenizer_path=args.tokenizer_path,
                    adapter_path=args.adapter_path,
                    persona=sample.get("persona", ""),
                    context=sample.get("context", ""),
                    state=sample.get("state", ""),
                    player_input=sample.get("player", ""),
                    safe_mode=args.npc_safe_mode,
                )
                total += 1
                if (
                    isinstance(result, dict)
                    and result.get("say")
                    and result.get("action") in ALLOWED_ACTIONS
                    and result.get("emotion") in ALLOWED_EMOTIONS
                ):
                    valid += 1
            if total:
                print(f"NPC schema adherence (valid JSON/enums): {valid/total:.3f} ({valid}/{total})")
        except Exception as exc:
            print(f"NPC adherence evaluation failed: {exc}")

    print("Regression prompt generations:")
    run_regression_prompts(model, tokenizer, device)


if __name__ == "__main__":
    main()
