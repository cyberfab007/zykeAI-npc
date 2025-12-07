"""
Build a cleaned language modeling corpus from public datasets with streaming, filtering, and deduplication.

Sources: OpenWebText and Wikipedia (Hugging Face datasets).
Outputs: train/val/test text files under data/processed/.
"""
import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import load_dataset, IterableDataset


DEFAULT_PROFANITY = {
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "dick",
    "cunt",
    "faggot",
}


EMAIL_RE = re.compile(r"[\\w.+-]+@[\\w-]+\\.[\\w.-]+")
PHONE_RE = re.compile(r"(\\+?\\d[\\d\\-\\s]{7,}\\d)")


def stream_openwebtext() -> Iterable[str]:
    ds: IterableDataset = load_dataset("openwebtext", split="train", streaming=True)
    for sample in ds:
        text = sample.get("text")
        if text:
            yield text


def stream_wikipedia() -> Iterable[str]:
    ds: IterableDataset = load_dataset(
        "wikipedia", "20220301.en", split="train", streaming=True
    )
    for sample in ds:
        text = sample.get("text")
        if text:
            yield text


def stream_local_dir(path: Path) -> Iterable[str]:
    for fp in path.rglob("*"):
        if fp.is_file():
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                if text:
                    yield text
            except (OSError, UnicodeDecodeError):
                continue


def passes_filters(
    text: str,
    min_tokens: int,
    max_tokens: int,
    profanity: set[str],
) -> bool:
    tokens = text.strip().split()
    if not (min_tokens <= len(tokens) <= max_tokens):
        return False
    lower = text.lower()
    if any(word in lower for word in profanity):
        return False
    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    return True


def deduped(samples: Iterable[str], max_items: int) -> Iterator[str]:
    seen = set()
    for text in samples:
        h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield text
        if 0 < max_items <= len(seen):
            break


def write_split(
    rows: List[str],
    train_ratio: float,
    val_ratio: float,
    output_dir: Path,
) -> None:
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    splits = {
        "train.txt": rows[:train_end],
        "val.txt": rows[train_end:val_end],
        "test.txt": rows[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in splits.items():
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line.strip() + "\\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream, filter, dedup, and split text datasets."
    )
    parser.add_argument(
        "--sources",
        default="openwebtext,wikipedia",
        help="Comma-separated list of sources: openwebtext,wikipedia",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=50000,
        help="Maximum samples to collect per source (after dedup/filter).",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Total target samples (after dedup/filter); 0 uses max-per-source for each.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=5,
        help="Minimum token count to keep a sample.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum token count to keep a sample.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.98, help="Train split ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.01, help="Validation split ratio."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write train/val/test text files.",
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Source weights (comma-separated, e.g., openwebtext=1,wikipedia=1,local=3).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local text directory to include as 'local' source.",
    )
    parser.add_argument(
        "--profanity",
        nargs="*",
        default=None,
        help="Optional override list of profanity words to filter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    profanity = set(args.profanity) if args.profanity else DEFAULT_PROFANITY

    source_map = {
        "openwebtext": stream_openwebtext,
        "wikipedia": stream_wikipedia,
    }
    if args.local_dir:
        local_path = Path(args.local_dir)
        source_map["local"] = lambda: stream_local_dir(local_path)
    selected_sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    def parse_weights(weight_str: str) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        if not weight_str:
            return weights
        for item in weight_str.split(","):
            if "=" not in item:
                continue
            name, val = item.split("=", 1)
            try:
                weights[name.strip()] = float(val)
            except ValueError:
                continue
        return weights

    weights = parse_weights(args.weights)
    if args.max_total > 0:
        total_weight = sum(weights.get(s, 1.0) for s in selected_sources)
        per_source_target = {
            s: int(args.max_total * (weights.get(s, 1.0) / total_weight))
            for s in selected_sources
        }
    else:
        per_source_target = {s: args.max_per_source for s in selected_sources}

    collected: List[str] = []
    for source in selected_sources:
        if source not in source_map:
            print(f"Unknown source '{source}', skipping.")
            continue
        print(f"Streaming source: {source}")
        stream_fn = source_map[source]
        filtered = (
            text
            for text in stream_fn()
            if passes_filters(text, args.min_tokens, args.max_tokens, profanity)
        )
        target = per_source_target.get(source, args.max_per_source)
        for text in deduped(filtered, target):
            collected.append(text)
        print(f"Collected {len(collected)} total samples after {source}")

    if not collected:
        print("No samples collected; nothing to write.")
        return

    write_split(
        collected,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        output_dir=output_dir,
    )
    print(f"Wrote splits to {output_dir}")


if __name__ == "__main__":
    main()
