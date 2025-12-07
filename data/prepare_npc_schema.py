"""
Validate and split NPC schema data for adapter training.

Input: JSONL with fields: persona, context, state, player, target (dict with say/action/emotion/thoughts).
Outputs: train/val JSONL files under data/processed/npc_{train,val}.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


REQUIRED_FIELDS = {"persona", "context", "state", "player", "target"}
TARGET_FIELDS = {"say", "action", "emotion", "thoughts"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NPC schema dataset.")
    parser.add_argument(
        "--input",
        default="data/npc_sample.jsonl",
        help="Path to input JSONL with NPC schema samples.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write npc_train.jsonl and npc_val.jsonl",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                continue
    return data


def validate_sample(sample: Dict) -> bool:
    if not REQUIRED_FIELDS.issubset(sample.keys()):
        return False
    target = sample.get("target", {})
    if not isinstance(target, dict):
        return False
    if not TARGET_FIELDS.issubset(target.keys()):
        return False
    return True


def dedup(samples: List[Dict]) -> List[Dict]:
    seen = set()
    uniq = []
    for s in samples:
        key = f"{s.get('persona','')}|{s.get('context','')}|{s.get('state','')}|{s.get('player','')}|{json.dumps(s.get('target',{}), sort_keys=True)}"
        h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(s)
    return uniq


def split(samples: List[Dict], val_ratio: float):
    n = len(samples)
    val_count = max(1, int(n * val_ratio)) if n > 1 else 0
    return samples[val_count:], samples[:val_count]


def write_jsonl(path: Path, samples: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    data = load_jsonl(Path(args.input))
    data = [s for s in data if validate_sample(s)]
    data = dedup(data)
    train, val = split(data, args.val_ratio)
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "npc_train.jsonl", train)
    write_jsonl(out_dir / "npc_val.jsonl", val)
    print(f"Wrote {len(train)} train and {len(val)} val samples to {out_dir}")


if __name__ == "__main__":
    main()
