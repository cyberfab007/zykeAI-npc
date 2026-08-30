"""
Builds experience blocks (JSONL) from local text/JSONL corpora.
Each block conforms to schemas/experience_block.json:
{
  "block_id": "...",
  "policy_version": 0,
  "env_id": "...",
  "npc_type": "...",
  "steps": [
    { "obs": [token_ids], "action": next_token_id, "reward": 0.0, "done": bool }
  ]
}
"""
import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Generator, List, Optional

from transformers import AutoTokenizer


def iter_texts(path: Path) -> Generator[str, None, None]:
    """Yield raw text strings from .txt or .jsonl files (recursing into directories)."""
    if path.is_dir():
        for file in sorted(path.rglob("*")):
            yield from iter_texts(file)
        return

    if path.suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
        return

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Try common fields in order
                for key in ("text", "content", "prompt", "completion"):
                    if key in obj and obj[key]:
                        yield str(obj[key]).strip()
                        break
        return

    # Unsupported extension; skip silently


def build_steps(token_ids: List[int], seq_len: int) -> List[dict]:
    """Create (obs, action, reward, done) steps from a token sequence."""
    steps: List[dict] = []
    if len(token_ids) <= seq_len:
        return steps
    for idx in range(seq_len, len(token_ids)):
        obs = token_ids[idx - seq_len : idx]
        action = token_ids[idx]
        steps.append({"obs": obs, "action": action, "reward": 0.0, "done": False})
    if steps:
        steps[-1]["done"] = True
    return steps


def write_blocks(
    texts: Generator[str, None, None],
    tokenizer,
    seq_len: int,
    steps_per_block: int,
    out_path: Path,
    policy_version: int,
    env_id: str,
    npc_type: str,
    max_blocks: Optional[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    block_steps: List[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for text in texts:
            tokenized = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )
            ids = tokenized["input_ids"]
            steps = build_steps(ids, seq_len=seq_len)
            for step in steps:
                block_steps.append(step)
                if len(block_steps) >= steps_per_block:
                    block = {
                        "block_id": str(uuid.uuid4()),
                        "policy_version": policy_version,
                        "env_id": env_id,
                        "npc_type": npc_type,
                        "steps": block_steps,
                    }
                    canonical = json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    block["block_hash"] = hashlib.sha256(canonical).hexdigest()
                    f.write(json.dumps(block) + "\n")
                    written += 1
                    block_steps = []
                    if max_blocks and written >= max_blocks:
                        return
        # Flush remainder
        if block_steps and (not max_blocks or written < max_blocks):
            block = {
                "block_id": str(uuid.uuid4()),
                "policy_version": policy_version,
                "env_id": env_id,
                "npc_type": npc_type,
                "steps": block_steps,
            }
            canonical = json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")
            block["block_hash"] = hashlib.sha256(canonical).hexdigest()
            f.write(json.dumps(block) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Convert local corpora into experience blocks JSONL.")
    p.add_argument("--input-path", required=True, help="File or directory of .txt/.jsonl data (recurses).")
    p.add_argument("--output", default="data/processed/experience_blocks.jsonl", help="Output JSONL path.")
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-13b-hf", help="HF tokenizer name or local path.")
    p.add_argument("--seq-len", type=int, default=128, help="Context length for obs (tokens).")
    p.add_argument("--steps-per-block", type=int, default=64, help="Number of steps per block.")
    p.add_argument("--policy-version", type=int, default=0, help="Policy/model version tag to embed.")
    p.add_argument("--env-id", default="offline_corpus", help="Environment/source identifier.")
    p.add_argument("--npc-type", default="generic", help="NPC type tag.")
    p.add_argument("--max-blocks", type=int, default=None, help="Optional cap on number of blocks.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    out_path = Path(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = iter_texts(input_path)
    write_blocks(
        texts=texts,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        steps_per_block=args.steps_per_block,
        out_path=out_path,
        policy_version=args.policy_version,
        env_id=args.env_id,
        npc_type=args.npc_type,
        max_blocks=args.max_blocks,
    )
    print(f"Wrote experience blocks to {out_path}")


if __name__ == "__main__":
    main()
