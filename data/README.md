# Data

- `raw/` and `processed/` are gitignored; place source corpora in `raw/` and write outputs to `processed/`.
- Use `prepare_data.py` as a starting point for cleaning/splitting your text.
- `lib/wikipedia-downloader` contains a helper to fetch the 2020 Wikipedia dump and convert it to JSON shards.
- `make_experience_blocks.py` turns local corpora (.txt/.jsonl) into training blocks that match `schemas/experience_block.json`, suitable for the distributed trainer.
- `starter_blocks/` holds tiny JSONL starter sets for smoke tests and warmup (general text, dialogue, code, lore, instructions). Each line: `{"input": "...", "target": "...", "tags": [...]}`.

Example:
```
python data/make_experience_blocks.py \
  --input-path data/raw/ebooks \
  --output data/processed/experience_blocks.jsonl \
  --tokenizer meta-llama/Llama-2-13b-hf \
  --seq-len 128 \
  --steps-per-block 64 \
  --env-id ebooks \
  --npc-type generic
```
