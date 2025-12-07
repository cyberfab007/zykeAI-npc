# Dataset Card

This project builds a cleaned language modeling corpus using public datasets streamed via Hugging Face `datasets`.

## Sources
- OpenWebText (`openwebtext`)
- English Wikipedia (`wikipedia`, snapshot `20220301.en`)

## Processing
- Streaming ingestion to avoid full downloads.
- Length filter: keep samples with token counts between configurable `min_tokens` and `max_tokens` (default 5–256, whitespace tokens).
- Profanity filter: simple word-list exclusion (see `data/build_dataset.py`).
- PII filter: drop samples containing email-like or phone-like patterns.
- Deduplication: SHA-1 hash over sample text.
- Weights/mixing: `--max-total` and `--weights source=weight,...` to control per-source share; optional `--local-dir` to include local text as `local` source.
- Splits: configurable train/val/test ratios (default 98/1/1), written to `data/processed/train.txt`, `val.txt`, `test.txt`.

## Usage
```
python data/build_dataset.py \
  --sources openwebtext,wikipedia \
  --max-total 100000 \
  --weights openwebtext=1,wikipedia=1,local=3 \
  --local-dir data/raw/ebooks \
  --min-tokens 5 --max-tokens 256 \
  --output-dir data/processed
```

## Notes
- Adjust `--max-per-source` to control total corpus size.
- For different profanity rules, pass `--profanity word1 word2 ...`.
- If you need stricter PII filtering, extend the regexes in `data/build_dataset.py`.***
