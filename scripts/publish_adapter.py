#!/usr/bin/env python3
"""
Publish a local LoRA adapter directory to the Hugging Face Hub using a manifest entry.

Usage:
  python scripts/publish_adapter.py --adapter-name myadapter --manifest models/adapters/manifest.json --repo-id your-user/your-adapter-repo
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

from src.models.adapter_manifest import AdapterManifestError, load_manifest


def parse_args():
    p = argparse.ArgumentParser(description="Publish a LoRA adapter to the Hugging Face Hub.")
    p.add_argument("--adapter-name", required=True, help="Adapter name as listed in the manifest.")
    p.add_argument("--manifest", default="models/adapters/manifest.json", help="Path to adapter manifest.")
    p.add_argument("--repo-id", required=True, help="HF repo id to upload to (e.g., username/adapter-name).")
    p.add_argument("--private", action="store_true", help="Create repo as private.")
    p.add_argument("--token", default=None, help="HF token (or set HUGGINGFACE_HUB_TOKEN).")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if args.adapter_name not in manifest:
        raise AdapterManifestError(f"{args.adapter_name} not found in {args.manifest}")
    entry = manifest[args.adapter_name]
    adapter_dir = Path(entry["adapter_path"])
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=bool(args.private))
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        path_in_repo=".",
        commit_message=f"Upload adapter {args.adapter_name} version {entry.get('version','unknown')}",
    )
    print(f"Uploaded {adapter_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
