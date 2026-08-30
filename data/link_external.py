"""
Create symlinks from the repo to an external data root (e.g., /mnt/SSD1TB/ZYKE_DATA).

This links:
- data/raw -> <external>/data/raw
- data/processed -> <external>/data/processed
- models -> <external>/models
- results -> <external>/results
- logs -> <external>/logs
"""
import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple


LINK_MAP = {
    Path("data/raw"): Path("data/raw"),
    Path("data/processed"): Path("data/processed"),
    Path("models"): Path("models"),
    Path("results"): Path("results"),
    Path("logs"): Path("logs"),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relink(src: Path, dest: Path) -> None:
    if src.is_symlink() or src.exists():
        if src.is_symlink():
            src.unlink()
        else:
            raise RuntimeError(f"{src} exists and is not a symlink; remove or move it first.")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.symlink_to(dest)


def link_all(external_root: Path) -> Iterable[Tuple[Path, Path]]:
    created = []
    for local, relative in LINK_MAP.items():
        target = external_root / relative
        ensure_dir(target)
        relink(local, target)
        created.append((local, target))
    return created


def parse_args():
    parser = argparse.ArgumentParser(
        description="Link data/models/results/logs to an external root (e.g., /mnt/SSD1TB/ZYKE_DATA)."
    )
    parser.add_argument(
        "--external-root",
        default=os.environ.get("EXTERNAL_ROOT", "/mnt/SSD1TB/ZYKE_DATA"),
        help="External root path to house data/models/results/logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    external_root = Path(args.external_root).expanduser().resolve()
    external_root.mkdir(parents=True, exist_ok=True)
    mapped = link_all(external_root)
    for local, target in mapped:
        print(f"Linked {local} -> {target}")


if __name__ == "__main__":
    main()
