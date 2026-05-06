"""Fetch the FoodLensVN Phase-1 dataset from HuggingFace Hub.

Pulls ``Tamir39/foodlensvn`` and materializes it under ``<dest>``:

  <dest>/
    annotations/{train,val,test}.json
    images/{raw,squared}/{train,val,test}/<file>.jpg

After this, ``scripts/build_dataset.py --data-dir <dest>`` works locally.

Auth: a public dataset doesn't strictly require a token, but ``hf auth login``
(or ``HF_TOKEN`` env var) avoids rate limits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Tamir39/foodlensvn"
REPO_TYPE = "dataset"
DEFAULT_DEST = Path("data/foodlensvn")


def _download(dest: Path, force: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: uv sync")

    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {REPO_ID} -> {dest}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(dest),
        force_download=force,
        token=os.environ.get("HF_TOKEN"),
    )


def _summarize(dest: Path) -> None:
    ann_dir = dest / "annotations"
    img_dir = dest / "images"
    if not ann_dir.is_dir() or not img_dir.is_dir():
        sys.exit(
            f"unexpected layout under {dest}; expected annotations/ and images/ subdirs"
        )
    n_imgs = sum(1 for p in img_dir.rglob("*.jpg"))
    n_jsons = sum(1 for p in ann_dir.glob("*.json"))
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(
        f"done: {n_imgs} images, {n_jsons} annotation files, "
        f"{total_bytes / 1024 / 1024:.1f} MB at {dest}"
    )
    print(f"\nnext: python scripts/build_dataset.py --data-dir {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dest",
        default=os.environ.get("FOODLENS_DATA_DIR") or str(DEFAULT_DEST),
        help=(
            "destination directory (default: $FOODLENS_DATA_DIR or "
            f"'{DEFAULT_DEST}')."
        ),
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.is_dir() and (dest / "annotations").is_dir() and not args.force:
        print(f"{dest} already populated; skipping download (pass --force to redownload)")
        _summarize(dest)
        return

    _download(dest, force=args.force)
    _summarize(dest)


if __name__ == "__main__":
    main()
