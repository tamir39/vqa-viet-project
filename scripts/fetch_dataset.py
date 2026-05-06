"""Fetch the FoodLensVN Phase-1 dataset from Kaggle.

Pulls ``phvngtngtm/foodlensvn`` and unzips into ``<dest>``, which then mirrors
the same layout Kaggle mounts at ``/kaggle/input/foodlensvn``:

  <dest>/
    annotations/{train,val,test}.json
    images/{raw,squared}/{train,val,test}/<file>.jpg

After this, ``scripts/build_dataset.py --data-dir <dest>`` works locally.

Auth: requires ``~/.kaggle/kaggle.json`` (download from
https://www.kaggle.com/settings -> "Create New Token").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATASET = "phvngtngtm/foodlensvn"
DEFAULT_DEST = Path("data/foodlensvn")


def _ensure_kaggle_creds() -> None:
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]
    if not any(p.exists() for p in candidates):
        sys.exit(
            "missing Kaggle credentials. Create a token at "
            "https://www.kaggle.com/settings and save it as "
            "~/.kaggle/kaggle.json"
        )


def _download(dest: Path, force: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle package not installed. Run: uv add kaggle  (or pip install kaggle)")

    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DATASET} -> {dest}")
    api.dataset_download_files(DATASET, path=str(dest), unzip=True, force=force, quiet=False)


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
        default=os.environ.get("KAGGLE_INPUT_DIR") or str(DEFAULT_DEST),
        help=(
            "destination directory (default: $KAGGLE_INPUT_DIR or "
            f"'{DEFAULT_DEST}'). On Kaggle the dataset is already mounted; "
            "this script is a no-op there."
        ),
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.is_dir() and (dest / "annotations").is_dir() and not args.force:
        print(f"{dest} already populated; skipping download (pass --force to redownload)")
        _summarize(dest)
        return

    _ensure_kaggle_creds()
    _download(dest, force=args.force)
    _summarize(dest)


if __name__ == "__main__":
    main()
