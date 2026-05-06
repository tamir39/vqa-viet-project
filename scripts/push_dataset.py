"""Push the FoodLensVN Phase-1 dataset to HuggingFace Hub.

Repo: https://huggingface.co/datasets/Tamir39/foodlensvn (public, CC-BY-SA-4.0).

Run once after `hf auth login` (or with HF_TOKEN env set):

    python scripts/push_dataset.py

Source: ``dist/foodlensvn_kaggle/`` (built by ``dist/build_kaggle_dataset.py``).
Override with ``--source`` if you renamed the staging directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "Tamir39/foodlensvn"
REPO_TYPE = "dataset"
DEFAULT_SOURCE = ROOT / "dist" / "foodlensvn_kaggle"
HF_README = ROOT / "data" / "HF_README.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"local staging dir to upload (default: {DEFAULT_SOURCE.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"HF dataset repo id (default: {REPO_ID})",
    )
    args = parser.parse_args()

    src: Path = args.source
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")
    if not (src / "annotations").is_dir() or not (src / "images").is_dir():
        raise FileNotFoundError(
            f"expected annotations/ and images/ under {src} — run dist/build_kaggle_dataset.py first"
        )
    if not HF_README.is_file():
        raise FileNotFoundError(f"missing dataset card: {HF_README}")

    api = HfApi()
    print(f"whoami: {api.whoami()['name']}")

    create_repo(args.repo_id, repo_type=REPO_TYPE, exist_ok=True, private=False)
    print(f"repo ready: https://huggingface.co/datasets/{args.repo_id}")

    print(f"  uploading dataset card -> README.md")
    api.upload_file(
        path_or_fileobj=str(HF_README),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        commit_message="upload dataset card",
    )

    print(f"  uploading {src} -> /")
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo_id,
        repo_type=REPO_TYPE,
        ignore_patterns=["dataset-metadata.json", ".DS_Store", "**/__pycache__/**"],
        commit_message="upload Phase-1 corpus (5572 rows, 20 dishes, raw+squared)",
    )

    print(f"\ndone -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
