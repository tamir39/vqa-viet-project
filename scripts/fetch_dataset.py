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
        from huggingface_hub import HfApi, snapshot_download
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
        max_workers=8,
    )

    # Verify completeness: HF list_repo_files vs. what landed locally.
    # snapshot_download is meant to be idempotent — but a previous run that
    # was killed mid-transfer can leave a partial tree. Compare counts and
    # re-call snapshot_download (which only fetches what's missing) until the
    # local tree matches the remote.
    api = HfApi()
    expected = {f for f in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
                if f.endswith(".jpg") or f.endswith(".json")}
    for attempt in range(3):
        local = {
            str(p.relative_to(dest)).replace("\\", "/")
            for p in dest.rglob("*")
            if p.is_file() and (p.suffix == ".jpg" or p.suffix == ".json")
        }
        missing = expected - local
        if not missing:
            return
        print(
            f"  retry {attempt + 1}/3: {len(missing)} files still missing "
            f"(e.g. {sorted(missing)[0]}) — resuming download"
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(dest),
            token=os.environ.get("HF_TOKEN"),
            max_workers=8,
        )

    sys.exit(
        f"download still incomplete after retries: {len(missing)} files missing. "
        "re-run with --force to redownload from scratch."
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
    # Always call _download — snapshot_download skips files whose etag matches
    # the local copy, so re-running is cheap when nothing changed but heals a
    # partial tree from an interrupted earlier run.
    _download(dest, force=args.force)
    _summarize(dest)


if __name__ == "__main__":
    main()
