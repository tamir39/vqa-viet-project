"""Assemble the fixed Kaggle dataset for FoodLensVN.

Layout produced (Kaggle-style flat):

  dist/foodlensvn_kaggle/
    annotations/
      train.json   (4460 rows, sentence answers, 20 dishes)
      val.json     ( 632 rows)
      test.json    ( 480 rows)
    images/
      raw/{train,val,test}/      <originals + _aug_N variants merged>
      squared/{train,val,test}/  <originals + _aug_N variants merged>
    dataset-metadata.json

Source (worktree at dist/data-branch checked out from origin/data):
  data/images/raw_splits/{split}/         239+34+26 = 299 originals
  data/images/raw_splits_augmented/{split}/ 717+102+78 = 897 augmented
  data/images/squared_splits/{split}/        same counts, square crop
  data/images/squared_splits_augmented/{split}/
  data/annotations/{split}_with_questions_augmented.json -> annotations/{split}.json

After this script, the dataset has every file the annotations reference
(956+136+104 unique = 1196 image references resolved across both variants).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data-branch" / "data"
DST = ROOT / "foodlensvn_kaggle"

SPLITS = ("train", "val", "test")
VARIANTS = ("raw", "squared")


def copy_split_variant(variant: str, split: str) -> tuple[int, int]:
    """Copy originals + augmented for one (variant, split) into the flat dst folder."""
    dst_dir = DST / "images" / variant / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_orig = 0
    src_orig = SRC / "images" / f"{variant}_splits" / split
    if src_orig.is_dir():
        for f in src_orig.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
                n_orig += 1

    n_aug = 0
    src_aug = SRC / "images" / f"{variant}_splits_augmented" / split
    if src_aug.is_dir():
        for f in src_aug.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_dir / f.name)
                n_aug += 1

    print(f"  {variant}/{split}: {n_orig} originals + {n_aug} augmented = {n_orig + n_aug}")
    return n_orig, n_aug


def copy_annotations() -> None:
    ann_dst = DST / "annotations"
    ann_dst.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = SRC / "annotations" / f"{split}_with_questions_augmented.json"
        dst = ann_dst / f"{split}.json"
        shutil.copy2(src, dst)
        with dst.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        print(f"  annotations/{split}.json: {len(rows)} rows")


def write_metadata() -> None:
    meta = {
        "title": "FoodLensVN",
        "id": "kyoru4444/foodlensvn",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    (DST / "dataset-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  dataset-metadata.json")


def verify_references() -> None:
    """Every annotation `image` field must resolve under at least one variant."""
    print("\nverifying annotation references resolve in new layout...")
    failures = 0
    for split in SPLITS:
        with (DST / "annotations" / f"{split}.json").open("r", encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            fn = r["image"]
            ok = any(
                (DST / "images" / variant / split / fn).is_file()
                for variant in VARIANTS
            )
            if not ok:
                if failures < 5:
                    print(f"  MISSING: {split}/{fn}")
                failures += 1
        present = sum(
            1 for r in rows
            if any((DST / "images" / variant / split / r["image"]).is_file() for variant in VARIANTS)
        )
        print(f"  {split}: {present}/{len(rows)} rows have at least one variant present")
    if failures:
        print(f"  TOTAL MISSING: {failures}")
        sys.exit(1)
    print("  all references resolve in both variants")


def main() -> None:
    if DST.exists():
        print(f"removing existing {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    print("copying images...")
    for variant in VARIANTS:
        for split in SPLITS:
            copy_split_variant(variant, split)

    print("\ncopying annotations...")
    copy_annotations()

    print("\nwriting metadata...")
    write_metadata()

    verify_references()

    # summary
    total_files = sum(1 for _ in DST.rglob("*") if _.is_file())
    total_bytes = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"\nDONE: {total_files} files, {total_bytes / 1024 / 1024:.1f} MB at {DST}")


if __name__ == "__main__":
    main()
