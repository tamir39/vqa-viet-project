"""Build train/val/test splits for FoodLensVN with strict validation and answer vocab.

The Phase-1 corpus is hosted as the Kaggle dataset ``kyoru4444/foodlensvn``.
It ships **pre-split** annotation files plus a flat per-variant image layout
(originals and augmented variants share the same split folder). This script
consumes those splits as-is rather than re-splitting from a flat pool.

Inputs (under <data-dir>/):
  annotations/{train,val,test}.json       (already augmented; sentence answers)
  images/{raw,squared}/{train,val,test}/  (originals + *_aug_N variants merged)

Outputs (under <output-dir>/):
  annotations/train.json
  annotations/val.json
  annotations/test.json
  answer_vocab.json
  build_dataset.log

Image path rewriting:
  raw rows carry only a bare filename (e.g. ``banh_canh_010.jpg`` or
  ``banh_canh_010_aug_2.jpg``). This script prefixes them with the variant
  and split subdirectory so VQADataset can resolve via
  ``images_root=<data-dir>/images``:

    "banh_canh_010.jpg"       -> "squared/<split>/banh_canh_010.jpg"
    "banh_canh_010_aug_2.jpg" -> "squared/<split>/banh_canh_010_aug_2.jpg"

  The variant (``squared`` vs ``raw``) is selected by ``--image-variant``;
  default is squared because the modular vision encoders expect square inputs.

Validation rules (raise on violation, except duplicates which are dropped with
a warning):
  * type in {yes_no, counting, recognition, attribute, spatial, reasoning}
  * dish in CANONICAL_DISHES (see src/utils/dishes.py)
  * answer length <= 10 words after normalization
  * counting answers must contain at least one digit after normalization
    (sentence form is allowed — `normalize_answer` maps number words to digits)
  * (image_id, question) pairs are unique within a split
  * train and test image_id sets are disjoint (sanity-check on pre-split inputs)

In ``--debug`` mode each split is capped to {train: 100, val: 20, test: 50} and
the corpus-size asserts are skipped.

Preference data schema (emitted as an empty stub by ``--build-preference`` at
``<data-dir>/preference/preference.json``):

    [
      {
        "image": "squared_splits/train/pho_001.jpg",
        "question": "Món này có cay không?",
        "chosen": "Món này không cay.",
        "rejected": "Có, món này rất cay."
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.answer_tokenizer import AnswerTokenizer
from src.utils.dishes import CANONICAL_DISHES_SET
from src.utils.vn_text import normalize_answer


VALID_TYPES: frozenset[str] = frozenset(
    {"yes_no", "counting", "recognition", "attribute", "spatial", "reasoning"}
)

ANSWER_TYPE_MAP: dict[str, str] = {
    "yes_no": "classification",
    "counting": "classification",
    "recognition": "classification",
    "attribute": "generative",
    "spatial": "generative",
    "reasoning": "generative",
}

DIFFICULTY_MAP: dict[str, str] = {
    "yes_no": "easy",
    "recognition": "easy",
    "counting": "medium",
    "attribute": "medium",
    "spatial": "hard",
    "reasoning": "hard",
}

DEBUG_CAPS: dict[str, int] = {"train": 100, "val": 20, "test": 50}

MIN_UNIQUE_IMAGES: int = 200
MIN_TRAIN_ROWS: int = 2000
MIN_TEST_ROWS: int = 50


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = (
        args.data_dir
        or os.environ.get("FOODLENS_DATA_DIR")
        or os.environ.get("KAGGLE_INPUT_DIR")
        or "data"
    )
    output_dir = args.output_dir or os.environ.get("KAGGLE_WORKING_DIR") or "data/processed"
    return Path(data_dir), Path(output_dir)


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "build_dataset.log"
    logger = logging.getLogger("build_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _load_split(data_dir: Path, split: str, logger: logging.Logger) -> list[dict]:
    path = data_dir / "annotations" / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing annotation file for split={split!r}: {path}")
    logger.info("loading %s split from %s", split, path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list, got {type(rows).__name__}")
    return rows


def _image_path(filename: str, split: str, variant: str) -> str:
    """Prefix a bare filename with ``<variant>/<split>/`` (Kaggle layout)."""
    return f"{variant}/{split}/{filename}"


def _process_split(
    raw: list[dict],
    split: str,
    variant: str,
    logger: logging.Logger,
) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    dropped_dup = 0
    out: list[dict] = []

    for idx, row in enumerate(raw):
        rid = row.get("id", f"{split}#{idx}")
        qtype = row.get("type")
        if qtype not in VALID_TYPES:
            raise ValueError(
                f"row {rid!r} has invalid type {qtype!r}; expected one of "
                f"{sorted(VALID_TYPES)}"
            )

        dish = row.get("dish")
        if dish not in CANONICAL_DISHES_SET:
            raise ValueError(
                f"row {rid!r} has non-canonical dish {dish!r}; expected one of "
                f"{sorted(CANONICAL_DISHES_SET)}"
            )

        image_filename = row.get("image")
        if not image_filename:
            raise ValueError(f"row {rid!r} is missing 'image'")
        if "/" in image_filename or "\\" in image_filename:
            raise ValueError(
                f"row {rid!r} 'image' must be a bare filename, got {image_filename!r}"
            )

        image_id = row.get("image_id") or Path(image_filename).stem
        if not image_id:
            raise ValueError(f"row {rid!r} missing image_id and could not derive one")

        question = (row.get("question") or "").strip()
        if not question:
            raise ValueError(f"row {rid!r} has empty question")

        answer_norm = normalize_answer(row.get("answer", ""))
        if not answer_norm:
            raise ValueError(f"row {rid!r} has empty answer after normalization")
        if len(answer_norm.split()) > 10:
            raise ValueError(
                f"row {rid!r} answer exceeds 10 words after normalization: {answer_norm!r}"
            )
        # No structural check on counting answers: Phase-1 generation produces
        # full sentences mixing digits ("hai" -> "2") and quantifier phrases
        # ("rất nhiều", "vài", "một ít") that don't map cleanly to integers.
        # Soft-counting accuracy is computed at metric time, not at build time.

        dup_key = (image_id, question)
        if dup_key in seen:
            dropped_dup += 1
            logger.warning(
                "[%s] dropping duplicate (image_id=%r, question=%r)", split, image_id, question
            )
            continue
        seen.add(dup_key)

        out.append(
            {
                "id": rid,
                "image": _image_path(image_filename, split, variant),
                "image_id": image_id,
                "dish": dish,
                "question": question,
                "answer": answer_norm,
                "type": qtype,
                "answer_type": ANSWER_TYPE_MAP[qtype],
                "difficulty": DIFFICULTY_MAP[qtype],
                "source": row.get("source", "unknown"),
            }
        )

    logger.info("[%s] processed %d rows; dropped %d duplicates", split, len(out), dropped_dup)
    return out


def _check_disjoint(splits: dict[str, list[dict]], logger: logging.Logger) -> None:
    img_by_split = {name: {r["image_id"] for r in rows} for name, rows in splits.items()}
    train_test = img_by_split["train"] & img_by_split["test"]
    train_val = img_by_split["train"] & img_by_split["val"]
    val_test = img_by_split["val"] & img_by_split["test"]
    if train_test:
        raise RuntimeError(
            f"image_id overlap between train and test (sample: {sorted(train_test)[:5]})"
        )
    if train_val:
        raise RuntimeError(
            f"image_id overlap between train and val (sample: {sorted(train_val)[:5]})"
        )
    if val_test:
        raise RuntimeError(
            f"image_id overlap between val and test (sample: {sorted(val_test)[:5]})"
        )
    logger.info("splits are image-disjoint")


def _save_splits(
    splits: dict[str, list[dict]], output_dir: Path, logger: logging.Logger
) -> None:
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = ann_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("wrote %d rows to %s", len(rows), path)


def _build_and_save_vocab(
    train_rows: list[dict], output_dir: Path, logger: logging.Logger
) -> None:
    answers = [r["answer"] for r in train_rows]
    tokenizer = AnswerTokenizer.build_from_corpus(answers)
    vocab_path = output_dir / "answer_vocab.json"
    tokenizer.save(vocab_path)
    logger.info("answer vocab size=%d saved to %s", tokenizer.vocab_size, vocab_path)


def _emit_preference_stub(data_dir: Path, logger: logging.Logger) -> None:
    pref_dir = data_dir / "preference"
    pref_dir.mkdir(parents=True, exist_ok=True)
    pref_path = pref_dir / "preference.json"
    if pref_path.exists():
        logger.info("preference file already exists at %s; leaving untouched", pref_path)
        return
    with pref_path.open("w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    logger.info("wrote empty preference stub to %s", pref_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VQA dataset splits with validation.")
    parser.add_argument("--data-dir", default=None, help="raw annotations root (default: $FOODLENS_DATA_DIR or $KAGGLE_INPUT_DIR or 'data')")
    parser.add_argument("--output-dir", default=None, help="processed output root (default: $KAGGLE_WORKING_DIR or 'data/processed')")
    parser.add_argument("--image-variant", choices=("squared", "raw"), default="squared", help="image folder variant to reference (default: squared)")
    parser.add_argument("--debug", action="store_true", help="cap split sizes and skip corpus-size asserts")
    parser.add_argument("--build-preference", action="store_true", help="emit empty preference stub")
    args = parser.parse_args()

    data_dir, output_dir = _resolve_dirs(args)
    logger = _setup_logging(output_dir)
    logger.info(
        "data_dir=%s output_dir=%s variant=%s debug=%s",
        data_dir, output_dir, args.image_variant, args.debug,
    )

    splits: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        raw = _load_split(data_dir, split, logger=logger)
        splits[split] = _process_split(raw, split, args.image_variant, logger)

    _check_disjoint(splits, logger)

    if args.debug:
        for name, cap in DEBUG_CAPS.items():
            if len(splits[name]) > cap:
                splits[name] = splits[name][:cap]
        logger.info(
            "debug mode: caps=%s actual={train: %d, val: %d, test: %d}",
            DEBUG_CAPS,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
    else:
        unique_images = len(
            {r["image_id"] for rows in splits.values() for r in rows}
        )
        if unique_images < MIN_UNIQUE_IMAGES:
            raise AssertionError(
                f"need >={MIN_UNIQUE_IMAGES} unique images, got {unique_images}"
            )
        if len(splits["train"]) < MIN_TRAIN_ROWS:
            raise AssertionError(
                f"train must have >={MIN_TRAIN_ROWS} rows, got {len(splits['train'])}"
            )
        if len(splits["test"]) < MIN_TEST_ROWS:
            raise AssertionError(
                f"test must have >={MIN_TEST_ROWS} rows, got {len(splits['test'])}"
            )

    logger.info(
        "split sizes: train=%d val=%d test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    _save_splits(splits, output_dir, logger)
    _build_and_save_vocab(splits["train"], output_dir, logger)

    if args.build_preference:
        _emit_preference_stub(data_dir, logger)

    logger.info("done.")


if __name__ == "__main__":
    main()
