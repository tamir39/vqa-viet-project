"""Build train/val/test splits for FoodLensVN with strict validation and answer vocab.

Reads raw annotations from <data-dir>/annotations/raw.json (preferred) or
<data-dir>/annotations/train.json (fallback) and writes:

  <output-dir>/annotations/train.json
  <output-dir>/annotations/val.json
  <output-dir>/annotations/test.json
  <output-dir>/answer_vocab.json
  <output-dir>/build_dataset.log

Default directories come from KAGGLE_INPUT_DIR / KAGGLE_WORKING_DIR env vars,
falling back to ``data/`` and ``data/processed/``. All directories are
auto-created. No internet access required.

Validation rules (raise on violation, except duplicates which are dropped with
a warning):
  * type in {yes_no, counting, recognition, attribute, spatial, reasoning}
  * dish in CANONICAL_DISHES (see src/utils/dishes.py)
  * answer length <= 10 words after normalization
  * counting answers must be digit strings after normalization
  * (image_id, question) pairs are unique
  * train and test image_id sets are disjoint

Splits are 80/10/10 at the image_id level, stratified per dish, deterministic
under SEED=42. In ``--debug`` mode, each split is capped to {train: 100, val:
20, test: 50} and the corpus-size asserts (>=200 unique images, >=2000 train,
>=50 test) are skipped.

Preference data schema (emitted as an empty stub by ``--build-preference`` at
``<data-dir>/preference/preference.json``):

    [
      {
        "image": "raw/images/pho_001.jpg",
        "question": "Món này có cay không?",
        "chosen": "không",
        "rejected": "có"
      }
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
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

SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
DEBUG_CAPS: dict[str, int] = {"train": 100, "val": 20, "test": 50}
SEED: int = 42

MIN_UNIQUE_IMAGES: int = 200
MIN_TRAIN_ROWS: int = 2000
MIN_TEST_ROWS: int = 50


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = args.data_dir or os.environ.get("KAGGLE_INPUT_DIR") or "data"
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


def _load_raw(data_dir: Path, logger: logging.Logger) -> list[dict]:
    candidates = [
        data_dir / "annotations" / "raw.json",
        data_dir / "annotations" / "train.json",
    ]
    for path in candidates:
        if path.exists():
            logger.info("loading raw annotations from %s", path)
            with path.open("r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError(f"{path} must contain a JSON list, got {type(rows).__name__}")
            return rows
    raise FileNotFoundError(
        "no raw annotations found; expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _ensure_image_id(row: dict) -> str:
    if row.get("image_id"):
        return str(row["image_id"])
    image = row.get("image", "")
    return Path(image).stem if image else ""


def _process_rows(raw: list[dict], logger: logging.Logger) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    dropped_dup = 0
    out: list[dict] = []

    for idx, row in enumerate(raw):
        rid = row.get("id", f"row#{idx}")
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

        image = row.get("image")
        if not image:
            raise ValueError(f"row {rid!r} is missing 'image'")

        image_id = _ensure_image_id(row)
        if not image_id:
            raise ValueError(f"row {rid!r} is missing image_id and could not derive one")

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
        if qtype == "counting" and not answer_norm.isdigit():
            raise ValueError(
                f"row {rid!r} counting answer must be digits only, got {answer_norm!r}"
            )

        dup_key = (image_id, question)
        if dup_key in seen:
            dropped_dup += 1
            logger.warning("dropping duplicate (image_id=%r, question=%r)", image_id, question)
            continue
        seen.add(dup_key)

        out.append(
            {
                "id": rid,
                "image": image,
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

    logger.info("processed %d rows; dropped %d duplicates", len(out), dropped_dup)
    return out


def _split_image_ids(image_ids: list[str], rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    images = sorted(image_ids)
    rng.shuffle(images)
    n = len(images)

    if n == 0:
        return [], [], []
    if n == 1:
        return images, [], []
    if n == 2:
        return images[:1], [], images[1:]

    n_test = max(1, round(n * SPLIT_RATIOS["test"]))
    n_val = max(1, round(n * SPLIT_RATIOS["val"]))
    n_train = n - n_test - n_val
    if n_train <= 0:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    train = images[:n_train]
    val = images[n_train : n_train + n_val]
    test = images[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def _split_by_image(
    rows: list[dict], logger: logging.Logger, debug: bool
) -> dict[str, list[dict]]:
    img_to_dish: dict[str, str] = {}
    img_to_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        img_to_rows[r["image_id"]].append(r)
        existing = img_to_dish.get(r["image_id"])
        if existing is not None and existing != r["dish"]:
            raise ValueError(
                f"image_id {r['image_id']!r} maps to multiple dishes: "
                f"{existing!r} and {r['dish']!r}"
            )
        img_to_dish[r["image_id"]] = r["dish"]

    by_dish: dict[str, list[str]] = defaultdict(list)
    for img_id, dish in img_to_dish.items():
        by_dish[dish].append(img_id)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for dish in sorted(by_dish):
        train_imgs, val_imgs, test_imgs = _split_image_ids(by_dish[dish], rng)
        for split_name, imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            for img_id in imgs:
                splits[split_name].extend(img_to_rows[img_id])

    train_imgs_set = {r["image_id"] for r in splits["train"]}
    test_imgs_set = {r["image_id"] for r in splits["test"]}
    overlap = train_imgs_set & test_imgs_set
    if overlap:
        raise RuntimeError(
            f"image_id overlap between train and test (sample: {sorted(overlap)[:5]})"
        )

    for name in splits:
        rng.shuffle(splits[name])

    if debug:
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
        unique_images = len({r["image_id"] for r in rows})
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
    return splits


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
    parser.add_argument("--data-dir", default=None, help="raw annotations root (default: $KAGGLE_INPUT_DIR or 'data')")
    parser.add_argument("--output-dir", default=None, help="processed output root (default: $KAGGLE_WORKING_DIR or 'data/processed')")
    parser.add_argument("--debug", action="store_true", help="cap split sizes and skip corpus-size asserts")
    parser.add_argument("--build-preference", action="store_true", help="emit empty preference stub")
    args = parser.parse_args()

    data_dir, output_dir = _resolve_dirs(args)
    logger = _setup_logging(output_dir)
    logger.info("data_dir=%s output_dir=%s debug=%s", data_dir, output_dir, args.debug)

    raw = _load_raw(data_dir, logger)
    rows = _process_rows(raw, logger)
    splits = _split_by_image(rows, logger, debug=args.debug)
    _save_splits(splits, output_dir, logger)
    _build_and_save_vocab(splits["train"], output_dir, logger)

    if args.build_preference:
        _emit_preference_stub(data_dir, logger)

    logger.info("done.")


if __name__ == "__main__":
    main()
