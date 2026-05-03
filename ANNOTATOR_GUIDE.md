# FoodLensVN — Annotator Guide

One-page reference for collecting `(image, question, answer)` rows. The build pipeline (`scripts/build_dataset.py`) **rejects** rows that violate any rule below — please skim this once before you start, then keep it open as a cheat sheet.

---

## What we're building

A Vietnamese Food VQA dataset: an image of a Vietnamese dish + a Vietnamese question → a short Vietnamese answer (≤10 words). Target: **≥2000 rows · ≥200 unique images · ≥10 dishes · ≥50 hand-curated test rows**.

Locked dish set (use these `dish` keys exactly): `pho`, `bun_bo_hue`, `banh_mi`, `com_tam`, `bun_cha`, `goi_cuon`, `cha_gio`, `banh_xeo`, `mi_quang`, `hu_tieu`.

---

## Image rules

- **Source must be license-tag-able**: `scrape` (CC / Wikimedia / Unsplash / explicitly free), `dataset` (30VNFoods etc.), or `self_shot` (you took it). Do **not** include images you can't redistribute.
- **One dish per image.** If the photo shows two dishes, pick the dominant one or skip.
- **Visible enough to answer**: ingredients should be recognizable. Reject blurry, heavily filtered, or extreme close-up shots.
- Resize so max edge ≤ 1024 px. File name = `image_id` (e.g., `pho_001.jpg`); store under `data/raw/images/`.
- Aim for **≥20 unique images per dish** with variety: different angles, plating styles, regional variants.

---

## Question style

- **Vietnamese with diacritics.** No abbreviations. End with `?`.
- **Specific.** "Có bao nhiêu lát ớt?" ✓. "Mô tả món này?" ✗ (too open).
- **One answer.** If two correct answers exist, narrow the question.
- **Grounded in the image.** Don't ask things only a recipe book would know.

---

## The 6 question types

Each row's `type` field must be one of these. Aim for rough balance (no type < 10% of corpus).

| `type` | What to ask | Example Q | Example A |
|--------|-------------|-----------|-----------|
| `yes_no` | Yes/No checks on visible features | `Món này có cay không?` | `không` |
| `counting` | Count visible items | `Có bao nhiêu lát ớt?` | `2` |
| `recognition` | "What dish is this?" | `Đây là món gì?` | `phở` |
| `attribute` | Color / texture / ingredient | `Nước dùng có màu gì?` | `nâu` |
| `spatial` | Where something is in the frame | `Hành nằm ở phía nào của tô?` | `trên` |
| `reasoning` | Inference (region, meal context, pairing) | `Món này có nguồn gốc từ vùng nào?` | `huế` |

---

## Answer rules (the strict ones)

The build pipeline runs every answer through `normalize_answer` and validates the result. Write answers in their **canonical form** below to save round-trips.

1. **≤10 words** after normalization. Counted by whitespace split.
2. **Lowercase** is fine — normalization will lowercase anyway.
3. **Yes/No** → exactly `có` or `không`. Not `dạ có`, `vâng`, `đúng`, `sai`.
4. **Counting** → digit only. The pipeline auto-converts:
   - `hai` → `2`, `ba` → `3`, `mười` → `10`
   - `2 cái` → `2`, `hai miếng` → `2`
   - But just write `2` and skip the conversion.
5. **Classifiers** (`cái`, `miếng`, `phần`, `tô`, `bát`) are stripped after digits. Don't lean on them.
6. **No trailing punctuation** (`.`, `…`, `!`, etc.).
7. **Counting type** must produce a **digit** after normalization. `"khoảng 3"` is invalid; pick a definite count or skip the question.

### Canonicalization quick check

| You write | Stored as | Notes |
|-----------|-----------|-------|
| `Có` / `CÓ.` / `co` | `có` | Yes/no normalized |
| `Không` / `khong` | `không` | Yes/no normalized |
| `Hai cái` | `2` | Number word + classifier |
| `mười miếng` | `10` | Idem |
| `Phở bò` | `phở bò` | Free-form, just lowercased |
| `khoảng 3` | (rejected if counting) | Use `3` or rephrase |

---

## Splits & test set

- Splits are computed by the pipeline at the **`image_id` level** (80/10/10, stratified by `dish`). Don't pre-split manually.
- The **test set is hand-curated** (≥50 rows). Mark a small image pool as test-only by giving them `image_id`s that the pipeline puts in test (image-disjoint from train). In practice: tag rows internally during annotation; the engineer running `build_dataset.py` will materialize the split.
- The same `image_id` can appear in multiple rows (different questions over one image). That's expected.

---

## Common pitfalls (read this twice)

- ❌ Asking about taste you can't see (`Món này có ngon không?`).
- ❌ Two-part questions joined by `và` / `hoặc`.
- ❌ Answers like `nhiều`, `vài`, `khoảng 3` for counting.
- ❌ Answers in English or with English brand names.
- ❌ Reusing the exact same `(image_id, question)` pair — the pipeline drops duplicates with a warning.
- ❌ Spelling `bun_bo` instead of `bun_bo_hue` for the `dish` field. Use the locked keys.

---

## Row schema (for the engineer importing your sheet)

```json
{
  "id": "vfvqa-000001",
  "image": "raw/images/pho_001.jpg",
  "image_id": "pho_001",
  "dish": "pho",
  "question": "Món này có cay không?",
  "answer": "không",
  "type": "yes_no",
  "source": "scrape"
}
```

`answer_type` and `difficulty` are auto-filled — leave them out.

---

**When in doubt, skip the row.** A smaller, cleaner dataset trains better than a larger noisy one.
