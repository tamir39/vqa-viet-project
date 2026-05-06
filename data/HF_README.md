---
language:
  - vi
license: cc-by-sa-4.0
task_categories:
  - visual-question-answering
  - image-to-text
pretty_name: FoodLensVN — Vietnamese Food VQA
size_categories:
  - 1K<n<10K
tags:
  - vietnamese
  - vqa
  - food
  - multimodal
---

# FoodLensVN — Vietnamese Food VQA

A small but clean Vietnamese-language Visual Question Answering dataset over **20 canonical Vietnamese dishes**. Built as the Phase-1 corpus for the [FoodLensVN](https://github.com/tamir39/vqa-viet-project) project (final-year Deep Learning report).

## Contents

- **5,572 (image, question, answer) rows** — `4,460 train / 632 val / 480 test`.
- **299 unique source images** + 897 augmented variants (3× per source) → **1,196 image refs per variant**.
- Two image variants: `raw/` (original aspect ratio, max edge ≤ 1024) and `squared/` (center-padded to a square — recommended for training).
- Splits are **image-disjoint** at the `image_id` level, stratified by `dish`.
- Answers are full Vietnamese sentences (≈7–9 words, ≤10 after canonicalization).

### Layout

```
foodlensvn/
├─ annotations/
│  ├─ train.json   # 4460 rows
│  ├─ val.json     # 632 rows
│  └─ test.json    # 480 rows
└─ images/
   ├─ raw/{train,val,test}/<file>.jpg
   └─ squared/{train,val,test}/<file>.jpg
```

### Row schema

```json
{
  "id": "vfvqa-000001",
  "image": "squared/train/pho_001.jpg",
  "image_id": "pho_001",
  "dish": "pho",
  "question": "Món này có cay không?",
  "answer": "Không, món này không cay.",
  "type": "yes_no",
  "source": "scrape"
}
```

### Locked dish set (20)

`pho`, `bun_bo_hue`, `banh_mi`, `com_tam`, `bun_cha`, `goi_cuon`, `cha_gio`, `banh_xeo`, `mi_quang`, `hu_tieu`, `banh_cuon`, `bun_thit_nuong`, `cao_lau`, `bot_chien`, `banh_khot`, `xoi_xeo`, `chao_long`, `bun_dau_mam_tom`, `bun_mam`, `banh_canh`.

### Question types (6)

`yes_no`, `counting`, `recognition`, `attribute`, `spatial`, `reasoning` — roughly balanced per split. See [ANNOTATOR_GUIDE.md](https://github.com/tamir39/vqa-viet-project/blob/main/ANNOTATOR_GUIDE.md) in the repo for details.

## Loading

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Tamir39/foodlensvn",
    repo_type="dataset",
    local_dir="data/foodlensvn",
)
# Then point scripts/build_dataset.py at <local_dir>.
```

On Kaggle, set the `HF_TOKEN` notebook secret (read-scoped is enough for this public dataset, but a token avoids rate limits) and call `snapshot_download` from inside the notebook.

## Construction

- **Images**: scraped from Wikimedia / Unsplash / explicit-CC sources, then squared to a fixed canvas. Augmented copies (3× per source) generated with light geometric + color jitter to enlarge training distribution without leaking new content into val/test.
- **Q/A**: drafted with Gemini 3.1 Flash Lite over each image, filtered through a strict Vietnamese validator (`normalize_answer`: yes/no canonicalization, number-word→digit, classifier strip, ≤10 words, no English).
- Splits computed at the `image_id` level (not row level) so the same image never crosses splits.

## License & attribution

- Released under **CC-BY-SA-4.0**.
- Underlying images come from public CC sources; redistribution is preserved per the `source` tag on each row.
- If you use this dataset, please cite the FoodLensVN project: https://github.com/tamir39/vqa-viet-project

## Limitations

- 20 dishes only — does not cover the long tail of Vietnamese cuisine.
- Single-annotator pipeline (Gemini-generated, human-validated) → no inter-annotator agreement signal.
- Sentence-style answers; counting questions mix digits and quantifiers (`vài`, `rất nhiều`) — soft-counting accuracy belongs at eval time, not validation time.
- Scraped images: licensing is best-effort. If you spot a wrongly-tagged source, please open an issue on the repo.
