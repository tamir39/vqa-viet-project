# FoodLensVN — PLANNING

> Day-to-day map for engineers and AI agents working in this repo. Pairs with `PRD.md` (the *what*).

---

## 1. Architecture summary

FoodLensVN is a **research codebase**, not a service. Two pipelines share a common data layer, canonicalization, and metric suite — they only diverge below the trainer boundary.

```
                         ┌────────────────────────────────────┐
                         │  Raw annotations + images          │
                         │  data/annotations/train.json       │
                         │  data/raw/images/*.jpg             │
                         └───────────────┬────────────────────┘
                                         │
                            scripts/build_dataset.py
                            (validate · canonicalize · split · vocab)
                                         │
                         ┌───────────────┴────────────────────┐
                         ▼                                    ▼
       data/processed/annotations/{train,val,test}.json   answer_vocab.json
                         │                                    │
            ┌────────────┼────────────────────┐               │
            ▼            ▼                    ▼               │
     ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
     │ Modular A1   │ │ Modular A2   │ │ Multimodal B1/B2 │   │
     │              │ │              │ │                  │   │
     │ image_enc    │ │ image_enc    │ │ Qwen2-VL-2B      │   │
     │   (timm)     │ │   (timm)     │ │  (4-bit NF4)     │   │
     │ text_enc     │ │ text_enc     │ │                  │   │
     │   (PhoBERT)  │ │   (PhoBERT)  │ │ B1: zero-shot    │   │
     │ fusion       │ │ fusion       │ │ B2: LoRA SFT     │   │
     │   (co-attn)  │ │   (co-attn)  │ │   via TRL        │   │
     │ LSTM dec ───►│ │ Transformer  │ │                  │   │
     │              │ │     dec ────►│ │                  │   │
     └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘   │
            └────────────────┴───────┬──────────┘             │
                                     ▼                        │
                         ┌──────────────────────────┐         │
                         │  src/utils/metrics/      │◄────────┘
                         │  VQA-Acc · BLEU · ROUGE  │
                         │  METEOR · BERTScore      │
                         │  LLM-judge               │
                         └──────────────┬───────────┘
                                        ▼
                         reports/<config>_metrics.json
                         reports/<config>_errors.json
                                        │
                                        ▼
                              app/demo.py (Gradio)
```

### Core principles (binding)

1. **Single canonicalization function.** `normalize_answer` is applied at dataset build time, at training-label tokenization, and at inference output cleaning. No second-guessing it elsewhere.
2. **Image-disjoint splits.** Splits are computed at the `image_id` level, never at the row level — otherwise the same image leaks across train/test.
3. **Separate answer vocabulary.** `AnswerTokenizer` is the answer-side vocab for modular decoders. PhoBERT remains question-side only. Do not blur this boundary.
4. **Configs inherit from `base_config.yaml`.** Each of A1/A2/B1/B2 only overrides what changes.
5. **Kaggle envelope is binding.** Every default must fit ≤16 GB GPU and ≤9-hour runtime. If a feature requires more, gate it behind a flag and document it.
6. **No hardcoded absolute paths.** All scripts go through `--data-dir` / `--output-dir` flags or `KAGGLE_INPUT_DIR` / `KAGGLE_WORKING_DIR` env vars.

---

## 2. Module catalog

| Module | Responsibility | Status |
|--------|----------------|--------|
| `src/utils/vn_text.py` | Single source of truth for `normalize_answer` | ✅ Implemented |
| `src/data_loader/answer_tokenizer.py` | Answer-side vocab for modular decoders | ✅ Implemented |
| `src/models/multimodal/qwen_vl.py` | Qwen2-VL loader, prompt builder, output cleaner | ✅ Implemented |
| `scripts/build_dataset.py` | Validate · canonicalize · split · build vocab | ✅ Implemented |
| `scripts/check_env.py` | GPU + library + config sanity check | ✅ Implemented |
| `src/data_loader/vqa_dataset.py` | Torch `Dataset` for VQA rows + image loading | ⏳ Planned |
| `src/data_loader/collate.py` | Batched padding for modular pipeline | ⏳ Planned |
| `src/data_loader/augment.py` | Image + text augmentation (no back-translation) | ⏳ Planned |
| `src/models/encoders/image_encoder.py` | ResNet/ViT/EfficientNet via `timm` | ⏳ Planned |
| `src/models/encoders/text_encoder.py` | PhoBERT-base wrapper | ⏳ Planned |
| `src/models/fusion/cross_attention.py` | Co-attention fusion (primary) | ⏳ Planned |
| `src/models/decoders/lstm_decoder.py` | LSTM decoder for A1 | ⏳ Planned |
| `src/models/decoders/transformer_decoder.py` | Transformer decoder for A2 | ⏳ Planned |
| `src/trainer/modular_trainer.py` | A1/A2 training loop with AMP + ckpt | ⏳ Planned |
| `src/trainer/peft_trainer.py` | B2 LoRA SFT via TRL | ⏳ Planned |
| `src/utils/metrics/` | VQA-Acc, BLEU/ROUGE/METEOR, BERTScore, LLM-judge | ⏳ Planned |
| `scripts/train.py` | Unified entry-point (`--config <yaml>`) | ⏳ Planned |
| `scripts/eval.py` | Eval + error-analysis output | ⏳ Planned |
| `scripts/infer.py` | Single-example inference CLI | ⏳ Planned |
| `app/demo.py` | Gradio demo serving all four configs | ⏳ Planned |

---

## 3. Folder structure (binding)

```
FoodLensVN/
├── app/                           # Gradio demo
│   └── demo.py
├── configs/
│   ├── base_config.yaml           # shared defaults
│   ├── A1.yaml                    # modular + LSTM
│   ├── A2.yaml                    # modular + Transformer
│   ├── B1.yaml                    # Qwen2-VL zero-shot
│   └── B2.yaml                    # Qwen2-VL LoRA SFT
├── data/
│   ├── annotations/               # split-baked Q/A files: {train,val,test}_with_questions[_augmented].json (tracked)
│   ├── images/                    # squared/raw + per-split + per-split-augmented (gitignored; download from Kaggle)
│   ├── processed/                 # build_dataset.py output (gitignored)
│   └── preference/                # DPO pairs (gitignored except stub)
├── notebooks/
│   └── kaggle_template.ipynb
├── reports/                       # checkpoints / logs / metrics / errors
├── scripts/
│   ├── check_env.py
│   ├── build_dataset.py
│   ├── train.py                   # planned
│   ├── eval.py                    # planned
│   └── infer.py                   # planned
├── src/
│   ├── data_loader/
│   │   ├── answer_tokenizer.py
│   │   ├── vqa_dataset.py         # planned
│   │   ├── collate.py             # planned
│   │   └── augment.py             # planned
│   ├── models/
│   │   ├── encoders/              # image_encoder.py · text_encoder.py
│   │   ├── fusion/                # cross_attention.py
│   │   ├── decoders/              # lstm_decoder.py · transformer_decoder.py
│   │   └── multimodal/
│   │       └── qwen_vl.py
│   ├── trainer/                   # modular_trainer.py · peft_trainer.py
│   └── utils/
│       ├── vn_text.py
│       └── metrics/               # planned
├── tests/                         # planned
├── main.py                        # stub entry-point
├── pyproject.toml
├── uv.lock
└── DLEndterm.docx                 # course brief (gitignored)
```

> **Rule:** new code extends this layout. Do not flatten directories or move files between top-level groups without a PRD update.

---

## 4. Data contracts

### 4.1 Annotation row

```json
{
  "id": "vfvqa-000001",
  "image": "raw/images/pho_001.jpg",
  "image_id": "pho_001",
  "dish": "pho",
  "question": "Món này có cay không?",
  "answer": "không",
  "type": "yes_no",
  "answer_type": "classification",
  "difficulty": "easy",
  "source": "scrape"
}
```

- `type ∈ {yes_no, counting, recognition, attribute, spatial, reasoning}`
- `dish ∈ CANONICAL_DISHES` — the locked 20-dish set; see §4.5. New dishes require a PRD update.
- `answer_type` is auto-derived: `yes_no | counting | recognition → classification`; `attribute | spatial | reasoning → generative`.
- `difficulty` is auto-derived: `yes_no | recognition → easy`; `counting | attribute → medium`; `spatial | reasoning → hard`.
- `source ∈ {scrape, dataset, self_shot}` for license tracking.

> **Answer style note (Phase 1).** Answers are full Vietnamese sentences (≈7–9 words), generated by the Phase-1 sample generator (`data/code/sample_generators/gen.py`). The pipeline is therefore **fully generative**, including for `yes_no`, `counting`, and `recognition` — short-label classification is no longer a special case. `AnswerTokenizer` learns its vocabulary from this corpus.

### 4.2 Validation rules (enforced by `build_dataset.py`)

- `type` is in the 6-value enum.
- `len(normalize_answer(answer).split()) ≤ 10`.
- No structural check on counting answers. Phase-1 generation produces sentences mixing digit-mappable number words (`hai`→`2`) and quantifier phrases (`rất nhiều`, `vài`, `một ít`). Soft-counting metrics handle scoring at eval time; the build-time validator only enforces ≤10 words and non-empty.
- `(image_id, question)` is unique within a split (duplicates dropped with warning, count logged).
- Train and test `image_id` sets are disjoint.

In non-debug mode also: ≥200 unique images, ≥2000 train rows, ≥50 test rows.

### 4.3 Answer canonicalization (`normalize_answer`)

In order:
1. Lowercase, strip, collapse internal whitespace.
2. Strip trailing punctuation `[.,!?…:;]`.
3. Yes/no whole-string match (early return): `co | có | yes → "có"`; `khong | không | no → "không"`.
4. Per-token Vietnamese-number-word → digit (0–10, both diacritic and bare forms; `bon | bốn | tu | tư → 4`).
5. Strip trailing classifier suffix (`cái | miếng | phần | tô | bát`).

### 4.5 Canonical dish set (locked)

Single source of truth: [`src/utils/dishes.py`](src/utils/dishes.py) (`CANONICAL_DISHES`). All raw rows must use these `dish` keys; `build_dataset.py` rejects unknown values. The set was expanded from 10 → 20 in Phase 1 to match the scraped corpus.

| `dish` key | Display | Region | Role |
|------------|---------|--------|------|
| `pho` | Phở | North | Iconic noodle soup |
| `bun_bo_hue` | Bún bò Huế | Central | Spicy beef noodle soup |
| `banh_mi` | Bánh mì | Pan-regional | Sandwich |
| `com_tam` | Cơm tấm | South | Broken rice plate |
| `bun_cha` | Bún chả | North (Hanoi) | Grilled pork + noodles |
| `goi_cuon` | Gỏi cuốn | South | Fresh spring rolls |
| `cha_gio` | Chả giò / nem rán | South | Fried spring rolls |
| `banh_xeo` | Bánh xèo | Central / South | Sizzling crepe |
| `mi_quang` | Mì Quảng | Central | Turmeric noodles, dry-style |
| `hu_tieu` | Hủ tiếu | South | Noodle soup (pho disambiguation target) |
| `banh_cuon` | Bánh cuốn | North | Steamed rice rolls |
| `bun_thit_nuong` | Bún thịt nướng | South | Grilled pork over vermicelli |
| `cao_lau` | Cao lầu | Central (Hội An) | Regional noodle dish |
| `bot_chien` | Bột chiên | South | Fried rice-flour cake |
| `banh_khot` | Bánh khọt | South | Mini savoury pancakes |
| `xoi_xeo` | Xôi xéo | North | Sticky rice with mung bean |
| `chao_long` | Cháo lòng | Pan-regional | Offal congee |
| `bun_dau_mam_tom` | Bún đậu mắm tôm | North | Vermicelli + tofu + shrimp paste |
| `bun_mam` | Bún mắm | South (Mekong) | Fermented-fish noodle soup |
| `banh_canh` | Bánh canh | Central / South | Thick tapioca noodle soup |

### 4.6 Preference row (DPO/PPO bonus)

```json
{
  "id": "vfpref-000001",
  "image": "raw/images/pho_001.jpg",
  "question": "Món này có cay không?",
  "chosen": "không",
  "rejected": "có cay nhiều"
}
```

---

## 5. Training contracts

### 5.1 Modular (A1/A2)

- Image encoder: `timm` model (default ResNet50; ablation: ViT-S/16) producing `(B, P, D_img)` patch features.
- Text encoder: PhoBERT-base producing `(B, T, D_txt)` token features.
- Fusion: co-attention → `(B, D_fused)` (primary); element-wise / concat (ablations).
- Decoder: LSTM (A1) or Transformer (A2) over `AnswerTokenizer` vocab; teacher-forced cross-entropy at train, greedy decode at inference (`MAX_LEN = 12`).
- Loss: token-level CE with `ignore_index = PAD_ID (0)`.
- Optimizer: AdamW; warmup → cosine decay; AMP (`bf16`/`fp16`) when CUDA.

### 5.2 Multimodal (B1/B2)

- Base model: `Qwen/Qwen2-VL-2B-Instruct`. 4-bit NF4 by default; bf16 fallback for non-quantized.
- B1: zero-shot. No training. Inference only.
- B2: LoRA SFT via TRL `SFTTrainer`. LoRA on attention proj layers only; targets defined in `B2.yaml`.
- Strict Vietnamese system prompt (already in `qwen_vl.py`); SFT format reuses the same chat template.
- Generation: `max_new_tokens=20`, `do_sample=False`. Output post-processed via `_clean_output` → `normalize_answer`.

### 5.3 Evaluation suite

| Metric | Implementation | Notes |
|--------|----------------|-------|
| VQA Accuracy (exact) | Exact-match against canonical gold | After `normalize_answer` on both sides. |
| VQA Accuracy (soft) | Token-overlap with thresholds | Type-aware (yes_no requires exact). |
| BLEU | `sacrebleu` | BLEU-1..4. |
| ROUGE-L | `rouge-score` | F1. |
| METEOR | `nltk.translate.meteor_score` | With WordNet fallback or token-level. |
| BERTScore | `bert-score` with `xlm-roberta-base` | Vietnamese-friendly. |
| LLM-judge | Local model (Qwen2-VL or other) prompting a yes/no equivalence call | Offline only; no paid APIs. |

All metrics report **per-type** and **per-difficulty** breakdowns. Errors written to `reports/<config>_errors.json`.

---

## 6. Configuration

`configs/base_config.yaml` holds shared defaults (paths, seed, batch size, image resolution, epoch count, scheduler). Each of `A1.yaml`, `A2.yaml`, `B1.yaml`, `B2.yaml` overrides only what changes (e.g., decoder type, LoRA targets, model id).

Loading rule: `train.py` and `eval.py` deep-merge `base_config.yaml` ← config-specific yaml. Do not duplicate base values.

---

## 7. Kaggle workflow

- `notebooks/kaggle_template.ipynb` clones the repo, runs `uv sync`, sets `KAGGLE_NO_INTERNET=1` if applicable, and verifies GPU.
- Pre-stage HF models (`Qwen/Qwen2-VL-2B-Instruct`, `vinai/phobert-base`, `xlm-roberta-base`) into a Kaggle dataset for offline mode.
- Outputs go to `KAGGLE_WORKING_DIR=/kaggle/working`. Inputs come from `KAGGLE_INPUT_DIR=/kaggle/input/foodlensvn`.

---

## 8. Decision log

- **Qwen2-VL-2B-Instruct over LLaVA / InstructBLIP** — Vietnamese performance is materially better and the 2B size fits the Kaggle envelope with NF4.
- **PhoBERT over multilingual BERT** — domain match (Vietnamese) and the project explicitly compares with/without it in the modular ablation.
- **Separate `AnswerTokenizer`** — the answer space is small and structured (yes/no, digits, dish names, attributes); a domain vocab gives a much smaller decoder output and faster convergence than reusing PhoBERT's 64k vocab.
- **Image-disjoint splits at `image_id` level** — naive row-level splitting leaks the same image across train/test and inflates accuracy ~10–20 points.
- **No back-translation augmentation** — explicitly excluded by the course brief because it can corrupt the canonical answer form.
- **`uv` over pip / poetry** — faster, deterministic, and locks the Python version (≥3.14 in this repo).

---

## Invariants (DO NOT VIOLATE)

- `PRD.md` is the source of truth for requirements. `DLEndterm.docx` is the source of truth for the course contract.
- Do not change folder structure without a PRD update.
- Do not bypass `normalize_answer` for any answer string anywhere in the pipeline.
- Do not split at the row level — always at `image_id`.
- Do not push `*.docx` or per-repo `CLAUDE.md` (gitignored).
- `main` branch is touched only on shipping; daily work goes to feature branches off `develop`.
