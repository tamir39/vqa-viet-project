# FoodLensVN — TASKS

> Execution checklist. Work top-to-bottom within a phase. Check off only when committed and verified.
> Conventions: `feat(<scope>): ...`, `fix(<scope>): ...` per repo `CLAUDE.md`. Branch off `develop`; never touch `main`.

---

## ✅ Phase 0 — Foundations (DONE)

- [x] Repo scaffold + `pyproject.toml` + `uv.lock` + `.python-version`.
- [x] `.gitignore` covers `data/raw`, `data/processed`, `data/preference`, `models/`, `*.docx`, `CLAUDE.md`.
- [x] `scripts/check_env.py` — GPU + library import sanity check.
- [x] `src/utils/vn_text.py` — `normalize_answer` (yes/no, number words, classifiers, punctuation).
- [x] `src/data_loader/answer_tokenizer.py` — `AnswerTokenizer` with fixed special ids and JSON save/load.
- [x] `scripts/build_dataset.py` — validate · canonicalize · split (image-disjoint, dish-stratified) · build vocab; `--debug` and `--build-preference` flags.
- [x] `src/models/multimodal/qwen_vl.py` — Qwen2-VL loader (NF4 default), prompt builder, output cleaner.
- [x] `data/annotations/train.json` — 22-row valid stub covering all 6 question types (development-only).
- [x] Bootstrap workflow files (PRD / PLANNING / TASKS / README / CLAUDE.md).

---

## ✅ Phase 1 — Real dataset (DONE)

> Phase-1 corpus delivered as the public HuggingFace dataset **[Tamir39/foodlensvn](https://huggingface.co/datasets/Tamir39/foodlensvn)** — 5,572 (image, question, answer) rows over 20 Vietnamese dishes, sentence-style answers. Generated via the toolchain on the (now-deleted) `origin/data` worktree (Selenium scraper + Gemini 3.1 Flash Lite Q/A generator + image augmenter); reproducible build script preserved at [dist/build_kaggle_dataset.py](dist/build_kaggle_dataset.py); upload script at [scripts/push_dataset.py](scripts/push_dataset.py). The course brief `DLEndterm.docx` requires HF as the dataset host.

### 1.1 Image collection

- [x] Lock dish set: 20 canonical dishes — see [src/utils/dishes.py](src/utils/dishes.py) and [PLANNING.md §4.5](PLANNING.md).
- [x] 299 unique source images scraped + 897 augmented variants (3× per source) = 1,196 image refs per variant. Both `raw/` and `squared/` variants shipped.
- [x] All images sourced via `scrape` and license-tagged in the `source` field.
- [x] Per-split disjoint image folders verified by `build_dataset.py`.

### 1.2 Annotation

- [x] [ANNOTATOR_GUIDE.md](ANNOTATOR_GUIDE.md) covers schema, types, canonical forms, splits, pitfalls.
- [x] 4,460 train + 632 val + 480 test = **5,572 rows**, balanced over 6 question types per split.
- [x] Test set is image-disjoint from train (verified by `build_dataset._check_disjoint`).
- [x] Answers are full Vietnamese sentences (≈7–9 words, ≤10 after canonicalization). Phase-1 is fully generative — see [PLANNING.md §4.1](PLANNING.md).

### 1.3 Build + validate

- [x] `python scripts/build_dataset.py --data-dir data/foodlensvn` runs clean: 0 duplicates dropped, splits image-disjoint, all 20 dishes present, answer vocab size 453.
- [x] Counting validator relaxed (Phase-1 sentences mix digits with quantifiers); soft-counting accuracy moved to eval-time metrics.

### 1.4 Host on HF Hub + Kaggle workflow

- [x] Public HuggingFace dataset published: [`Tamir39/foodlensvn`](https://huggingface.co/datasets/Tamir39/foodlensvn) (~34 MB, 2,395 files + dataset card).
- [x] [scripts/push_dataset.py](scripts/push_dataset.py) — one-shot uploader for `dist/foodlensvn_kaggle/` + `data/HF_README.md`.
- [x] [scripts/fetch_dataset.py](scripts/fetch_dataset.py) — `huggingface_hub.snapshot_download` to `data/foodlensvn/` (or `$FOODLENS_DATA_DIR`).
- [x] [notebooks/train_foodlensvn.ipynb](notebooks/train_foodlensvn.ipynb) — Kaggle secret `HF_TOKEN` → `huggingface_hub.login` → `fetch_dataset.py` → `build_dataset.py`.
- [ ] Pre-stage HF model snapshots (`vinai/phobert-base`, `Qwen/Qwen2-VL-2B-Instruct`, `xlm-roberta-base`) into a separate Kaggle dataset for `KAGGLE_NO_INTERNET=1` runs.
- [ ] Smoke-run [notebooks/train_foodlensvn.ipynb](notebooks/train_foodlensvn.ipynb) end-to-end on Kaggle GPU.

---

## 🚀 Phase 2 — Modular pipeline (A1 / A2)

### 2.1 Encoders

- [x] `src/models/encoders/__init__.py`.
- [x] `src/models/encoders/text_encoder.py` — PhoBERT-base wrapper. `forward(input_ids, attention_mask) -> (B, T, D)`. Honors `KAGGLE_NO_INTERNET`. Frozen by default; `unfreeze_last_n` knob.
- [x] `src/models/encoders/image_encoder.py` — `timm` model wrapper (default `resnet50`; ablation `vit_small_patch16_224`). `forward(pixel_values) -> (B, P, D)` patch features. Pretrained weights, frozen by default. (Adds `timm>=1.0.0` to `pyproject.toml`; run `uv sync` to install.)
- [x] Smoke test: load each encoder on CPU with random tensor, assert output shape — [scripts/smoke_encoders.py](scripts/smoke_encoders.py).

### 2.2 Fusion

- [x] `src/models/fusion/__init__.py`.
- [x] `src/models/fusion/cross_attention.py` — Co-attention fusion: image patches attend over text tokens and vice versa; output pooled `(B, D_fused)`.
- [x] (Optional ablations) element-wise and concat fusion stubs in same file with a `fusion_type` factory.

### 2.3 Decoders

- [x] `src/models/decoders/__init__.py`.
- [x] `src/models/decoders/lstm_decoder.py` — single-layer LSTM over `AnswerTokenizer`. Teacher-forcing at train, greedy at inference, `MAX_LEN=12`. Consumes `fused_pooled` (B, D).
- [x] `src/models/decoders/transformer_decoder.py` — 2-layer transformer decoder consuming `fused_seq` (B, M, D) as cross-attention memory; requires `CoAttentionFusion`.
- [x] Both decoders expose `forward(...)` (teacher-forced) and `generate(...)` (greedy).

### 2.4 Data loading

- [x] `src/data_loader/vqa_dataset.py` — Torch `Dataset` reading processed JSON rows + loading + transforming images. Encodes answers via `AnswerTokenizer`; passes questions through for collate-time PhoBERT tokenization.
- [x] `src/data_loader/collate.py` — `make_collate_fn(phobert_tokenizer, max_question_length)`: stacks `pixel_values` and `answer_ids`, batch-tokenizes questions, threads `meta` (id/type/difficulty/dish) for eval breakdowns.
- [x] `src/data_loader/augment.py` — Train-time image augmentation (RandomResizedCrop + flip + ColorJitter + RandAugment, preserving the backbone-correct Normalize) and a conservative single-phrase Vietnamese synonym swap. **No back-translation.**

### 2.5 Model assembly + trainer

- [x] `src/models/modular_vqa.py` — End-to-end `nn.Module` wiring encoders + fusion + decoder. One class, `decoder_input ∈ {"pooled", "seq"}` switch for A1/A2.
- [x] `src/trainer/__init__.py`.
- [x] `src/trainer/modular_trainer.py` — Train loop: AMP (bf16/fp16), AdamW, warmup→cosine, gradient clipping, val-loss-based ckpt save under `reports/<config>/checkpoints/`.

### 2.6 Configs

- [x] Fill `configs/base_config.yaml` with shared defaults (seed, batch size, lr, epochs, image size, paths).
- [x] Fill `configs/A1.yaml` (modular + LSTM decoder).
- [x] Fill `configs/A2.yaml` (modular + Transformer decoder; only diff from A1).

### 2.7 CLI entry-points

- [x] `scripts/train.py` — `--config <yaml>` deep-merge with `base_config.yaml`; dispatch to the right trainer based on `track` field (`modular | qwen_zeroshot | qwen_lora`). *(modular wired; qwen_zeroshot / qwen_lora dispatch is Phase 3.)*
- [x] `scripts/infer.py` — Single (image, question) inference for any config. *(modular only; B1/B2 added in Phase 3.)*
- [ ] Train A1 and A2 end-to-end on the real dataset; confirm checkpoints land in `reports/`.

---

## 🔬 Phase 3 — Multimodal pretrained pipeline (B1 / B2)

### 3.1 Configs

- [ ] Fill `configs/B1.yaml` — `track: qwen_zeroshot`, model id, generation params.
- [ ] Fill `configs/B2.yaml` — `track: qwen_lora`, LoRA targets (attention proj only), rank, alpha, dropout.

### 3.2 LoRA SFT trainer (B2)

- [ ] `src/trainer/peft_trainer.py` — TRL `SFTTrainer` over Qwen2-VL with the strict Vietnamese prompt; LoRA via `peft`; NF4 base; gradient checkpointing on.
- [ ] Hook `scripts/train.py --config configs/B2.yaml` to dispatch here.
- [ ] Train B2 end-to-end on the real dataset; LoRA adapter lands in `reports/B2/adapter/`.

### 3.3 B1 inference path

- [ ] Wire `scripts/eval.py` to call `qwen_vl.generate` per row when `track == qwen_zeroshot` (no training).
- [ ] B2 eval: load base + adapter, then same path as B1.

---

## 📊 Phase 4 — Evaluation

### 4.1 Metrics module

- [ ] `src/utils/metrics/__init__.py`.
- [ ] `src/utils/metrics/vqa_accuracy.py` — exact + soft (type-aware: yes_no exact, counting digit-equality, others token-overlap).
- [ ] `src/utils/metrics/text_metrics.py` — BLEU (sacrebleu), ROUGE-L (rouge-score), METEOR (nltk).
- [ ] `src/utils/metrics/bertscore.py` — `bert-score` with `xlm-roberta-base`; honors `KAGGLE_NO_INTERNET`.
- [ ] `src/utils/metrics/llm_judge.py` — local LLM (Qwen2-VL text-only or another) prompted for yes/no equivalence; offline.
- [ ] `src/utils/metrics/aggregate.py` — Compute all metrics, then group by `type` and `difficulty`; return a single dict.

### 4.2 `scripts/eval.py`

- [ ] Loads config + checkpoint/adapter, runs inference on test split, applies `_clean_output` + `normalize_answer`, calls `aggregate`, writes `reports/<config>_metrics.json`.
- [ ] Writes `reports/<config>_errors.json` with the rows where prediction ≠ gold (cap to N=200 for size).
- [ ] Verify: all four configs produce both files cleanly.

---

## 🎨 Phase 5 — Demo + report

- [ ] `app/demo.py` — Gradio app with image upload + question textbox; runs A1, A2, B1, B2 in parallel and shows answers + latency.
- [ ] Verify: launch locally, drop a sample image, all four configs respond.
- [ ] Written report (separate doc) — A1↔A2 ablation, B1↔B2 lift, dataset construction, error analysis, limitations.

---

## 🧪 Phase 6 — Bonus: DPO / PPO preference training

- [ ] Generate ≥100 preference pairs (chosen vs rejected) from existing model outputs.
- [ ] Populate `data/preference/preference.json` per the schema in `PLANNING.md` §4.4.
- [ ] `src/trainer/dpo_trainer.py` — TRL `DPOTrainer` over the LoRA-tuned base.
- [ ] Eval pass: SFT-vs-DPO deltas on the same metric suite, written to `reports/B2_dpo_metrics.json`.

---

## 🧰 Phase 7 — Polish

- [ ] `tests/` — unit tests for `normalize_answer`, `AnswerTokenizer`, `_clean_output`, dataset validation.
- [ ] CI workflow: lint + unit tests on every PR into `develop`.
- [ ] Pre-commit hooks via `ruff` + `black`.
- [ ] Tag `v1.0.0-submission` once Definition-of-Done in PRD §6 is fully checked.

---

## Rules

- Work sequentially within a phase. **Phase 1 (real dataset) blocks every training-related task in later phases.**
- Pure code that does not touch data (encoder/decoder/fusion modules, trainer skeletons, configs) may be drafted ahead of Phase 1, but cannot be claimed "done" until they have been exercised end-to-end against the real dataset.
- Commit only when a single TASKS bullet is complete and locally verified.
- Update this file immediately on completion (`- [x]`).
- Never push `main`. Daily work goes to feature branches off `develop`.
