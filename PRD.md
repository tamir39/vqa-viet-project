# FoodLensVN — Product Requirements Document

> Source of truth for **what** FoodLensVN is, **who** it serves, and **what done looks like**.
> Architecture (the **how**) lives in `PLANNING.md`. The binding course contract is `DLEndterm.docx`.

---

## 1. Project Goals

### 1.1 Course deliverable (Phase 1)

FoodLensVN is the final-term submission for **BÀI 1 (7 điểm)** of the Deep Learning course. It is a Vietnamese-language Visual Question Answering system: given an image of a Vietnamese dish and a Vietnamese question, the system returns a short Vietnamese answer (≤10 words after canonicalization).

The deliverable must include **all four mandatory configurations**:

| ID | Track | Description |
|----|-------|-------------|
| **A1** | Modular | Image encoder + PhoBERT/BiLSTM text encoder + co-attention fusion + **LSTM** decoder |
| **A2** | Modular | Same as A1 with a **Transformer** decoder (A1↔A2 is the core ablation) |
| **B1** | Multimodal pretrained | **Qwen2-VL-2B-Instruct** zero-shot |
| **B2** | Multimodal pretrained | Qwen2-VL-2B-Instruct fine-tuned with LoRA/PEFT |

Plus: a Gradio demo, a written report, and an evaluation pass on the held-out test split using the metric suite in §2.

### 1.2 Bonus track

- **DPO / PPO preference training** (≥100 pairs) on the modular or LoRA-fine-tuned model, with a side-by-side comparison vs the SFT baseline.

### 1.3 Non-Goals

- Any language other than Vietnamese.
- Open-vocabulary food domains beyond Vietnamese cuisine.
- Real-time inference SLAs / production hosting.
- Datasets requiring paid APIs or licenses we cannot redistribute.
- Models above the Kaggle ≤16 GB GPU envelope without quantization.

---

## 2. Scope

| Area | Scope |
|------|-------|
| **Dataset** | ≥2000 train rows · ≥200 unique images · 80/10/10 split at `image_id` level, stratified by dish · ≥50 hand-curated test samples, image-disjoint from train · 6 question types (yes_no, counting, recognition, attribute, spatial, reasoning). |
| **Canonicalization** | One pure function (`normalize_answer`) shared by data pipeline, training labels, and inference output cleaning. |
| **Modular pipeline (A1/A2)** | Image encoder (ResNet/ViT via `timm`), PhoBERT text encoder, co-attention fusion, LSTM and Transformer decoders consuming a domain-specific `AnswerTokenizer`. |
| **Multimodal pipeline (B1/B2)** | Qwen2-VL-2B-Instruct loader (4-bit NF4 default for ≤16 GB GPU), strict Vietnamese system prompt, chat-template inference, LoRA SFT trainer for B2. |
| **Evaluation** | VQA Accuracy (exact + soft), BLEU, ROUGE-L, METEOR, BERTScore (xlm-roberta-base), offline LLM-as-judge — all with per-type and per-difficulty breakdowns. |
| **Demo** | Gradio app exposing all four configs side-by-side on a user-supplied image + question. |
| **Reporting** | `<config>_errors.json` per config + a written analysis covering A1↔A2 ablation and B1↔B2 lift. |

---

## 3. Constraints

### 3.1 Technical

- **Language:** Python ≥ 3.14 (per `pyproject.toml`); managed by `uv`.
- **GPU envelope:** ≤16 GB VRAM (Kaggle T4 / P100 baseline). Qwen2-VL B1/B2 use 4-bit NF4 quantization; the modular pipeline trains in bf16 where available, fp32 otherwise.
- **No internet at train time:** when `KAGGLE_NO_INTERNET=1`, all HuggingFace loads use `local_files_only=True`. Models must be pre-staged into the Kaggle dataset mount.
- **Paths:** every script accepts `--data-dir` / `--output-dir` and falls back to `KAGGLE_INPUT_DIR` / `KAGGLE_WORKING_DIR` env vars, then to relative `data/` paths. No hardcoded absolutes.
- **Vietnamese text** is preserved with diacritics throughout the pipeline; lowercasing is confined to `normalize_answer`.

### 3.2 Architectural (binding)

- The modular and multimodal tracks share the same dataset, the same `normalize_answer`, and the same metric suite — they only diverge below the trainer boundary.
- The **answer-side vocabulary** (`AnswerTokenizer`) is separate from PhoBERT's input tokenizer and is only used by the modular decoders.
- Folder structure laid out in `PLANNING.md` is binding. New modules extend it; do not flatten or reorganize.
- Every config (A1/A2/B1/B2) inherits from `configs/base_config.yaml` and overrides only what changes.

### 3.3 Operational

- Must run end-to-end on a single Kaggle GPU notebook session (≤9 hours).
- All artifacts (checkpoints, logs, eval JSON) write under `<output-dir>` so they survive Kaggle "Save & Run All".
- Course materials (`*.docx`) and per-repo `CLAUDE.md` are gitignored and never pushed.

---

## 4. User Stories & Acceptance Criteria

### US-1 — Course evaluator runs A1 vs A2 ablation

**As a** course evaluator, **I want** to reproduce the A1↔A2 comparison from a single command, **so that** I can verify the decoder ablation claim in the report.

- [ ] `python scripts/train.py --config configs/A1.yaml` produces a checkpoint and a metrics JSON.
- [ ] Same with `A2.yaml`.
- [ ] `python scripts/eval.py --config configs/A1.yaml` and `A2.yaml` print per-type / per-difficulty breakdowns to stdout and write `reports/A1_metrics.json` / `A2_metrics.json` plus `*_errors.json`.

### US-2 — Course evaluator runs B1 zero-shot then B2 LoRA-tuned

**As a** course evaluator, **I want** to see the lift from Qwen2-VL zero-shot to Qwen2-VL LoRA-tuned, **so that** I can verify the multimodal pretrained track.

- [ ] `python scripts/eval.py --config configs/B1.yaml` runs zero-shot inference end-to-end on the test split.
- [ ] `python scripts/train.py --config configs/B2.yaml` produces a LoRA adapter checkpoint.
- [ ] `python scripts/eval.py --config configs/B2.yaml` loads the adapter and evaluates with the same metric suite.

### US-3 — Student demos the system live

**As a** student presenting the project, **I want** to drop an image into a UI and see all four configs answer the same question, **so that** I can demonstrate the comparison interactively.

- [ ] `python app/demo.py` launches a Gradio app on localhost.
- [ ] The app accepts an image upload + a Vietnamese question and returns four side-by-side answers (A1, A2, B1, B2) plus inference latency per config.

### US-4 — Reproducible dataset build

**As a** student, **I want** the dataset build to be deterministic given the same seed and raw inputs, **so that** every team member sees the same splits.

- [ ] `python scripts/build_dataset.py --data-dir data --output-dir data/processed` is byte-stable across runs (SEED=42).
- [ ] `--debug` produces a runnable subset (`{train:100, val:20, test:50}`) without tripping the corpus-size asserts.
- [ ] All validation rules from §2 fire on bad input with a clear error message.

### US-5 — Bonus: DPO preference comparison

**As a** student attempting the bonus track, **I want** to train a preference-tuned variant on ≥100 pairs and compare it to the SFT baseline, **so that** I can earn the bonus marks.

- [ ] `data/preference/preference.json` schema is documented and a stub exists.
- [ ] A trainer entry-point produces a DPO/PPO checkpoint.
- [ ] The eval pass reports SFT-vs-preference deltas on the same metric suite.

---

## 5. Out-of-Scope (Explicit)

- Any model > 7B parameters end-to-end on Kaggle without quantization.
- Multilingual inference (English questions, etc.).
- Streaming / chat-style multi-turn QA.
- Back-translation augmentation (explicitly excluded; paraphrase + synonym only).
- Production-grade serving infrastructure.

---

## 6. Definition of Done (Course submission)

- [ ] Dataset built: ≥2000 train · ≥200 unique images · ≥50 hand-curated test rows; splits image-disjoint.
- [ ] Four configs (A1, A2, B1, B2) train + eval cleanly with reproducible commands.
- [ ] Metric suite (VQA-Acc exact+soft, BLEU, ROUGE-L, METEOR, BERTScore, LLM-judge) implemented and run on all four configs.
- [ ] Per-type and per-difficulty breakdowns reported per config.
- [ ] Gradio demo runs locally and serves all four configs.
- [ ] Written report covers A1↔A2 ablation, B1↔B2 lift, dataset construction, and limitations.
- [ ] All artifacts under `reports/` (metrics JSON, errors JSON, sample outputs, training logs).
- [ ] Kaggle notebook runs end-to-end on a single GPU session within the time/memory envelope.
