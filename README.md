# FoodLensVN — Vietnamese Food Visual Question Answering

Final-term Deep Learning project (BÀI 1, 7 điểm). A modular VQA pipeline for Vietnamese cuisine: input is an image of a Vietnamese dish + a Vietnamese question; output is a short Vietnamese answer (≤10 words).

The system implements **four mandatory configurations** required by the course brief (`DLEndterm.docx`):

| Config | Approach | Description |
|--------|----------|-------------|
| **A1** | Modular | ResNet/ViT image encoder + PhoBERT/BiLSTM text encoder + co-attention fusion + **LSTM** decoder |
| **A2** | Modular | Same as A1, but with a **Transformer** decoder (the A1↔A2 ablation is the core comparison) |
| **B1** | Multimodal pretrained | **Qwen2-VL-2B-Instruct** zero-shot |
| **B2** | Multimodal pretrained | Qwen2-VL-2B-Instruct fine-tuned with LoRA/PEFT |

Evaluation suite: VQA Accuracy (exact + soft), BLEU, ROUGE-L, METEOR, BERTScore (xlm-roberta-base), and an offline LLM-as-judge — all with per-type and per-difficulty breakdowns.

---

## Current status

This repo is in **scaffold + foundations** stage. Schema, canonicalization, vocabulary, dataset builder, and the Qwen2-VL prompt/loader are implemented and tested. Encoders, decoders, fusion, trainers, evaluation, and the demo app are still empty placeholders.

### Implemented

- **`src/utils/vn_text.py`** — `normalize_answer(text)`: the single source of truth for canonicalizing Vietnamese answers (lowercase + trim + yes/no whole-string match + Vietnamese-number-word → digit mapping + classifier-suffix strip + trailing-punctuation strip).
- **`src/data_loader/answer_tokenizer.py`** — `AnswerTokenizer`: small domain-specific answer vocabulary for the modular decoders (separate from the question-side PhoBERT tokenizer). Fixed special ids (`<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3), `MAX_LEN=12`, frequency-sorted vocab with `min_freq` and `max_vocab_size` controls, JSON save/load.
- **`scripts/build_dataset.py`** — Validates raw annotations, enriches with `answer_type` and `difficulty`, deduplicates `(image_id, question)`, splits 80/10/10 at the **image_id level** stratified by dish (image-disjoint train↔test), and builds the answer vocab from training answers. Supports `--debug` (size caps) and `--build-preference` (DPO stub).
- **`src/models/multimodal/qwen_vl.py`** — Qwen2-VL-2B-Instruct loader (4-bit NF4 by default for ≤16 GB GPU; bf16 fallback), strict Vietnamese system prompt, chat-template message builder, and an output cleaner that strips prompt echoes (`Trả lời:` / `Câu hỏi:`), trailing punctuation, and re-canonicalizes via `normalize_answer`.
- **`scripts/check_env.py`** — Sanity check for GPU + library imports + base config loading.
- **`data/annotations/train.json`** — 22-row valid stub across 3 dishes covering all 6 question types (used for `--debug` smoke runs until real data lands).

### Not yet implemented

- `src/models/encoders/` — image encoder (ResNet/ViT/EfficientNet via `timm`) and text encoder (PhoBERT, BiLSTM)
- `src/models/fusion/` — co-attention (primary), element-wise, concat (ablations)
- `src/models/decoders/` — LSTM and Transformer decoders consuming the `AnswerTokenizer` vocab
- `src/trainer/` — modular trainer (A1/A2 with token-level CE) and PEFT trainer (B2 with LoRA via TRL)
- `src/utils/metrics/` — VQA accuracy, BLEU/ROUGE/METEOR, BERTScore, LLM-judge
- `src/data_loader/vqa_dataset.py`, `collate.py`, `augment.py` — dataset class, batching, paraphrase/synonym augmentation
- `scripts/train.py`, `scripts/eval.py`, `scripts/infer.py`
- `app/demo.py` — Gradio interactive UI
- Filled `configs/A1.yaml`, `A2.yaml`, `B1.yaml`, `B2.yaml`
- Real dataset (≥2000 train / ≥200 unique images / ≥50 hand-curated test)
- DPO/PPO preference training (bonus track)

---

## Repository layout

```
FoodLensVN/
├── app/                        # Gradio demo (planned)
├── configs/
│   ├── base_config.yaml        # shared defaults (paths, hparams)
│   ├── A1.yaml                 # placeholder
│   ├── A2.yaml                 # placeholder
│   ├── B1.yaml                 # placeholder
│   └── B2.yaml                 # placeholder
├── data/
│   ├── annotations/
│   │   └── train.json          # raw pool (tracked; 22-row stub)
│   ├── raw/                    # images (gitignored)
│   ├── processed/              # build_dataset.py outputs (gitignored)
│   │   ├── annotations/{train,val,test}.json
│   │   ├── answer_vocab.json
│   │   └── build_dataset.log
│   └── preference/             # DPO pairs (gitignored except stub)
├── notebooks/
│   └── kaggle_template.ipynb
├── reports/                    # checkpoints / logs / results (planned)
├── scripts/
│   ├── check_env.py
│   └── build_dataset.py
├── src/
│   ├── data_loader/
│   │   └── answer_tokenizer.py
│   ├── models/
│   │   ├── encoders/           # placeholder
│   │   ├── decoders/           # placeholder
│   │   ├── fusion/             # placeholder
│   │   └── multimodal/
│   │       └── qwen_vl.py
│   ├── trainer/                # placeholder
│   └── utils/
│       └── vn_text.py
├── tests/                      # planned
├── main.py                     # stub
├── pyproject.toml
├── uv.lock
├── .python-version
└── DLEndterm.docx              # course brief (Vietnamese; the contract)
```

---

## Dataset schema

Each annotation row (in raw input and all processed splits):

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

**Field contract:**
- `id` — globally unique (`vfvqa-XXXXXX`).
- `image` — path relative to `data/`.
- `image_id` — used to enforce no-overlap between train and test (split is performed at this level).
- `dish` — canonical key (`pho`, `bun_bo`, `banh_mi`, `com_tam`, …); used to stratify splits.
- `question` — Vietnamese with diacritics.
- `answer` — Vietnamese, ≤10 words after `normalize_answer`.
- `type` — one of `yes_no | counting | recognition | attribute | spatial | reasoning`.
- `answer_type` — auto-derived (see below); `classification` for yes_no/counting/recognition, `generative` for attribute/spatial/reasoning.
- `difficulty` — auto-derived; `easy` (yes_no, recognition), `medium` (counting, attribute), `hard` (spatial, reasoning).
- `source` — `scrape | dataset | self_shot` for license tracking.

**Validation rules** enforced by `build_dataset.py` (raise on violation; duplicates dropped with warning):
- `type` must be in the 6-value enum.
- `answer.split()` must be ≤10 tokens after canonicalization.
- `type == "counting"` ⇒ `answer.isdigit()` after canonicalization (e.g., `"hai"` → `"2"`, `"2 cái"` → `"2"`).
- `(image_id, question)` is unique within a split.
- Train and test image_id sets are disjoint.

In non-debug mode the script also asserts ≥200 unique images, ≥2000 train rows, and ≥50 test rows.

---

## Approach B: Qwen2-VL-2B-Instruct prompt contract

Strict, reused for both B1 (zero-shot) and B2 (LoRA SFT format):

```
SYSTEM: Bạn là hệ thống VQA. Trả lời bằng tiếng Việt. Chỉ trả lời, không giải thích. Tối đa 10 từ.
USER:   <image>
        Câu hỏi: {question}
        Trả lời:
```

Generation: `max_new_tokens=20`, `do_sample=False` (greedy). Output is post-processed to cut at the first newline, strip `Trả lời:` / `Câu hỏi:` echoes and trailing punctuation, then run through `normalize_answer` so it shares the same canonical form as gold answers used in evaluation.

---

## Setup

### Prerequisites
- Python ≥ 3.14 (per `pyproject.toml`)
- [uv](https://github.com/astral-sh/uv)
- (Optional) CUDA-capable GPU for training; ≤16 GB VRAM is sufficient (Qwen2-VL B2 uses 4-bit NF4)

### Local install

```bash
uv venv
.venv\Scripts\activate                  # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)
# source .venv/bin/activate              # macOS/Linux
uv sync
```

### Sanity check

```bash
python scripts/check_env.py
```

---

## Data pipeline

Build splits + answer vocab from the raw annotation pool:

```bash
# Local debug (uses the 22-row stub at data/annotations/train.json):
python scripts/build_dataset.py --debug --data-dir data --output-dir data/processed

# Full mode (requires real data ≥200 images / ≥2000 train rows):
python scripts/build_dataset.py --data-dir data --output-dir data/processed

# With DPO preference stub:
python scripts/build_dataset.py --debug --build-preference
```

CLI flags:
- `--data-dir` — raw root (default: `$KAGGLE_INPUT_DIR` or `data`)
- `--output-dir` — processed root (default: `$KAGGLE_WORKING_DIR` or `data/processed`)
- `--debug` — caps splits to `{train: 100, val: 20, test: 50}` and skips corpus-size asserts
- `--build-preference` — emits an empty `data/preference/preference.json` skeleton

Outputs (under `<output-dir>`):
- `annotations/train.json`, `val.json`, `test.json`
- `answer_vocab.json`
- `build_dataset.log`

---

## Kaggle workflow

The `notebooks/kaggle_template.ipynb` notebook handles cloning, dependency install, and GPU sanity check. Set `KAGGLE_NO_INTERNET=1` to force `local_files_only=True` for HuggingFace loads when running on a no-internet Kaggle accelerator.

Default paths used by `build_dataset.py` and (later) trainers when env vars are set:
- `KAGGLE_INPUT_DIR=/kaggle/input/foodlensvn`
- `KAGGLE_WORKING_DIR=/kaggle/working`

All scripts use relative + configurable paths — no hardcoded absolutes.

---

## Module quick reference

### `from src.utils.vn_text import normalize_answer`

```python
normalize_answer("Hai cái")       # -> "2"
normalize_answer("Khong")         # -> "không"
normalize_answer("CÓ.")           # -> "có"
normalize_answer("mười miếng")    # -> "10"
normalize_answer("không cái")     # -> "0"   (counting context)
```

Pipeline: lowercase + trim + collapse whitespace → strip trailing punctuation → yes/no whole-string match (early return) → per-token Vietnamese-number-word → digit → strip trailing classifier (`cái`, `miếng`, `phần`, `tô`, `bát`).

### `from src.data_loader.answer_tokenizer import AnswerTokenizer`

```python
t = AnswerTokenizer.build_from_corpus(train_answers, min_freq=1, max_vocab_size=1000)
ids = t.encode("có")                        # 12 ids: [BOS, ..., EOS, PAD, ...]
text = t.decode(ids)                        # "có"
t.save("data/processed/answer_vocab.json")
t = AnswerTokenizer.load("data/processed/answer_vocab.json")
```

### `from src.models.multimodal.qwen_vl import load_qwen_vl, generate, build_messages`

```python
model, processor = load_qwen_vl(quantize_4bit=True)   # 4-bit NF4 on CUDA
answer = generate(model, processor, image_pil, "Đây là món gì?")
```

---

## Verification

The smoke-test path that all four implemented files cover:

```bash
# 1. Canonicalization
python -c "from src.utils.vn_text import normalize_answer as N; \
print(N('Hai cái'), N('Khong'), N('CÓ.'), N('mười miếng'))"

# 2. Tokenizer round-trip
python -c "from src.data_loader.answer_tokenizer import AnswerTokenizer; \
t=AnswerTokenizer.build_from_corpus(['có','không','2','phở bò','có']); \
print(t.decode(t.encode('có')))"

# 3. Build dataset (debug)
python scripts/build_dataset.py --debug --data-dir data --output-dir data/processed

# 4. Qwen2-VL prompt + output cleaning (no model load required)
python -c "from src.models.multimodal.qwen_vl import build_messages, _clean_output; \
print(_clean_output('Trả lời: Không\nCâu hỏi: ...'))"
```

---

## Roadmap (next implementation passes)

1. Encoders (`src/models/encoders/{image,text}_encoder.py`) — ResNet50/ViT via `timm` and PhoBERT-base.
2. Co-attention fusion + LSTM and Transformer decoders consuming `AnswerTokenizer` vocab.
3. `vqa_dataset.py` + collate fn + image/text augmentation (no back-translation).
4. Modular trainer (A1/A2) with AMP, schedulers, ckpt.
5. Metrics module (VQA-Acc, BLEU/ROUGE/METEOR, BERTScore xlm-roberta-base, local LLM-judge) with per-type/difficulty/answer-type breakdowns.
6. PEFT trainer for B2 (LoRA via TRL `SFTTrainer`, NF4 base).
7. `scripts/eval.py` with error-analysis output (`<config>_errors.json`).
8. Gradio demo (`app/demo.py`).
9. Real data ingest (web scrape + 30VNFoods + self-shot for hand-curated test set).
10. Bonus: DPO/PPO preference training (≥100 pairs), RL vs SFT comparison.

---

## Conventions

- All paths in code/configs are relative and configurable (env-var overridable).
- Vietnamese text is preserved with diacritics; only `normalize_answer` lowercases.
- The answer-side vocabulary is **separate** from PhoBERT (PhoBERT remains the question-side input encoder); decoders use `AnswerTokenizer` exclusively.
- Branch protection: `main` is touched only on shipping; daily work goes to feature branches off `develop`.
