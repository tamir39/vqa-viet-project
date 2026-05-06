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

Phase-1 dataset is shipped on HuggingFace Hub (**[Tamir39/foodlensvn](https://huggingface.co/datasets/Tamir39/foodlensvn)**, 5,572 rows, 20 dishes, ~34 MB). Foundations + the full modular pipeline (encoders, fusion, decoders, dataset class, model assembly) are implemented. Trainers, evaluation, and the demo app are still pending.

### Implemented

- **`src/utils/vn_text.py`** — `normalize_answer(text)`: single source of truth for Vietnamese answer canonicalization.
- **`src/utils/dishes.py`** — `CANONICAL_DISHES` (locked 20-dish set) + display names.
- **`src/data_loader/answer_tokenizer.py`** — `AnswerTokenizer`: domain-specific answer vocab (`<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3), `MAX_LEN=12`, JSON save/load. Phase-1 vocab size: 453.
- **`src/data_loader/vqa_dataset.py`** + **`collate.py`** + **`augment.py`** — dataset class reading processed splits, batched padding/PhoBERT tokenization, image augmentation.
- **`src/models/encoders/{image,text}_encoder.py`** — `timm` image encoder (default ResNet50; ablation ViT-S/16) and PhoBERT text encoder.
- **`src/models/fusion/cross_attention.py`** — co-attention fusion (primary) + element-wise / concat (ablations).
- **`src/models/decoders/{lstm,transformer}_decoder.py`** — LSTM (A1) and Transformer (A2) decoders over `AnswerTokenizer`.
- **`src/models/modular_vqa.py`** — end-to-end module wiring encoders → fusion → decoder.
- **`scripts/build_dataset.py`** — validates raw annotations, enriches with `answer_type` / `difficulty`, deduplicates, prefixes image paths to `<variant>/<split>/<file>`. Reads pre-split inputs from `<data-dir>/annotations/{train,val,test}.json`.
- **`scripts/fetch_dataset.py`** — pulls the HF dataset to `data/foodlensvn/` for local dev.
- **`scripts/push_dataset.py`** — uploads the staging dir to `Tamir39/foodlensvn` on HF Hub.
- **`src/models/multimodal/qwen_vl.py`** — Qwen2-VL-2B-Instruct loader (4-bit NF4), strict Vietnamese prompt, output cleaner.
- **`scripts/check_env.py`** — GPU + library + config sanity check.

### Not yet implemented

- `src/trainer/` — modular trainer (A1/A2 with token-level CE) and PEFT trainer (B2 with LoRA via TRL)
- `src/utils/metrics/` — VQA accuracy, BLEU/ROUGE/METEOR, BERTScore, LLM-judge
- `scripts/train.py`, `scripts/eval.py`, `scripts/infer.py`
- `app/demo.py` — Gradio interactive UI
- Filled `configs/A1.yaml`, `A2.yaml`, `B1.yaml`, `B2.yaml`
- Pre-staged HF model snapshots into a Kaggle dataset for offline (`KAGGLE_NO_INTERNET=1`) runs
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
│   ├── foodlensvn/             # HF-fetched (gitignored; via scripts/fetch_dataset.py)
│   │   ├── annotations/{train,val,test}.json
│   │   └── images/{raw,squared}/{train,val,test}/*.jpg
│   ├── processed/              # build_dataset.py outputs (gitignored)
│   │   ├── annotations/{train,val,test}.json
│   │   ├── answer_vocab.json
│   │   └── build_dataset.log
│   └── preference/             # DPO pairs (gitignored except stub)
├── notebooks/
│   └── train_foodlensvn.ipynb
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
  "image": "squared/train/pho_001.jpg",
  "image_id": "pho_001",
  "dish": "pho",
  "question": "Món này có cay không?",
  "answer": "không, món này không cay",
  "type": "yes_no",
  "answer_type": "classification",
  "difficulty": "easy",
  "source": "scrape"
}
```

**Field contract:**
- `id` — globally unique (`vfvqa-XXXXXX[-N]`).
- `image` — `<variant>/<split>/<filename>.jpg` after `build_dataset.py` has prefixed the bare filename. Resolves under `<images-root>` (default `data/foodlensvn/images`).
- `image_id` — used to enforce no-overlap between train and test (split is performed at this level).
- `dish` — canonical key from the locked **20-dish set** in [src/utils/dishes.py](src/utils/dishes.py); rejected by `build_dataset.py` if non-canonical.
- `question` — Vietnamese with diacritics.
- `answer` — Vietnamese, ≤10 words after `normalize_answer`. Phase-1 answers are full sentences (≈7–9 words).
- `type` — one of `yes_no | counting | recognition | attribute | spatial | reasoning`.
- `answer_type` — auto-derived: `yes_no | counting | recognition → classification`; `attribute | spatial | reasoning → generative`.
- `difficulty` — auto-derived: `yes_no | recognition → easy`; `counting | attribute → medium`; `spatial | reasoning → hard`.
- `source` — `scrape | dataset | self_shot` for license tracking.

**Validation rules** enforced by `build_dataset.py` (raise on violation; duplicates dropped with warning):
- `type` must be in the 6-value enum.
- `dish` must be in the canonical 20-dish set.
- `answer.split()` must be ≤10 tokens after canonicalization.
- `(image_id, question)` is unique within a split.
- Train, val, test `image_id` sets are pairwise disjoint.

In non-debug mode the script also asserts ≥200 unique images, ≥2000 train rows, and ≥50 test rows. Phase-1 ships 1,196 unique images and 4,460 train rows, so these pass without `--debug`.

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

The Phase-1 corpus is hosted on HuggingFace Hub as **[Tamir39/foodlensvn](https://huggingface.co/datasets/Tamir39/foodlensvn)** — 5,572 annotation rows over 20 dishes, with `raw/` and `squared/` image variants. It is **not** committed to git.

```bash
# 1. Fetch dataset locally (one-time; `hf auth login` recommended for rate-limit headroom)
python scripts/fetch_dataset.py                          # -> data/foodlensvn/

# 2. Build processed splits + answer vocab
python scripts/build_dataset.py --data-dir data/foodlensvn

# Debug mode (caps splits to {train:100, val:20, test:50}):
python scripts/build_dataset.py --data-dir data/foodlensvn --debug
```

On Kaggle, the notebook authenticates with the `HF_TOKEN` secret and calls `fetch_dataset.py` directly — no Kaggle dataset attachment needed.

CLI flags for `build_dataset.py`:
- `--data-dir` — raw root (default: `$FOODLENS_DATA_DIR` or `$KAGGLE_INPUT_DIR` or `data`)
- `--output-dir` — processed root (default: `$KAGGLE_WORKING_DIR` or `data/processed`)
- `--image-variant` — `squared` (default) or `raw`; sets the prefix in row `"image"` paths
- `--debug` — caps splits and skips corpus-size asserts
- `--build-preference` — emits an empty `data/preference/preference.json` skeleton

Outputs (under `<output-dir>`):
- `annotations/{train,val,test}.json` — rows with `"image": "<variant>/<split>/<file>.jpg"`
- `answer_vocab.json`
- `build_dataset.log`

---

## Kaggle workflow

The `notebooks/train_foodlensvn.ipynb` notebook handles cloning, dependency install, HF authentication via the `HF_TOKEN` secret, dataset fetch from HF Hub, and the GPU sanity check. Set `KAGGLE_NO_INTERNET=1` to force `local_files_only=True` for HuggingFace *model* loads (PhoBERT / Qwen / xlm-roberta) when those snapshots are pre-staged in a separate Kaggle dataset.

Trained checkpoints push back to HF too — e.g. `Tamir39/foodlensvn-A1`, `Tamir39/foodlensvn-A2`, `Tamir39/foodlensvn-B2-lora` — so they survive Kaggle session timeouts and can be loaded straight into the demo.

Default paths used inside the notebook:
- `FOODLENS_DATA_DIR=/kaggle/working/data/foodlensvn`
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
