# Vietnamese Visual Question Answering (VQA) System

This project implements a modular Deep Learning pipeline for a Visual Question Answering (VQA) system tailored for the Vietnamese language, specifically focusing on domain-specific datasets (e.g., Vietnamese Cuisine).

## 🏗 Project Architecture

The project follows a modular AI structure to ensure scalability, maintainability, and reproducibility.

```text
vqa-viet-project/
├── app/              # Demo application entry points
├── configs/          # YAML configuration files (A1, A2, B1, B2)
├── data/             # Dataset management
│   ├── raw/          # Original images/data
│   ├── processed/    # Preprocessed features/data
│   ├── annotations/  # JSON annotation files
│   └── preference/   # Preference data for RLHF
├── notebooks/        # EDA and experimentation notebooks
├── reports/          # Evaluation results and visualizations
├── scripts/          # Entry points for train/eval/infer
├── src/              # Core project source code
│   ├── data_loader/  # Data loading and batching logic
│   ├── models/       # Model architecture (encoders, decoders, fusion)
│   ├── trainer/      # Training loop and engine
│   └── utils/        # Metric calculations and helper functions
├── tests/            # Unit tests for data integrity/model shapes
├── .gitignore        # Ignored files (data/*, models/*, __pycache__)
└── README.md         # Project documentation
```

### Component Descriptions

- **`configs/`**: Stores YAML files defining hyperparameters and model architecture settings for different experimental configurations (e.g., A1, A2, B1, B2).
- **`data/`**: Manages the dataset lifecycle. Note: The `.gitignore` file is configured to ignore `data/raw/`, `data/processed/`, and `data/preference/` to avoid storing large binary files in the repository. Only `data/annotations/*.json` files are tracked.
- **`src/`**: Contains the core logic:
  - `models/`: Modular components including encoders for visual/textual features, decoders for answer generation, and fusion modules for feature integration.
  - `data_loader/`: Custom PyTorch `Dataset` and `DataLoader` implementations.
  - `trainer/`: Orchestration of the training process, logging, and checkpointing.
- **`scripts/`**: Executable scripts for operational tasks.
- **`tests/`**: Unit tests designed to ensure data ingestion integrity and verify tensor shapes during model development.

## 🚀 Setup & Workflow

### Prerequisites
- Python 3.x
- [uv](https://github.com/astral-sh/uv) (for package and environment management)

### Installation
```bash
# Initialize project environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### Training Workflow
To train a model using a specific configuration, use the provided training script:

```bash
python scripts/train.py --config configs/base_config.yaml
```
*(You may create specific configuration files in `configs/` like `config_A1.yaml` and pass them to the script.)*
