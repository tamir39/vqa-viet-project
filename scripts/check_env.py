"""Sanity-check script for the FoodLensVN dev environment.

Verifies GPU access, library imports, and base_config.yaml loading.
Robust to one library failing — keeps going and reports a final summary.
"""

import os

# Pin a backend that always exists. Some Kaggle kernels export
# MPLBACKEND=module://matplotlib_inline.backend_inline; .venv subprocesses
# don't have matplotlib_inline, so any pyplot import (e.g. bert_score) blows
# up. Setting Agg here at the top of *every* script is belt-and-suspenders.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["MPLBACKEND"] = "Agg"  # force-override, not just default

import sys

import torch
import yaml


def check_env() -> None:
    print("--- Environment Sanity Check ---")

    # 1. GPU
    print("\n[1/3] Checking GPU availability...")
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: CUDA is not available. Training will be performed on CPU.")

    # 2. Library imports
    libraries = [
        "torch", "torchvision", "transformers", "peft", "bitsandbytes",
        "pyvi", "underthesea", "evaluate", "bert_score", "rouge_score",
        "scipy", "nltk", "yaml", "PIL", "tqdm", "gradio", "timm",
        "huggingface_hub",
    ]
    print("\n[2/3] Checking library imports:")
    failed: list[str] = []
    for lib in libraries:
        try:
            if lib == "PIL":
                from PIL import Image  # noqa: F401
            else:
                __import__(lib)
            print(f"  [OK] {lib}")
        except Exception as e:  # noqa: BLE001  — keep going through failures
            print(f"  [FAIL] {lib} - {type(e).__name__}: {e}")
            failed.append(lib)

    # 3. Base config
    config_path = "configs/base_config.yaml"
    print(f"\n[3/3] Checking config loading from {config_path}...")
    config_ok = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            required = {"dataset", "loader", "model", "training"}
            missing = required - set(cfg.keys())
            if missing:
                print(f"  [WARN] base_config missing keys: {sorted(missing)}")
            else:
                print(f"  [OK] {config_path} loaded with required sections")
                config_ok = True
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] error loading {config_path}: {e}")
    else:
        print(f"  [FAIL] {config_path} not found")

    print("\n" + "=" * 30)
    if not failed and config_ok:
        print("Conclusion: Environment is properly set up!")
    else:
        bits = []
        if failed:
            bits.append(f"{len(failed)} library import(s) failed: {failed}")
        if not config_ok:
            bits.append("base_config.yaml not loadable")
        print("Conclusion: Environment setup is INCOMPLETE — " + "; ".join(bits))
    print("=" * 30)


if __name__ == "__main__":
    check_env()
