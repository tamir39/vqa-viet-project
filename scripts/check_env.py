import torch
import yaml
import os
import sys

def check_env():
    """
    Sanity check script to verify the development environment for Vietnamese VQA project.
    """
    print("--- Environment Sanity Check ---")
    
    # 1. Check GPU availability
    print(f"\n[1/3] Checking GPU availability...")
    gpu_available = torch.cuda.is_available()
    print(f"CUDA available: {gpu_available}")
    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Note: CUDA is not available. Training will be performed on CPU.")
    
    # 2. Check library imports
    libraries = [
        "torch", "torchvision", "transformers", "peft", "bitsandbytes",
        "pyvi", "underthesea", "evaluate", "bert_score", "rouge_score",
        "scipy", "nltk", "yaml", "PIL", "tqdm", "gradio"
    ]
    
    print("\n[2/3] Checking library imports:")
    all_imported = True
    for lib in libraries:
        try:
            # Handle some special import names
            if lib == "PIL":
                from PIL import Image
            elif lib == "yaml":
                import yaml
            else:
                __import__(lib)
            print(f"  [OK] {lib}")
        except ImportError as e:
            print(f"  [FAIL] {lib} - {e}")
            all_imported = False
            
    # 3. Check base_config.yaml loading
    config_path = "configs/base_config.yaml"
    print(f"\n[3/3] Checking config loading from {config_path}...")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"  [OK] Successfully loaded {config_path}")
            # Verify basic keys exist
            if 'hyperparameters' in config and 'models' in config:
                print("  [OK] Config structure looks valid.")
            else:
                print("  [WARN] Config structure might be missing expected keys.")
        except Exception as e:
            print(f"  [FAIL] Error loading {config_path}: {e}")
            all_imported = False
    else:
        print(f"  [FAIL] {config_path} not found")
        all_imported = False

    print("\n" + "="*30)
    if all_imported:
        print("Conclusion: Environment is properly set up!")
    else:
        print("Conclusion: Environment setup is INCOMPLETE. Please check the failures above.")
    print("="*30)

if __name__ == "__main__":
    check_env()
