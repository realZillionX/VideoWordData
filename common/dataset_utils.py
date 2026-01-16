"""
Dataset Loading Utilities

Provides functions to load datasets with local-first strategy:
1. First try to load from local disk (saved by download_datasets.py)
2. Fall back to HuggingFace if local not available
"""

from pathlib import Path
from datasets import load_dataset, load_from_disk

# Default local dataset directory
DEFAULT_LOCAL_BASE = "/inspire/hdd/project/embodied-multimodality/public"

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "hf_config": "main",
        "local_dir": "gsm8k",
    },
    "openmath2_gsm8k": {
        "hf_name": "ai2-adapt-dev/openmath-2-gsm8k",
        "hf_config": None,
        "local_dir": "openmath2_gsm8k",
    },
    "belle_school_math": {
        "hf_name": "BelleGroup/school_math_0.25M",
        "hf_config": None,
        "local_dir": "belle_school_math",
    },
    "tinystories": {
        "hf_name": "roneneldan/TinyStories",
        "hf_config": None,
        "local_dir": "tinystories",
    },
    "tinystories_chinese": {
        "hf_name": "adam89/TinyStoriesChinese",
        "hf_config": None,
        "local_dir": "tinystories_chinese",
    },
}


def load_dataset_with_local_fallback(
    dataset_key: str,
    split: str = "train",
    local_base: str = DEFAULT_LOCAL_BASE,
) -> "Dataset":
    """
    Load dataset with local-first strategy.
    
    Args:
        dataset_key: Key identifying the dataset (e.g., 'gsm8k', 'tinystories')
        split: Dataset split to load (default: 'train')
        local_base: Base directory for local datasets
        
    Returns:
        Loaded dataset
    """
    config = DATASET_CONFIGS.get(dataset_key)
    if config is None:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASET_CONFIGS.keys())}")
    
    local_path = Path(local_base) / config["local_dir"]
    saved_path = local_path / "dataset"
    
    # Strategy 1: Try loading from saved disk format (fastest)
    if saved_path.exists():
        try:
            print(f"Loading {dataset_key} from local disk: {saved_path}")
            dataset = load_from_disk(str(saved_path))
            if split and isinstance(dataset, dict):
                dataset = dataset[split]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
            return dataset
        except Exception as e:
            print(f"Failed to load from disk: {e}")
    
    # Strategy 2: Try loading from local cache
    cache_path = local_path / ".cache"
    if cache_path.exists():
        try:
            print(f"Loading {dataset_key} from local cache: {cache_path}")
            if config["hf_config"]:
                dataset = load_dataset(
                    config["hf_name"],
                    config["hf_config"],
                    split=split,
                    cache_dir=str(cache_path)
                )
            else:
                dataset = load_dataset(
                    config["hf_name"],
                    split=split,
                    cache_dir=str(cache_path)
                )
            print(f"✓ Loaded {len(dataset)} samples from cache")
            return dataset
        except Exception as e:
            print(f"Failed to load from cache: {e}")
    
    # Strategy 3: Try loading from local directory as HuggingFace format
    if local_path.exists():
        try:
            print(f"Loading {dataset_key} from local HF format: {local_path}")
            if config["hf_config"]:
                dataset = load_dataset(str(local_path), config["hf_config"], split=split)
            else:
                dataset = load_dataset(str(local_path), split=split)
            print(f"✓ Loaded {len(dataset)} samples from local HF format")
            return dataset
        except Exception as e:
            print(f"Failed to load from local HF format: {e}")
    
    # Strategy 4: Fall back to HuggingFace (will download)
    print(f"Loading {dataset_key} from HuggingFace: {config['hf_name']}")
    if config["hf_config"]:
        dataset = load_dataset(config["hf_name"], config["hf_config"], split=split)
    else:
        dataset = load_dataset(config["hf_name"], split=split)
    print(f"✓ Loaded {len(dataset)} samples from HuggingFace")
    return dataset


# Convenience functions for each dataset
def load_gsm8k(split="train", local_base=DEFAULT_LOCAL_BASE):
    """Load GSM8K dataset"""
    return load_dataset_with_local_fallback("gsm8k", split, local_base)


def load_openmath2_gsm8k(split="train", local_base=DEFAULT_LOCAL_BASE):
    """Load OpenMath-2-GSM8K dataset"""
    return load_dataset_with_local_fallback("openmath2_gsm8k", split, local_base)


def load_belle_school_math(split="train", local_base=DEFAULT_LOCAL_BASE):
    """Load BELLE School Math dataset"""
    return load_dataset_with_local_fallback("belle_school_math", split, local_base)


def load_tinystories(split="train", local_base=DEFAULT_LOCAL_BASE):
    """Load TinyStories dataset"""
    return load_dataset_with_local_fallback("tinystories", split, local_base)


def load_tinystories_chinese(split="train", local_base=DEFAULT_LOCAL_BASE):
    """Load TinyStories Chinese dataset"""
    return load_dataset_with_local_fallback("tinystories_chinese", split, local_base)
