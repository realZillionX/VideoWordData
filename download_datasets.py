#!/usr/bin/env python3
"""
Dataset Download Script for VideoWordData

Downloads all required datasets to local storage for offline use.
Supports resumable downloads and local caching.

Usage:
    python download_datasets.py
    python download_datasets.py --base_dir /your/custom/path
    python download_datasets.py --dataset gsm8k  # Download specific dataset only
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset

# Default download directory
DEFAULT_BASE_DIR = "/inspire/hdd/project/embodied-multimodality/public"

# Dataset configurations
DATASETS = {
    "gsm8k": {
        "name": "openai/gsm8k",
        "config": "main",
        "local_dir": "gsm8k",
        "description": "GSM8K Math Problems (English, ~7.5K samples)"
    },
    "openmath2_gsm8k": {
        "name": "ai2-adapt-dev/openmath-2-gsm8k",
        "config": None,
        "local_dir": "openmath2_gsm8k",
        "description": "OpenMath-2-GSM8K (English, large scale)"
    },
    "belle_school_math": {
        "name": "BelleGroup/school_math_0.25M",
        "config": None,
        "local_dir": "belle_school_math",
        "description": "BELLE School Math (Chinese, ~250K samples)"
    },
    "gsm8k_chinese": {
        "name": "swulling/gsm8k_chinese",
        "config": None,
        "local_dir": "gsm8k_chinese",
        "description": "GSM8K Chinese Translation (~8.8K samples)"
    },
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "local_dir": "tinystories",
        "description": "TinyStories (English, ~2.1M samples)"
    }
}


def check_local_dataset(local_path):
    """Check if dataset exists locally (has arrow files or parquet files)"""
    if not local_path.exists():
        return False
    
    # Check for common dataset file patterns
    patterns = ["*.arrow", "*.parquet", "*.json", "*.jsonl", "dataset_info.json"]
    for pattern in patterns:
        if list(local_path.glob(pattern)):
            return True
        # Also check subdirectories
        if list(local_path.glob(f"**/{pattern}")):
            return True
    
    return False


def download_dataset(dataset_key, base_dir, force=False):
    """Download a single dataset"""
    config = DATASETS[dataset_key]
    local_path = Path(base_dir) / config["local_dir"]
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_key}")
    print(f"  Source: {config['name']}")
    print(f"  Description: {config['description']}")
    print(f"  Local Path: {local_path}")
    print(f"{'='*60}")
    
    # Check if already exists
    if not force and check_local_dataset(local_path):
        print(f"✓ Dataset already exists locally. Skipping download.")
        print(f"  (Use --force to re-download)")
        return True
    
    # Create directory
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"⏳ Downloading from HuggingFace...")
        
        # Load dataset (this triggers download)
        if config["config"]:
            dataset = load_dataset(
                config["name"], 
                config["config"],
                cache_dir=str(local_path / ".cache")
            )
        else:
            dataset = load_dataset(
                config["name"],
                cache_dir=str(local_path / ".cache")
            )
        
        # Save to disk in arrow format for fast loading
        save_path = local_path / "dataset"
        print(f"💾 Saving to {save_path}...")
        dataset.save_to_disk(str(save_path))
        
        # Print dataset info
        print(f"✓ Download complete!")
        if hasattr(dataset, 'num_rows'):
            print(f"  Samples: {dataset.num_rows}")
        elif isinstance(dataset, dict):
            for split, data in dataset.items():
                print(f"  {split}: {len(data)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_key}: {e}")
        return False


def load_local_dataset(dataset_key, base_dir):
    """Load dataset from local storage, download if not exists"""
    from datasets import load_from_disk
    
    config = DATASETS[dataset_key]
    local_path = Path(base_dir) / config["local_dir"]
    save_path = local_path / "dataset"
    
    # Try loading from saved disk format first
    if save_path.exists():
        try:
            print(f"Loading {dataset_key} from local disk...")
            return load_from_disk(str(save_path))
        except Exception as e:
            print(f"Failed to load from disk: {e}")
    
    # Try loading from cache
    cache_path = local_path / ".cache"
    if cache_path.exists():
        try:
            print(f"Loading {dataset_key} from cache...")
            if config["config"]:
                return load_dataset(
                    config["name"],
                    config["config"],
                    cache_dir=str(cache_path)
                )
            else:
                return load_dataset(
                    config["name"],
                    cache_dir=str(cache_path)
                )
        except Exception as e:
            print(f"Failed to load from cache: {e}")
    
    # Download if not available locally
    print(f"Dataset not found locally, downloading...")
    download_dataset(dataset_key, base_dir)
    
    # Try loading again
    if save_path.exists():
        return load_from_disk(str(save_path))
    
    # Fallback to online
    print(f"Loading {dataset_key} from HuggingFace...")
    if config["config"]:
        return load_dataset(config["name"], config["config"])
    else:
        return load_dataset(config["name"])


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for VideoWordData project"
    )
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default=DEFAULT_BASE_DIR,
        help=f"Base directory for datasets (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists locally"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\nAvailable Datasets:")
        print("-" * 60)
        for key, config in DATASETS.items():
            print(f"  {key}:")
            print(f"    Source: {config['name']}")
            print(f"    Description: {config['description']}")
            print()
        return
    
    # Create base directory
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Base Directory: {base_dir}")
    
    # Determine which datasets to download
    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]
    
    # Download datasets
    results = {}
    for dataset_key in datasets_to_download:
        success = download_dataset(dataset_key, base_dir, force=args.force)
        results[dataset_key] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for dataset_key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {dataset_key}: {status}")
    
    # Exit code
    if all(results.values()):
        print("\n✓ All datasets ready!")
        return 0
    else:
        print("\n⚠ Some datasets failed to download.")
        return 1


if __name__ == "__main__":
    exit(main())
