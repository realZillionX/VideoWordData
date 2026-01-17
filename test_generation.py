#!/usr/bin/env python3
"""
Test script to verify video generation for all datasets.
Uses real data from HuggingFace to test each script's video generation pipeline.

Usage:
    python test_generation.py           # Test all datasets
    python test_generation.py gsm8k     # Test specific dataset
    python test_generation.py --clean   # Remove test output after testing
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

# Test configuration
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"
NUM_TEST_SAMPLES = 5  # Number of samples to generate per dataset

# All datasets to test
DATASETS = {
    "gsm8k": {
        "inference": "inference/gsm8k.py",
        "rendering": "rendering/gsm8k.py",
    },
    "openmath2_gsm8k": {
        "inference": "inference/openmath2_gsm8k.py",
        "rendering": "rendering/openmath2_gsm8k.py",
    },
    "belle_school_math": {
        "inference": "inference/belle_school_math.py",
        "rendering": "rendering/belle_school_math.py",
    },
    "tinystories": {
        "inference": "inference/tinystories.py",
        "rendering": "rendering/tinystories.py",
    },
    "tinystories_chinese": {
        "inference": "inference/tinystories_chinese.py",
        "rendering": "rendering/tinystories_chinese.py",
    },
}


def run_test(script_path: str, dataset_name: str, task_type: str, num_samples: int = 5) -> dict:
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name} ({task_type})")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, script_path,
        "--base_dir", str(TEST_OUTPUT_DIR),
        "--num_samples", str(num_samples),
        "--num_workers", "4",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per test
        )
        
        success = result.returncode == 0
        
        # Count generated files
        video_dir = TEST_OUTPUT_DIR / dataset_name / "video"
        video_count = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
        
        jsonl_pattern = f"{dataset_name}_{task_type}_video_data_*.jsonl"
        jsonl_dir = TEST_OUTPUT_DIR / dataset_name
        jsonl_files = list(jsonl_dir.glob(jsonl_pattern)) if jsonl_dir.exists() else []
        jsonl_count = len(jsonl_files)
        
        # Count entries in JSONL
        jsonl_entries = 0
        for jf in jsonl_files:
            with open(jf) as f:
                jsonl_entries += sum(1 for _ in f)
        
        return {
            "dataset": dataset_name,
            "task": task_type,
            "success": success,
            "videos": video_count,
            "jsonl_files": jsonl_count,
            "jsonl_entries": jsonl_entries,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset_name,
            "task": task_type,
            "success": False,
            "videos": 0,
            "jsonl_files": 0,
            "jsonl_entries": 0,
            "error": "Timeout (300s exceeded)",
        }
    except Exception as e:
        return {
            "dataset": dataset_name,
            "task": task_type,
            "success": False,
            "videos": 0,
            "jsonl_files": 0,
            "jsonl_entries": 0,
            "error": str(e),
        }


def print_results(results: list):
    """Print test results in a table format."""
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<25} {'Task':<12} {'Status':<8} {'Videos':<8} {'JSONL Entries'}")
    print("-"*80)
    
    passed = 0
    failed = 0
    
    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"{r['dataset']:<25} {r['task']:<12} {status:<8} {r['videos']:<8} {r['jsonl_entries']}")
        
        if r["success"]:
            passed += 1
        else:
            failed += 1
            if "error" in r:
                print(f"   Error: {r['error']}")
            elif r.get("stderr"):
                # Show last line of stderr
                last_error = r["stderr"].strip().split("\n")[-1]
                print(f"   Error: {last_error[:70]}...")
    
    print("-"*80)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test video generation for all datasets")
    parser.add_argument("dataset", nargs="?", default="all",
                       help="Dataset to test (or 'all' for all datasets)")
    parser.add_argument("--task", choices=["inference", "rendering", "all"], default="inference",
                       help="Task type to test (default: inference)")
    parser.add_argument("--clean", action="store_true",
                       help="Remove test output directory after testing")
    parser.add_argument("--samples", type=int, default=NUM_TEST_SAMPLES,
                       help=f"Number of samples to test (default: {NUM_TEST_SAMPLES})")
    
    args = parser.parse_args()
    
    num_samples = args.samples
    
    # Prepare test output directory
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to test
    if args.dataset == "all":
        datasets_to_test = list(DATASETS.keys())
    else:
        if args.dataset not in DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(DATASETS.keys())}")
            return 1
        datasets_to_test = [args.dataset]
    
    # Determine which tasks to test
    if args.task == "all":
        tasks_to_test = ["inference", "rendering"]
    else:
        tasks_to_test = [args.task]
    
    # Run tests
    results = []
    for dataset in datasets_to_test:
        for task in tasks_to_test:
            script = DATASETS[dataset].get(task)
            if script:
                result = run_test(script, dataset, task, num_samples)
                results.append(result)
    
    # Print summary
    all_passed = print_results(results)
    
    # Cleanup if requested
    if args.clean and TEST_OUTPUT_DIR.exists():
        print(f"\nCleaning up: Removing {TEST_OUTPUT_DIR}")
        shutil.rmtree(TEST_OUTPUT_DIR)
    else:
        print(f"\nTest outputs saved to: {TEST_OUTPUT_DIR}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
