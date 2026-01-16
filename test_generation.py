#!/usr/bin/env python3
"""
Test Script for VideoWordData

Directly calls the main() functions from inference and rendering scripts
to verify the entire pipeline works correctly.

Usage:
    python test_generation.py
    python test_generation.py --base_dir /custom/output/path
    python test_generation.py --task inference  # Only test inference
    python test_generation.py --task rendering  # Only test rendering
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# All datasets
DATASETS = ["gsm8k", "openmath2_gsm8k", "belle_school_math", "tinystories", "tinystories_chinese"]

# Task types
TASK_TYPES = ["inference", "rendering"]


def run_script(script_path, base_dir, num_samples=1):
    """
    Run a generation script with subprocess to properly test it.
    
    Args:
        script_path: Path to the Python script
        base_dir: Output directory
        num_samples: Number of samples to generate
        
    Returns:
        tuple: (success: bool, output: str)
    """
    cmd = [
        sys.executable,
        str(script_path),
        "--base_dir", str(base_dir),
        "--num_samples", str(num_samples),
        "--num_workers", "1"  # Use single worker for testing
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per script
            cwd=str(PROJECT_ROOT)
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0
        
        return success, output
        
    except subprocess.TimeoutExpired:
        return False, "Timeout expired (5 minutes)"
    except Exception as e:
        return False, str(e)


def test_dataset(dataset, task_type, base_dir):
    """Test a single dataset with a specific task type"""
    script_path = PROJECT_ROOT / task_type / f"{dataset}.py"
    
    if not script_path.exists():
        return {
            "status": "skipped",
            "error": f"Script not found: {script_path}"
        }
    
    print(f"\n{'='*60}")
    print(f"Testing: {task_type}/{dataset}.py")
    print(f"{'='*60}")
    
    success, output = run_script(script_path, base_dir, num_samples=1)
    
    if success:
        print(f"✓ {task_type}/{dataset}.py passed")
        
        # Find generated files
        dataset_dir = Path(base_dir) / dataset
        video_dir = dataset_dir / "video"
        
        videos = list(video_dir.glob("*.mp4")) if video_dir.exists() else []
        jsonl_pattern = f"{dataset}_{task_type}_video_data_*.jsonl"
        jsonls = list(dataset_dir.glob(jsonl_pattern)) if dataset_dir.exists() else []
        
        return {
            "status": "success",
            "videos": [str(v) for v in videos],
            "jsonl_files": [str(j) for j in jsonls],
        }
    else:
        print(f"✗ {task_type}/{dataset}.py failed")
        # Print last 500 chars of output for debugging
        if len(output) > 500:
            print(f"...{output[-500:]}")
        else:
            print(output)
        
        return {
            "status": "failed",
            "error": output[-1000:] if len(output) > 1000 else output
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test video generation by calling actual inference/rendering scripts"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/zillionx/NLP/testData",
        help="Base directory for test output"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["inference", "rendering", "all"],
        default="all",
        help="Which task type to test (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS + ["all"],
        default="all",
        help="Which dataset to test (default: all)"
    )
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("VideoWordData Test Script")
    print("="*60)
    print(f"Output Directory: {base_dir}")
    print(f"Task Type: {args.task}")
    print(f"Dataset: {args.dataset}")
    print("="*60)
    
    # Determine what to test
    if args.task == "all":
        task_types = TASK_TYPES
    else:
        task_types = [args.task]
    
    if args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = [args.dataset]
    
    # Run tests
    results = {}
    
    for task_type in task_types:
        results[task_type] = {}
        for dataset in datasets:
            result = test_dataset(dataset, task_type, base_dir)
            results[task_type][dataset] = result
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for task_type in task_types:
        print(f"\n{task_type.upper()}:")
        for dataset in datasets:
            result = results[task_type][dataset]
            total_tests += 1
            
            if result["status"] == "success":
                passed_tests += 1
                videos = result.get("videos", [])
                jsonls = result.get("jsonl_files", [])
                print(f"  ✓ {dataset}")
                if videos:
                    print(f"    Videos: {len(videos)} files")
                if jsonls:
                    print(f"    JSONL: {len(jsonls)} files")
            elif result["status"] == "skipped":
                print(f"  ⊘ {dataset}: {result.get('error', 'skipped')}")
            else:
                print(f"  ✗ {dataset}: failed")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    # Save results
    results_file = base_dir / "test_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_file}")
    
    # List generated files
    print("\n" + "="*60)
    print("Generated Files")
    print("="*60)
    
    for task_type in task_types:
        for dataset in datasets:
            dataset_dir = base_dir / dataset
            if dataset_dir.exists():
                video_dir = dataset_dir / "video"
                if video_dir.exists():
                    for video in video_dir.glob("*.mp4"):
                        print(f"  {video}")
                for jsonl in dataset_dir.glob("*.jsonl"):
                    print(f"  {jsonl}")
    
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    exit(main())
