#!/usr/bin/env python3
"""
Test script for audio video generation across all datasets.
Generates 5 test videos per dataset.

Usage:
    python test_audio_all.py
    python test_audio_all.py --num_samples 3
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

# Test configuration
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output_audio"
DEFAULT_NUM_SAMPLES = 5

# Audio datasets to test
AUDIO_DATASETS = {
    "tinystories_inference": "inference_audio/tinystories.py",
    "tinystories_rendering": "rendering_audio/tinystories.py",
    "tinystories_chinese_inference": "inference_audio/tinystories_chinese.py",
    "tinystories_chinese_rendering": "rendering_audio/tinystories_chinese.py",
    "gsm8k_inference": "inference_audio/gsm8k.py",
    "gsm8k_rendering": "rendering_audio/gsm8k.py",
}


def run_test(script_path: str, dataset_name: str, num_samples: int) -> dict:
    """Run a single test and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {dataset_name}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, script_path,
        "--base_dir", str(TEST_OUTPUT_DIR),
        "--num_samples", str(num_samples),
        "--num_workers", "2",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        success = result.returncode == 0
        
        # Count generated files
        video_dir = TEST_OUTPUT_DIR / f"{dataset_name}_audio" / "video"
        video_count = len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0
        
        jsonl_dir = TEST_OUTPUT_DIR / f"{dataset_name}_audio"
        jsonl_files = list(jsonl_dir.glob("*.jsonl")) if jsonl_dir.exists() else []
        
        # Count JSONL entries
        jsonl_entries = 0
        for jf in jsonl_files:
            with open(jf) as f:
                jsonl_entries += sum(1 for _ in f)
        
        return {
            "dataset": dataset_name,
            "success": success,
            "videos": video_count,
            "jsonl_entries": jsonl_entries,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "dataset": dataset_name,
            "success": False,
            "videos": 0,
            "jsonl_entries": 0,
            "error": "Timeout (600s exceeded)",
        }
    except Exception as e:
        return {
            "dataset": dataset_name,
            "success": False,
            "videos": 0,
            "jsonl_entries": 0,
            "error": str(e),
        }


def print_results(results: list):
    """Print test results summary."""
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'Status':<10} {'Videos':<10} {'JSONL Entries'}")
    print("-"*70)
    
    passed = 0
    failed = 0
    
    for r in results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"{r['dataset']:<20} {status:<10} {r['videos']:<10} {r['jsonl_entries']}")
        
        if r["success"]:
            passed += 1
        else:
            failed += 1
            if "error" in r:
                print(f"   Error: {r['error']}")
            elif r.get("stderr"):
                last_error = r["stderr"].strip().split("\n")[-1]
                print(f"   Error: {last_error[:60]}...")
    
    print("-"*70)
    print(f"Total: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test audio video generation for all datasets")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
                       help=f"Number of samples to test (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--clean", action="store_true",
                       help="Remove test output directory after testing")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Test specific dataset only")
    
    args = parser.parse_args()
    
    # Prepare test output directory
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to test
    if args.dataset:
        if args.dataset not in AUDIO_DATASETS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(AUDIO_DATASETS.keys())}")
            return 1
        datasets_to_test = {args.dataset: AUDIO_DATASETS[args.dataset]}
    else:
        datasets_to_test = AUDIO_DATASETS
    
    # Run tests
    results = []
    for dataset, script in datasets_to_test.items():
        result = run_test(script, dataset, args.num_samples)
        results.append(result)
    
    # Print summary
    all_passed = print_results(results)
    
    # Show generated videos
    print(f"\nTest outputs saved to: {TEST_OUTPUT_DIR}")
    for dataset in datasets_to_test:
        video_dir = TEST_OUTPUT_DIR / f"{dataset}_audio" / "video"
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                print(f"\n{dataset} videos:")
                for v in videos[:5]:
                    print(f"  - {v.name}")
    
    # Cleanup if requested
    if args.clean and TEST_OUTPUT_DIR.exists():
        print(f"\nCleaning up: Removing {TEST_OUTPUT_DIR}")
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
