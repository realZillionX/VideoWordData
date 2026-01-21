
import time
import os
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# IMPORT COMMON FIRST (Critical for environment vars)
import common.audio_video_utils
from common.audio_video_utils import create_video_with_audio_subtitles_fast

def worker_task(args):
    idx, text, output_file = args
    try:
        start_time = time.time()
        success = create_video_with_audio_subtitles_fast(
            text, 
            str(output_file), 
            language="en",
            target_duration=10.0
        )
        duration = time.time() - start_time
        return idx, success, duration, None
    except Exception as e:
        return idx, False, 0.0, str(e)

def profile_multiprocess():
    print("Starting MULTIPROCESS profiling...")
    
    num_samples = 20
    num_workers = 4  # Test concurrency
    
    output_dir = Path("profile_mp_output")
    output_dir.mkdir(exist_ok=True)
    
    tasks = []
    text = "Once upon a time there was a test. This is a story about a fox and a dog. They generated videos very fast using pipes."
    
    for i in range(num_samples):
        tasks.append((i, text, output_dir / f"test_{i}.mp4"))
    
    print(f"Processing {num_samples} videos with {num_workers} workers...")
    
    start_global = time.time()
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(worker_task, tasks), total=len(tasks)))
        
    total_time = time.time() - start_global
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Throughput: {num_samples / total_time:.2f} videos/sec")
    
    failures = [r for r in results if not r[1]]
    if failures:
        print(f"FAILURES: {len(failures)}/{num_samples}")
        for f in failures[:5]:
            print(f"Rank {f[0]} error: {f[3]}")
    else:
        print("SUCCESS! All videos generated.")

if __name__ == "__main__":
    profile_multiprocess()
