"""
GSM8K Chinese Dataset to Video Conversion Script
Dataset: https://huggingface.co/datasets/swulling/gsm8k_chinese

Rendering Task (Prompt INCLUDES Answer)

JSONL Format:
{
    "video_path": "", 
    "visual_description": "", 
    "speech_description": "", 
    "audio_description": "", 
    "prompt": "Question + Answer (Chinese)"
}

Dataset Format:
{
    "question": "Question",
    "answer": "Answer"
}
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os
import tiktoken

from common.video_utils import (
    create_video_with_gradual_text,
    add_sentence_newlines,
    get_font_path,
    VIDEO_DURATION,
)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Directories will be created in main()

# Ensure directories exist
# Directories will be created in main()


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def clean_chinese_answer(text):
    """Clean up Chinese GSM8K answer format:
    1. Remove <<...>> calculation annotations if present
    2. Replace #### with '答案是：'
    3. Add period at the end if missing
    4. Add newline after each sentence
    """
    # Remove <<...>> patterns (calculation annotations)
    text = re.sub(r'<<[^>]*>>', '', text)
    
    # Replace #### with '答案是：'
    text = re.sub(r'####\s*', '答案是：', text)
    
    # Clean up extra whitespace (but preserve existing newlines)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    # Add period at the end if missing (use Chinese period)
    if text and text[-1] not in '.!?。！？':
        text += '。'
    
    # Add newline after each sentence (Chinese and English punctuation)
    text = re.sub(r'([.!?。！？])\s*', r'\1\n', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n+', '\n', text).strip()
    
    return text


def clean_chinese_question(text):
    """Clean up Chinese question format and add newline after each sentence"""
    # Clean up extra whitespace (but preserve existing newlines)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    # Add newline after each sentence (Chinese and English punctuation)
    text = re.sub(r'([.!?。！？])\s*', r'\1\n', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n+', '\n', text).strip()
    
    return text


def generate_jsonl_entry(video_path, prompt_text, response_text):
    """Generate a JSONL entry for rendering task (prompt includes answer)"""
    
    visual_desc = (
        f"一个教育视频帧，显示中文数学题解答任务。"
        f"问题以黑色文字显示在顶部，下方有分割线。"
        f"答案在{VIDEO_DURATION}秒内逐词显示，背景为白色。"
    )
    
    audio_desc = "无声视频。"
    
    # Rendering prompt - includes BOTH question AND answer
    prompt = f"中文数学题视频。问题：{prompt_text} 答案：{response_text}"
    
    return {
        "video_path": str(video_path),
        "visual_description": visual_desc,
        "speech_description": "",
        "audio_description": audio_desc,
        "prompt": prompt
    }


def process_sample(args):
    """Worker function to process a single sample (for multiprocessing)"""
    prompt_text, response_text, video_path = args
    
    # Check if video already exists to avoid re-generation
    if not Path(video_path).exists():
        create_video_with_gradual_text(prompt_text, response_text, video_path)
        
    entry = generate_jsonl_entry(video_path, prompt_text, response_text)
    return entry


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function with options for partial generation"""
    
    # Configuration
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "gsm8k_chinese"
    VIDEO_DIR = DATASET_DIR / "video"
    
    # Ensure directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"gsm8k_chinese_rendering_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load dataset (local-first strategy)
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "gsm8k_chinese" / "dataset"
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading GSM8K Chinese from local disk: {LOCAL_DATASET_PATH}")
            dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}, falling back to HuggingFace...")
            dataset = load_dataset("swulling/gsm8k_chinese", split="train")
    else:
        print("Loading GSM8K Chinese dataset from HuggingFace...")
        dataset = load_dataset("swulling/gsm8k_chinese", split="train")
    
    # Apply start_idx and num_samples
    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    sample_counter = 0
    
    # Token statistics
    total_prompt_tokens = 0
    total_response_tokens = 0
    
    # Collect samples for processing
    all_samples = []
    
    for idx, sample in enumerate(dataset):
        # Extract question and answer
        prompt_text = clean_chinese_question(sample.get("question", ""))
        response_text = clean_chinese_answer(sample.get("answer", ""))
        
        # Skip if prompt or response is too short
        if len(prompt_text.strip()) < 5 or len(response_text.strip()) < 5:
            continue
        
        # Skip if response is too long (for Chinese, count characters)
        if len(response_text.replace('\n', '').replace(' ', '')) > 300:
            continue
        
        video_filename = f"gsm8k_chinese_{start_idx + idx:06d}.mp4"
        video_path = VIDEO_DIR / video_filename
        all_samples.append((prompt_text, response_text, video_path))
        
        # Count tokens
        total_prompt_tokens += count_tokens(prompt_text)
        total_response_tokens += count_tokens(response_text)
        
        sample_counter += 1
    
    print(f"Processing {sample_counter} samples with {num_workers} workers...")
    
    # Process samples in parallel
    with Pool(processes=num_workers) as pool:
        with open(TASK_JSONL_PATH, 'w', encoding='utf-8') as jsonl_file:
            for entry in tqdm(pool.imap(process_sample, all_samples), total=len(all_samples)):
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                jsonl_file.flush()
    
    total_tokens = total_prompt_tokens + total_response_tokens
    print(f"Processed {sample_counter} videos. Total tokens: {total_tokens} (prompt: {total_prompt_tokens}, response: {total_response_tokens})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GSM8K Chinese video dataset (Rendering)")
    parser.add_argument("--base_dir", type=str, default=None,
                       help="Base directory for output (default: /inspire/hdd/project/embodied-multimodality/public/textcentric)")
    parser.add_argument("--num_samples", type=int, default=None, 
                       help="Number of samples to process (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting sample index (default: 0)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect from SLURM or CPU count)")
    
    args = parser.parse_args()
    main(base_dir=args.base_dir, num_samples=args.num_samples, start_idx=args.start_idx, 
         num_workers=args.num_workers)
