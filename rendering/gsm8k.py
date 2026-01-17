"""
GSM8K Dataset to Video Conversion Script
Dataset: https://huggingface.co/datasets/gsm8k

Rendering Task (Prompt INCLUDES Answer)

JSONL Format:
{
    "video_path": "", 
    "visual_description": "", 
    "speech_description": "", 
    "audio_description": "", 
    "prompt": "Question + Answer"
}
"""

import json
import re
from pathlib import Path
from datasets import load_dataset
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
import argparse
from multiprocessing import Pool, cpu_count
import os
import tiktoken
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.video_utils import create_video_with_gradual_text

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video settings - Strict format requirements
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
FPS = 193 / 10  # 193 frames in 10 seconds = 19.3 fps
VIDEO_DURATION = 10
TOTAL_FRAMES = 193
BACKGROUND_COLOR = (255, 255, 255)
PROMPT_COLOR = (0, 0, 0)
RESPONSE_COLOR = (0, 0, 0)

# Font settings - Monospace font for consistent character width
FONT_PATH = PROJECT_ROOT / "fonts" / "DejaVuSansMono.ttf"
PROMPT_FONT_SIZE = 28
RESPONSE_FONT_SIZE = 28
CHARS_PER_LINE = 38  # Adjusted for monospace font

# Ensure directories exist
# Directories will be created in main()


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def add_sentence_newlines(text):
    """Add newline after each sentence for better readability"""
    # Clean up extra whitespace (but preserve existing newlines)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    # Add newline after each sentence (. ! ? followed by space)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    return text


def clean_gsm8k_answer(text):
    """Clean up GSM8K answer format:
    1. Remove <<...>> calculation annotations
    2. Replace #### with 'So the answer is:'
    3. Add period at the end if missing
    4. Add newline after each sentence
    """
    # Remove <<...>> patterns (calculation annotations)
    text = re.sub(r'<<[^>]*>>', '', text)
    
    # Replace #### with 'So the answer is:'
    text = re.sub(r'####\s*', 'So the answer is: ', text)
    
    # Clean up extra whitespace (but preserve existing newlines)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    # Add period at the end if missing
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Add newline after each sentence (. ! ? followed by space)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    return text





def generate_jsonl_entry(video_path, prompt_text, response_text):
    """Generate a JSONL entry for rendering task (prompt includes answer)"""
    
    visual_desc = (
        f"A static educational video frame showing a math problem solving task. "
        f"The prompt appears in black text at the top, followed by a horizontal separator. "
        f"Below, the response gradually reveals word by word in black text over {VIDEO_DURATION} seconds, "
        f"demonstrating the solution on a clean white background."
    )
    
    audio_desc = "Silent video with no audio content."
    
    # Rendering prompt - includes BOTH question AND answer
    prompt = (
        f"An educational video on white background. "
        f"Question: \"{prompt_text}\" "
        f"Answer: \"{response_text}\""
    )
    
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
        success = create_video_with_gradual_text(prompt_text, response_text, video_path)
        if not success:
            return None
        
    entry = generate_jsonl_entry(video_path, prompt_text, response_text)
    return entry


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function with options for partial generation"""
    
    # Configuration
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "gsm8k"
    VIDEO_DIR = DATASET_DIR / "video"
    
    # Ensure directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"gsm8k_rendering_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load dataset (local-first strategy)
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "gsm8k" / "dataset"
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading GSM8K from local disk: {LOCAL_DATASET_PATH}")
            dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}, falling back to HuggingFace...")
            dataset = load_dataset("openai/gsm8k", "main", split="train")
    else:
        print("Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    
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
        prompt_text = add_sentence_newlines(sample.get("question", ""))
        response_text = clean_gsm8k_answer(sample.get("answer", ""))
        
        # Skip if prompt or response is too short
        if len(prompt_text.strip()) < 5 or len(response_text.strip()) < 5:
            continue
        
        # Skip if response is too long (more than 192 words, since we have 193 frames and first frame is prompt only)
        if len(response_text.split()) > 192:
            continue
        
        video_filename = f"gsm8k_{start_idx + idx:06d}.mp4"
        video_path = VIDEO_DIR / video_filename
        all_samples.append((prompt_text, response_text, video_path))
        
        # Count tokens
        total_prompt_tokens += count_tokens(prompt_text)
        total_response_tokens += count_tokens(response_text)
        
        sample_counter += 1
    
    print(f"Processing {sample_counter} samples with {num_workers} workers...")
    
    # Process samples in parallel
    # We process ALL samples to ensure JSONL is complete.
    # The process_sample function will skip video generation if file exists.
    with Pool(processes=num_workers) as pool:
        with open(TASK_JSONL_PATH, 'w', encoding='utf-8') as jsonl_file:
            for entry in tqdm(pool.imap(process_sample, all_samples), total=len(all_samples)):
                if entry:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
    
    total_tokens = total_prompt_tokens + total_response_tokens
    print(f"Processed {sample_counter} videos. Total tokens: {total_tokens} (prompt: {total_prompt_tokens}, response: {total_response_tokens})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GSM8K video dataset (Rendering)")
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