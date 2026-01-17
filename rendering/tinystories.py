"""
TinyStories Dataset to Video Conversion Script
Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Rendering Task (Prompt INCLUDES Continuation)

JSONL Format:
{
    "video_path": "", 
    "visual_description": "", 
    "speech_description": "", 
    "audio_description": "", 
    "prompt": "Story beginning + Continuation"
}

Dataset Format:
{
    "text": "Once upon a time... full story text"
}

For story datasets:
- First few sentences are displayed above the separator line
- Remaining sentences are displayed below the separator line (revealed word by word)
"""

import json
import re
import sys
from pathlib import Path

# Add parent directory to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from common.video_utils import (
    create_video_with_gradual_text,
    add_sentence_newlines,
    VIDEO_DURATION,
)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# How many sentences to show above the separator
NUM_INTRO_SENTENCES = 2


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_into_sentences(text):
    """Split text into sentences using common sentence delimiters"""
    # Simple sentence splitting: split on . ! ? followed by space or end
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_story(text, num_intro_sentences=NUM_INTRO_SENTENCES):
    """
    Split story into intro (above separator) and continuation (below separator).
    Returns (intro_text, continuation_text)
    """
    sentences = split_into_sentences(text)
    
    if len(sentences) <= num_intro_sentences:
        # If too few sentences, put first sentence above, rest below
        if len(sentences) == 1:
            return sentences[0], ""
        else:
            return sentences[0], ' '.join(sentences[1:])
    
    intro_sentences = sentences[:num_intro_sentences]
    continuation_sentences = sentences[num_intro_sentences:]
    
    intro_text = ' '.join(intro_sentences)
    continuation_text = ' '.join(continuation_sentences)
    
    # Add sentence newlines to both intro and continuation for better readability
    intro_text = add_sentence_newlines(intro_text)
    continuation_text = add_sentence_newlines(continuation_text)
    
    return intro_text, continuation_text


def generate_jsonl_entry(video_path, prompt_text, response_text):
    """Generate a JSONL entry for rendering task (prompt includes full story)"""
    
    visual_desc = (
        f"A static educational video frame showing a story reading task. "
        f"The story beginning appears in black text at the top, followed by a horizontal separator. "
        f"Below, the story continuation gradually reveals word by word in black text over {VIDEO_DURATION} seconds, "
        f"on a clean white background."
    )
    
    audio_desc = "Silent video with no audio content."
    
    # Rendering prompt - includes BOTH story beginning AND continuation
    prompt = (
        f"A story video on white background. "
        f"Story beginning: \"{prompt_text}\" "
        f"Story continuation: \"{response_text}\""
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
    DATASET_DIR = BASE_DIR / "tinystories"
    VIDEO_DIR = DATASET_DIR / "video"
    
    # Ensure directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"tinystories_rendering_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load dataset (local-first strategy)
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "tinystories" / "dataset"
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading TinyStories from local disk: {LOCAL_DATASET_PATH}")
            dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}, falling back to HuggingFace...")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
    else:
        print("Loading TinyStories dataset from HuggingFace...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
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
        # Extract story text
        story_text = sample.get("text", "")
        
        # Skip if story is too short
        if len(story_text.strip()) < 20:
            continue
        
        # Split story into intro and continuation
        prompt_text, response_text = split_story(story_text)
        
        # Skip if either part is too short
        if len(prompt_text.strip()) < 5 or len(response_text.strip()) < 5:
            continue
        
        # Truncate if continuation is too long (more than 192 words)
        words = response_text.split()
        if len(words) > 192:
            # Smart truncation
            limit = 192
            truncated_words = words[:limit]
            
            # Find closest sentence end
            last_punct_idx = -1
            for i in range(len(truncated_words) - 1, -1, -1):
                word = truncated_words[i]
                if word.endswith(('.', '!', '?')):
                    last_punct_idx = i
                    break
                # Check for sentence endings inside quotes
                if len(word) > 1 and word[:-1].endswith(('.', '!', '?')) and word.endswith(('"', "'", "”")):
                    last_punct_idx = i
                    break
            
            if last_punct_idx > 150:
                response_text = ' '.join(truncated_words[:last_punct_idx+1])
            else:
                response_text = ' '.join(truncated_words)
        
        video_filename = f"tinystories_{start_idx + idx:06d}.mp4"
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
    parser = argparse.ArgumentParser(description="Generate TinyStories video dataset (Rendering)")
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


def generate_jsonl_entry(video_path, prompt_text, response_text):
    """Generate a JSONL entry for rendering task (prompt includes full story)"""
    
    visual_desc = (
        f"A static educational video frame showing a story reading task. "
        f"The story beginning appears in black text at the top, followed by a horizontal separator. "
        f"Below, the story continuation gradually reveals word by word in black text over {VIDEO_DURATION} seconds, "
        f"on a clean white background."
    )
    
    audio_desc = "Silent video with no audio content."
    
    # Rendering prompt - includes BOTH story beginning AND continuation
    prompt = (
        f"A story video on white background. "
        f"Story beginning: \"{prompt_text}\" "
        f"Story continuation: \"{response_text}\""
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
        create_video_with_gradual_text(prompt_text, response_text, video_path)
        
    entry = generate_jsonl_entry(video_path, prompt_text, response_text)
    return entry


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function with options for partial generation"""
    
    # Configuration
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "tinystories"
    VIDEO_DIR = DATASET_DIR / "video"
    
    # Ensure directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"tinystories_rendering_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load dataset (local-first strategy)
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "tinystories" / "dataset"
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading TinyStories from local disk: {LOCAL_DATASET_PATH}")
            dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}, falling back to HuggingFace...")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
    else:
        print("Loading TinyStories dataset from HuggingFace...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
    
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
        # Extract story text
        story_text = sample.get("text", "")
        
        # Skip if story is too short
        if len(story_text.strip()) < 20:
            continue
        
        # Split story into intro and continuation
        prompt_text, response_text = split_story(story_text)
        
        # Skip if either part is too short
        if len(prompt_text.strip()) < 5 or len(response_text.strip()) < 5:
            continue
        
        # Skip if continuation is too long (more than 192 words, since we have 193 frames and first frame is prompt only)
        if len(response_text.split()) > 192:
            continue
        
        video_filename = f"tinystories_{start_idx + idx:06d}.mp4"
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
    parser = argparse.ArgumentParser(description="Generate TinyStories video dataset (Rendering)")
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
