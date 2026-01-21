"""
TinyStories Dataset to Audio Video Conversion Script
Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Inference Task with Audio (Prompt WITHOUT Content in JSONL)

Features:
- Fixed 10-second videos
- Multiple videos per data sample (different sentence breakpoints)
- Only subtitle display (no prompt text in video)
- TTS narration with synchronized subtitles

Video naming: tinystories_audio_{data_idx}_{video_idx}.mp4
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import common utils FIRST to ensure OMP_NUM_THREADS env vars are set 
# and ffmpeg/piper configurations are loaded before heavy libraries (numpy/datasets) initialize.
from common.audio_video_utils import (
    create_video_with_audio_subtitles_fast as create_video_with_audio_subtitles,
    estimate_audio_duration,
    split_text_into_sentences,
    find_sentences_within_duration,
    AUDIO_VIDEO_DURATION,
)

from datasets import load_dataset
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os
import tiktoken

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_VIDEOS_PER_SAMPLE = 3
NUM_INTRO_SENTENCES = 2  # Sentences to use as context/prompt only (not in video)


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_into_sentences(text):
    """Split text into sentences"""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_video_segments(story_text, language="en"):
    """
    Generate multiple video segments from a story.
    
    Each segment:
    - Uses different starting points in the story
    - Contains sentences that fit within 10 seconds
    
    Returns list of (prompt_sentences, spoken_sentences, video_idx)
    """
    sentences = split_into_sentences(story_text)
    
    # Need at least (Intro + 1) sentences to generate a video
    if len(sentences) < NUM_INTRO_SENTENCES + 1:
        # Fallback: if at least 2 sentences, use 1 as intro
        if len(sentences) >= 2:
            start_idx = 1
        else:
            return []
    else:
        start_idx = NUM_INTRO_SENTENCES
    
    segments = []
    video_idx = 0
    
    # Strategy: Create videos with different starting points
    # Video 0: Start from sentence 0, speak sentences that fit in 10s
    # Video 1: Start from where video 0 ended, etc.
    
    current_start = start_idx
    
    while current_start < len(sentences) and video_idx < MAX_VIDEOS_PER_SAMPLE:
        remaining_sentences = sentences[current_start:]
        
        # Find how many sentences fit in 10 seconds
        num_fit = find_sentences_within_duration(
            remaining_sentences, 
            AUDIO_VIDEO_DURATION, 
            language
        )
        
        if num_fit == 0:
            # Even first sentence is too long, skip
            current_start += 1
            continue
        
        # Prompt: sentences before the spoken part (for JSONL, not shown in video)
        prompt_sentences = sentences[:current_start]
        
        # Spoken sentences (will be narrated and shown as subtitles)
        spoken_sentences = remaining_sentences[:num_fit]
        
        if len(spoken_sentences) > 0:
            segments.append((prompt_sentences, spoken_sentences, video_idx))
            video_idx += 1
        
        # Move to next segment
        current_start += num_fit
    
    return segments


def generate_jsonl_entry(video_path, prompt_text, spoken_text):
    """Generate a JSONL entry for inference task"""
    
    visual_desc = (
        f"A 10-second video with TTS narration and synchronized subtitles. "
        f"White background with black subtitles at the bottom area."
    )
    
    audio_desc = f"TTS audio narration, 10 seconds."
    
    # Inference: prompt does NOT contain the spoken content
    prompt = (
        f"Story context: \"{prompt_text}\" " if prompt_text else ""
    ) + "The video shows the story continuation with synchronized subtitles."
    
    return {
        "video_path": str(video_path),
        "visual_description": visual_desc,
        "speech_description": spoken_text,
        "audio_description": audio_desc,
        "prompt": prompt
    }


def process_sample(args):
    """Worker function to process a single video segment"""
    prompt_text, spoken_text, video_path, language = args
    
    if Path(video_path).exists():
        entry = generate_jsonl_entry(video_path, prompt_text, spoken_text)
        return entry
    
    try:
        success = create_video_with_audio_subtitles(
            spoken_text, str(video_path), language=language
        )
        if not success:
            return None
        
        entry = generate_jsonl_entry(video_path, prompt_text, spoken_text)
        return entry
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function"""
    
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "tinystories_audio"
    VIDEO_DIR = DATASET_DIR / "video"
    
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    TASK_JSONL_PATH = DATASET_DIR / f"tinystories_inference_audio_video_data_{start_idx}.jsonl"
    
    if num_workers is None:
        num_workers = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())))
    
    # Load dataset
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
    
    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    language = "en"
    all_samples = []
    
    print("Generating video segments...")
    for idx, sample in enumerate(tqdm(dataset, desc="Analyzing samples")):
        story_text = sample.get("text", "")
        
        if len(story_text.strip()) < 20:
            continue
        
        segments = generate_video_segments(story_text, language)
        
        for prompt_sentences, spoken_sentences, video_idx in segments:
            prompt_text = ' '.join(prompt_sentences)
            spoken_text = ' '.join(spoken_sentences)
            
            video_filename = f"tinystories_audio_{start_idx + idx:06d}_{video_idx:02d}.mp4"
            video_path = VIDEO_DIR / video_filename
            
            all_samples.append((prompt_text, spoken_text, video_path, language))
    
    print(f"Processing {len(all_samples)} video segments with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        with open(TASK_JSONL_PATH, 'w', encoding='utf-8') as jsonl_file:
            for entry in tqdm(pool.imap(process_sample, all_samples), total=len(all_samples)):
                if entry:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
    
    print(f"Processed {len(all_samples)} videos.")
    print(f"Output directory: {DATASET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TinyStories Audio Video dataset (Inference)")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    
    args = parser.parse_args()
    main(base_dir=args.base_dir, num_samples=args.num_samples, start_idx=args.start_idx, 
         num_workers=args.num_workers)
