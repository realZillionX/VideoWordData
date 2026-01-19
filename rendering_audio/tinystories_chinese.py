"""
TinyStories Chinese Dataset to Audio Video Conversion Script
Dataset: https://huggingface.co/datasets/adam89/TinyStoriesChinese

Rendering Task with Audio (Prompt includes full content in JSONL)

Features:
- Fixed 10-second videos
- Multiple videos per data sample
- Only subtitle display
- TTS narration with synchronized subtitles (Chinese)

Video naming: tinystories_chinese_audio_{data_idx}_{video_idx}.mp4
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os
import tiktoken

from common.audio_video_utils import (
    create_video_with_audio_subtitles,
    estimate_audio_duration,
    split_text_into_sentences,
    find_sentences_within_duration,
    AUDIO_VIDEO_DURATION,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_VIDEOS_PER_SAMPLE = 3


def count_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_into_sentences_chinese(text):
    """Split Chinese text into sentences"""
    sentence_pattern = r'(?<=[。！？；])\s*'
    sentences = re.split(sentence_pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def generate_video_segments(story_text, language="zh"):
    """
    Generate multiple video segments from a story.
    Each segment contains sentences that fit within 10 seconds.
    """
    sentences = split_into_sentences_chinese(story_text)
    
    if len(sentences) < 2:
        return []
    
    segments = []
    video_idx = 0
    current_start = 0
    
    while current_start < len(sentences) and video_idx < MAX_VIDEOS_PER_SAMPLE:
        remaining_sentences = sentences[current_start:]
        
        num_fit = find_sentences_within_duration(
            remaining_sentences, AUDIO_VIDEO_DURATION, language
        )
        
        if num_fit == 0:
            current_start += 1
            continue
        
        prompt_sentences = sentences[:current_start] if current_start > 0 else []
        spoken_sentences = remaining_sentences[:num_fit]
        
        if len(spoken_sentences) > 0:
            segments.append((prompt_sentences, spoken_sentences, video_idx))
            video_idx += 1
        
        current_start += num_fit
    
    return segments


def generate_jsonl_entry(video_path, prompt_text, spoken_text):
    """Generate a JSONL entry for rendering task (includes spoken content)"""
    
    visual_desc = "10秒视频，带TTS朗读和同步字幕。白色背景，底部黑色字幕。"
    audio_desc = "TTS音频朗读，10秒。"
    
    # Rendering: prompt INCLUDES the spoken content
    full_content = f"{prompt_text}{spoken_text}".strip() if prompt_text else spoken_text
    prompt = f"故事带字幕：\"{full_content}\""
    
    return {
        "video_path": str(video_path),
        "visual_description": visual_desc,
        "speech_description": spoken_text,
        "audio_description": audio_desc,
        "prompt": prompt
    }


def process_sample(args):
    """Worker function"""
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
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "tinystories_chinese_audio"
    VIDEO_DIR = DATASET_DIR / "video"
    
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    TASK_JSONL_PATH = DATASET_DIR / f"tinystories_chinese_rendering_audio_video_data_{start_idx}.jsonl"
    
    if num_workers is None:
        num_workers = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())) // 2)
    
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "tinystories_chinese" / "dataset"
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading TinyStories Chinese from local disk: {LOCAL_DATASET_PATH}")
            dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"✓ Loaded {len(dataset)} samples from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}, falling back to HuggingFace...")
            dataset = load_dataset("adam89/TinyStoriesChinese", split="train")
    else:
        print("Loading TinyStories Chinese dataset from HuggingFace...")
        dataset = load_dataset("adam89/TinyStoriesChinese", split="train")
    
    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    language = "zh"
    all_samples = []
    
    print("Generating video segments...")
    for idx, sample in enumerate(tqdm(dataset, desc="Analyzing samples")):
        story_text = sample.get("text", "")
        
        if len(story_text.strip()) < 10:
            continue
        
        segments = generate_video_segments(story_text, language)
        
        for prompt_sentences, spoken_sentences, video_idx in segments:
            prompt_text = ''.join(prompt_sentences)
            spoken_text = ''.join(spoken_sentences)
            
            video_filename = f"tinystories_chinese_audio_{start_idx + idx:06d}_{video_idx:02d}.mp4"
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
    parser = argparse.ArgumentParser(description="Generate TinyStories Chinese Audio Video dataset (Rendering)")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    
    args = parser.parse_args()
    main(base_dir=args.base_dir, num_samples=args.num_samples, start_idx=args.start_idx, 
         num_workers=args.num_workers)
