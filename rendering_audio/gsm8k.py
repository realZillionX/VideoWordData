"""
GSM8K Dataset to Audio Video Conversion Script
Dataset: https://huggingface.co/datasets/openai/gsm8k

Rendering Task with Audio (Prompt includes included Answer in JSONL)

Features:
- Fixed 10-second videos
- Answer must fit within 10 seconds, otherwise skip
- Only subtitle display
- TTS narration of the answer with synchronized subtitles

Video naming: gsm8k_audio_{idx}.mp4
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
    can_fit_in_duration,
    AUDIO_VIDEO_DURATION,
)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def add_sentence_newlines(text):
    """Add newline after each sentence for better readability"""
    text = re.sub(r'[ \t]+', ' ', text).strip()
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    return text


def clean_gsm8k_answer(text):
    """Clean up GSM8K answer format"""
    # Remove <<...>> calculation annotations
    text = re.sub(r'<<[^>]*>>', '', text)
    # Replace #### with 'So the answer is:'
    text = re.sub(r'####\s*', 'So the answer is: ', text)
    # Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()
    # Add period at the end if missing
    if text and text[-1] not in '.!?':
        text += '.'
    return text


def generate_jsonl_entry(video_path, question_text, answer_text):
    """Generate a JSONL entry for rendering task (includes answer)"""
    
    visual_desc = (
        f"A 10-second educational video with TTS narration of a math solution. "
        f"White background with synchronized subtitles at the bottom."
    )
    
    audio_desc = f"TTS audio narration of the answer, 10 seconds."
    
    # Rendering: prompt includes the question AND the answer
    full_content = f"Question: {question_text}\nAnswer: {answer_text}"
    prompt = f"Math problem with solution: \"{full_content}\""
    
    return {
        "video_path": str(video_path),
        "visual_description": visual_desc,
        "speech_description": answer_text,
        "audio_description": audio_desc,
        "prompt": prompt
    }


def process_sample(args):
    """Worker function to process a single sample"""
    question_text, answer_text, video_path, language = args
    
    if Path(video_path).exists():
        entry = generate_jsonl_entry(video_path, question_text, answer_text)
        return entry
    
    try:
        success = create_video_with_audio_subtitles(
            answer_text, str(video_path), language=language
        )
        if not success:
            return None
        
        entry = generate_jsonl_entry(video_path, question_text, answer_text)
        return entry
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function"""
    
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "gsm8k_audio"
    VIDEO_DIR = DATASET_DIR / "video"
    
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    TASK_JSONL_PATH = DATASET_DIR / f"gsm8k_rendering_audio_video_data_{start_idx}.jsonl"
    
    if num_workers is None:
        num_workers = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count())))
    
    # Load dataset
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
    
    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    language = "en"
    all_samples = []
    skipped_too_long = 0
    
    print("Filtering samples that fit in 10 seconds...")
    for idx, sample in enumerate(tqdm(dataset, desc="Analyzing samples")):
        question_text = add_sentence_newlines(sample.get("question", ""))
        answer_text = clean_gsm8k_answer(sample.get("answer", ""))
        
        if len(question_text.strip()) < 5 or len(answer_text.strip()) < 5:
            continue
        
        # Check if answer fits in 10 seconds
        if not can_fit_in_duration(answer_text, AUDIO_VIDEO_DURATION, language):
            skipped_too_long += 1
            continue
        
        video_filename = f"gsm8k_audio_{start_idx + idx:06d}.mp4"
        video_path = VIDEO_DIR / video_filename
        
        all_samples.append((question_text, answer_text, video_path, language))
    
    print(f"Found {len(all_samples)} valid samples (skipped {skipped_too_long} too long)")
    print(f"Processing with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        with open(TASK_JSONL_PATH, 'w', encoding='utf-8') as jsonl_file:
            for entry in tqdm(pool.imap(process_sample, all_samples), total=len(all_samples)):
                if entry:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
    
    print(f"Processed {len(all_samples)} videos.")
    print(f"Output directory: {DATASET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GSM8K Audio Video dataset (Rendering)")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    
    args = parser.parse_args()
    main(base_dir=args.base_dir, num_samples=args.num_samples, start_idx=args.start_idx, 
         num_workers=args.num_workers)
