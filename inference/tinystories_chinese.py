"""
TinyStories Chinese Dataset to Video Conversion Script
Dataset: https://huggingface.co/datasets/adam89/TinyStoriesChinese

Inference Task (Prompt WITHOUT Continuation)

JSONL Format:
{
    "video_path": "", 
    "visual_description": "", 
    "speech_description": "", 
    "audio_description": "", 
    "prompt": "Story beginning only (Chinese)"
}

Dataset Format:
{
    "story": "English story",
    "story_zh": "Chinese translation",
    "instruction": {...},
    "summary": "..."
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
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os
import tiktoken

from common.video_utils import (
    create_video_with_gradual_text,
    add_sentence_newlines,
    get_font_path,
    get_chinese_font_path,
    VIDEO_DURATION,
)

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Number of intro sentences (displayed above separator)
NUM_INTRO_SENTENCES = 2


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def convert_quotes_to_chinese(text):
    """
    Convert English straight quotes to Chinese curly quotes in pairs.
    Odd-numbered " becomes " (U+201C, left/opening quote)
    Even-numbered " becomes " (U+201D, right/closing quote)
    """
    result = []
    quote_count = 0
    
    for char in text:
        if char == '"':
            quote_count += 1
            if quote_count % 2 == 1:
                result.append('\u201c')  # Opening quote (")
            else:
                result.append('\u201d')  # Closing quote (")
        else:
            result.append(char)
    
    return ''.join(result)


def clean_dialogue_formatting(text):
    """
    Clean up dialogue formatting to prevent closing quotes from starting new lines incorrectly.
    """
    import re
    
    # Only remove newlines that appear BEFORE a closing quote
    # Pattern: text + newline(s) + closing quote -> text + closing quote
    text = re.sub(r'\n+\s*\u201d', '\u201d', text)
    
    return text


def split_chinese_story(story_text, num_intro_sentences=NUM_INTRO_SENTENCES):
    """
    Split Chinese story into intro (prompt) and continuation (response).
    Uses Chinese sentence endings to split.
    """
    # Clean up whitespace
    story_text = re.sub(r'[ \t]+', ' ', story_text).strip()
    
    # Split by Chinese sentence endings (。！？) or English ones
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[。！？.!?])', story_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= num_intro_sentences:
        # Story too short, use first half as intro
        mid = max(1, len(sentences) // 2)
        intro_sentences = sentences[:mid]
        continuation_sentences = sentences[mid:]
    else:
        intro_sentences = sentences[:num_intro_sentences]
        continuation_sentences = sentences[num_intro_sentences:]
    
    # Join sentences
    intro_text = ''.join(intro_sentences)
    continuation_text = ''.join(continuation_sentences)
    
    # Add newlines after sentences for readability, but NOT if followed by close quote
    # This prevents close quote from appearing at line start
    # Matches sentence ending NOT followed by close quote (U+201D)
    # Add newlines after sentences for readable display
    # 1. Do NOT break if followed by closing quote (handled by step 2)
    intro_text = re.sub(r'([。！？.!?])(?!\u201d)', r'\1\n', intro_text)
    continuation_text = re.sub(r'([。！？.!?])(?!\u201d)', r'\1\n', continuation_text)
    
    # 2. explicit break AFTER closing quote if it follows punctuation
    intro_text = re.sub(r'([。！？.!?])\u201d', r'\1”\n', intro_text).strip()
    continuation_text = re.sub(r'([。！？.!?])\u201d', r'\1”\n', continuation_text).strip()
    
    return intro_text, continuation_text


def generate_jsonl_entry(video_path, prompt_text):
    """Generate a JSONL entry for inference task (prompt without continuation)"""
    
    visual_desc = (
        f"一个教育视频帧，显示中文故事续写任务。"
        f"故事开头以黑色文字显示在顶部，下方有分割线。"
        f"故事续写在{VIDEO_DURATION}秒内逐词显示，背景为白色。"
    )
    
    audio_desc = "无声视频。"
    
    # Inference prompt - does NOT include the continuation
    prompt = f"中文故事视频。故事开头：{prompt_text}"
    
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
        # Use Chinese font for Chinese text
        success = create_video_with_gradual_text(
            prompt_text, response_text, video_path, 
            font_path=get_chinese_font_path()
        )
        if not success:
            return None
        
    entry = generate_jsonl_entry(video_path, prompt_text)
    return entry


def main(base_dir=None, num_samples=None, start_idx=0, num_workers=None):
    """Main function with options for partial generation"""
    
    # Configuration
    if base_dir is None:
        base_dir = "/inspire/hdd/project/embodied-multimodality/public/textcentric"
    
    BASE_DIR = Path(base_dir)
    DATASET_DIR = BASE_DIR / "tinystories_chinese"
    VIDEO_DIR = DATASET_DIR / "video"
    
    # Ensure directories exist
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"tinystories_chinese_inference_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load dataset (WebDataset format with jsonl field)
    LOCAL_DATASET_BASE = "/inspire/hdd/project/embodied-multimodality/public"
    LOCAL_DATASET_PATH = Path(LOCAL_DATASET_BASE) / "tinystories_chinese" / "dataset"
    
    stories = []
    total_to_process = num_samples if num_samples else 100000  # Limit for memory
    
    if LOCAL_DATASET_PATH.exists():
        try:
            from datasets import load_from_disk
            print(f"Loading TinyStories Chinese from local disk: {LOCAL_DATASET_PATH}")
            raw_dataset = load_from_disk(str(LOCAL_DATASET_PATH))
            if isinstance(raw_dataset, dict):
                raw_dataset = raw_dataset["train"]
            
            # Parse jsonl field from each shard
            for shard in raw_dataset:
                jsonl_content = shard.get("jsonl", b"")
                if isinstance(jsonl_content, bytes):
                    jsonl_content = jsonl_content.decode("utf-8")
                
                for line in jsonl_content.strip().split("\n"):
                    if line.strip():
                        story_data = json.loads(line)
                        story_zh = story_data.get("story_zh", "")
                        if story_zh:
                            stories.append(story_zh)
                            if len(stories) >= start_idx + total_to_process:
                                break
                if len(stories) >= start_idx + total_to_process:
                    break
            
            print(f"✓ Loaded {len(stories)} stories from local disk")
        except Exception as e:
            print(f"Failed to load from local disk: {e}")
            stories = []
    
    if not stories:
        print("Loading TinyStories Chinese dataset from HuggingFace (streaming mode)...")
        raw_dataset = load_dataset("adam89/TinyStoriesChinese", split="train", streaming=True)
        
        for shard in raw_dataset:
            jsonl_content = shard.get("jsonl", b"")
            if isinstance(jsonl_content, bytes):
                jsonl_content = jsonl_content.decode("utf-8")
            
            for line in jsonl_content.strip().split("\n"):
                if line.strip():
                    story_data = json.loads(line)
                    story_zh = story_data.get("story_zh", "")
                    if story_zh:
                        stories.append(story_zh)
                        if len(stories) >= start_idx + total_to_process:
                            break
            if len(stories) >= start_idx + total_to_process:
                break
        
        print(f"✓ Loaded {len(stories)} stories from HuggingFace")
    
    # Apply start_idx
    if start_idx > 0:
        stories = stories[start_idx:]
    
    if num_samples:
        stories = stories[:num_samples]
    
    sample_counter = 0
    
    # Token statistics
    total_prompt_tokens = 0
    total_response_tokens = 0
    
    # Collect samples for processing
    all_samples = []
    
    for idx, story_text in enumerate(stories):
        if not story_text or len(story_text) < 20:
            continue
        
        # Convert English quotes to proper Chinese quotes
        story_text = convert_quotes_to_chinese(story_text)
        # Clean up dialogue formatting to prevent close quotes at line start
        story_text = clean_dialogue_formatting(story_text)
        
        prompt_text, response_text = split_chinese_story(story_text)
        
        # Skip if prompt or response is too short
        if len(prompt_text.strip()) < 5 or len(response_text.strip()) < 5:
            continue
        
        # Truncate if response is too long (more than 192 chars)
        # We try to truncate at the last sentence ending if possible
        response_text_clean = response_text.replace('\n', '').replace(' ', '')
        if len(response_text_clean) > 192:
            # We need to act on the original response_text which might have newlines
            # But the limitation is on "words/chars to reveal".
            # Simplest approach: truncate original text to length 192
            # But we want to preserve sentence structure if possible.
            
            # Find a cut point
            limit = 192
            if len(response_text) > limit:
                candidate = response_text[:limit]
                # Try to find last punctuation
                last_punct_idx = -1
                for punct in ['。', '！', '？', '!', '?', '.']:
                    p_idx = candidate.rfind(punct)
                    if p_idx > last_punct_idx:
                        last_punct_idx = p_idx
                
                if last_punct_idx > 150: # If we don't lose too much content (arbitrary threshold)
                    # Check if followed by closing quote
                    end_idx = last_punct_idx + 1
                    if end_idx < len(candidate) and candidate[end_idx] == '\u201d':
                        end_idx += 1
                    response_text = candidate[:end_idx]
                else:
                    # Hard truncate if no good punctuation found near end
                    response_text = candidate

        video_filename = f"tinystories_chinese_{start_idx + idx:06d}.mp4"
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
                if entry:
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
    
    total_tokens = total_prompt_tokens + total_response_tokens
    print(f"Processed {sample_counter} videos. Total tokens: {total_tokens} (prompt: {total_prompt_tokens}, response: {total_response_tokens})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TinyStories Chinese video dataset (Inference)")
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
