"""
JSONL Format:
{
    "video_path": "", 
    "visual_description": "", 
    "speech_description": "", 
    "audio_description": "", 
    "prompt": ""
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

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = Path("/inspire/hdd/project/embodied-multimodality/public/textcentric")
DATASET_DIR = BASE_DIR / "gsm8k"
VIDEO_DIR = DATASET_DIR / "video"

# Video settings - Strict format requirements
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
FPS = 193 / 10  # 193 frames in 10 seconds = 19.3 fps
VIDEO_DURATION = 10
TOTAL_FRAMES = 193
BACKGROUND_COLOR = (255, 255, 255)
PROMPT_COLOR = (0, 0, 0)
RESPONSE_COLOR = (0, 0, 0)

# Font settings - Larger fonts for better readability
FONT_PATH = PROJECT_ROOT / "fonts" / "DejaVuSans.ttf"
PROMPT_FONT_SIZE = 24
RESPONSE_FONT_SIZE = 24
CHARS_PER_LINE = 40  # Adjusted for larger font

# Ensure directories exist
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def count_tokens(text, encoding_name="cl100k_base"):
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


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


def create_video_with_gradual_text(prompt_text, response_text, output_path):
    """Video with true page-turning: accumulate content on each page until full, then turn page - 193 frames, 10 seconds, 360P"""
    
    # Split response into tokens (words and newlines)
    import re
    tokens = re.split(r'(\n+)', response_text)
    response_parts = []  # List of (word, newline_after)
    for token in tokens:
        if token.startswith('\n'):
            # Mark last word as having newlines after it
            if response_parts:
                response_parts[-1] = (response_parts[-1][0], token)
        else:
            # Split into words
            for word in token.split():
                response_parts.append((word, ''))
    
    total_words = len(response_parts)

    # Calculate frame counts - exactly 193 frames, one word per frame
    prompt_only_frames = 1  # First frame shows full prompt
    response_frames = TOTAL_FRAMES - prompt_only_frames
    
    # Pre-calculate which words to show at each frame (one word per frame)
    words_at_frame = []
    for frame_num in range(TOTAL_FRAMES):
        if frame_num < prompt_only_frames:
            words_at_frame.append(0)
        else:
            # Each frame adds exactly one word
            words_to_show = min(frame_num - prompt_only_frames + 1, total_words)
            words_at_frame.append(words_to_show)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    # Load font
    try:
        p_font = ImageFont.truetype(str(FONT_PATH), PROMPT_FONT_SIZE)
        r_font = ImageFont.truetype(str(FONT_PATH), RESPONSE_FONT_SIZE)
    except:
        p_font = ImageFont.load_default()
        r_font = ImageFont.load_default()
    
    # Layout parameters
    margin = 15
    line_height = RESPONSE_FONT_SIZE + 4
    max_y = VIDEO_HEIGHT - margin
    
    # Wrap prompt lines (preserve newlines)
    prompt_lines = []
    for line in prompt_text.split('\n'):
        if line.strip():
            wrapped = textwrap.wrap(line, width=CHARS_PER_LINE, break_long_words=False, break_on_hyphens=False)
            prompt_lines.extend(wrapped if wrapped else [''])
        else:
            prompt_lines.append('')
    
    # Track pagination state
    current_page_start_word = 0  # Which word starts current page
    is_first_page = True
    last_word_count = 0
    
    for frame_num in range(TOTAL_FRAMES):
        words_to_show = words_at_frame[frame_num]
        
        # Only recalculate when word count changes
        if words_to_show != last_word_count and words_to_show > 0:
            # Reconstruct text with newlines from current page start to words_to_show
            page_parts = response_parts[current_page_start_word:words_to_show]
            
            # Build text with newlines preserved
            text_with_newlines = []
            for word, newline_after in page_parts:
                text_with_newlines.append(word)
                if newline_after:
                    text_with_newlines.append(newline_after)
            
            # Join and split by newlines to get paragraphs
            page_text = ' '.join(text_with_newlines)
            paragraphs = page_text.split('\n')
            
            # Wrap each paragraph
            page_lines = []
            for para in paragraphs:
                para = para.strip()
                if para:
                    wrapped = textwrap.wrap(para, width=CHARS_PER_LINE, break_long_words=False, break_on_hyphens=False)
                    page_lines.extend(wrapped)
                else:
                    # Empty paragraph creates a blank line
                    if page_lines:  # Only add blank line if not at start
                        page_lines.append('')
            
            # Calculate required height
            y_test = margin
            
            if is_first_page:
                # First page includes prompt
                y_test += len(prompt_lines) * (PROMPT_FONT_SIZE + 2)
                y_test += 10  # Separator space
            
            # Check how many lines fit
            lines_that_fit = 0
            for line in page_lines:
                if y_test + line_height <= max_y:
                    lines_that_fit += 1
                    y_test += line_height
                else:
                    break
            
            # If not all lines fit, we need to turn the page
            if lines_that_fit < len(page_lines):
                # Calculate how many words fit on this page
                fitted_lines = page_lines[:lines_that_fit]
                fitted_text = ' '.join(fitted_lines)
                fitted_words = fitted_text.split()
                words_that_fit = len(fitted_words)
                
                # Turn page: next page starts after the words that fit
                current_page_start_word += words_that_fit
                is_first_page = False
            
            last_word_count = words_to_show
        
        # Render current page
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        y_pos = margin
        
        if words_to_show == 0:
            # Show only prompt
            for line in prompt_lines:
                draw.text((margin, y_pos), line, fill=PROMPT_COLOR, font=p_font)
                y_pos += PROMPT_FONT_SIZE + 2
        
        else:
            # Reconstruct current page content with newlines
            page_parts = response_parts[current_page_start_word:words_to_show]
            
            # Build text with newlines preserved
            text_with_newlines = []
            for word, newline_after in page_parts:
                text_with_newlines.append(word)
                if newline_after:
                    text_with_newlines.append(newline_after)
            
            # Join and split by newlines to get paragraphs
            page_text = ' '.join(text_with_newlines)
            paragraphs = page_text.split('\n')
            
            # Wrap each paragraph
            page_lines = []
            for para in paragraphs:
                para = para.strip()
                if para:
                    wrapped = textwrap.wrap(para, width=CHARS_PER_LINE, break_long_words=False, break_on_hyphens=False)
                    page_lines.extend(wrapped)
                else:
                    # Empty paragraph creates a blank line
                    if page_lines:
                        page_lines.append('')
            
            if is_first_page:
                # First page with prompt
                for line in prompt_lines:
                    draw.text((margin, y_pos), line, fill=PROMPT_COLOR, font=p_font)
                    y_pos += PROMPT_FONT_SIZE + 2
                
                y_pos += 5
                draw.line([(margin, y_pos), (VIDEO_WIDTH - margin, y_pos)], fill=(200, 200, 200), width=1)
                y_pos += 5
                
                # Render response lines
                for line in page_lines:
                    if y_pos + line_height <= max_y:
                        draw.text((margin, y_pos), line, fill=RESPONSE_COLOR, font=r_font)
                        y_pos += line_height
            else:
                # Response page (continued)
                for line in page_lines:
                    if y_pos + line_height <= max_y:
                        draw.text((margin, y_pos), line, fill=RESPONSE_COLOR, font=r_font)
                        y_pos += line_height
        
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def generate_jsonl_entry(video_path, prompt_text):
    """Generate a JSONL entry for prompt-response task"""
    
    visual_desc = (
        f"A static educational video frame showing a math problem solving task. "
        f"The prompt appears in black text at the top, followed by a horizontal separator. "
        f"Below, the response gradually reveals word by word in black text over {VIDEO_DURATION} seconds, "
        f"demonstrating the solution on a clean white background."
    )
    
    audio_desc = "Silent video with no audio content."
    
    prompt = (
        f"An educational video on white background. Prompt in black: "
        f"\"{prompt_text}\". "
        f"Response appears gradually in black, word by word over {VIDEO_DURATION} seconds."
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
        
    entry = generate_jsonl_entry(video_path, prompt_text)
    return entry


def main(num_samples=None, start_idx=0, num_workers=None):
    """Main function with options for partial generation"""
    
    # Use a unique JSONL file for each task to avoid concurrency issues
    TASK_JSONL_PATH = DATASET_DIR / f"gsm8k_video_data_{start_idx}.jsonl"
    
    # Determine number of workers
    if num_workers is None:
        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
    
    # Load GSM8K dataset from local path
    LOCAL_GSM8K_PATH = "/inspire/hdd/project/embodied-multimodality/public/gsm8k"
    
    try:
        # Try loading from local directory first
        dataset = load_dataset(LOCAL_GSM8K_PATH, "main", split="train")
        print(f"Loaded dataset from local path: {LOCAL_GSM8K_PATH}")
    except:
        try:
            # Try with JSON file in the directory
            dataset = load_dataset("json", data_files=str(LOCAL_GSM8K_PATH + "/*.jsonl"), split="train")
            print(f"Loaded dataset from JSON files in: {LOCAL_GSM8K_PATH}")
        except:
            # Fallback to HuggingFace if local loading fails
            print("Could not load from local path, trying HuggingFace...")
            dataset = load_dataset("gsm8k", "main", split="train")
    
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
        prompt_text = sample.get("question", "")
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
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                jsonl_file.flush()
    
    total_tokens = total_prompt_tokens + total_response_tokens
    print(f"Processed {sample_counter} videos. Total tokens: {total_tokens} (prompt: {total_prompt_tokens}, response: {total_response_tokens})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GSM8K video dataset")
    parser.add_argument("--num_samples", type=int, default=None, 
                       help="Number of samples to process (default: all)")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting sample index (default: 0)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect from SLURM or CPU count)")
    
    args = parser.parse_args()
    main(num_samples=args.num_samples, start_idx=args.start_idx, 
         num_workers=args.num_workers)