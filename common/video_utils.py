"""
Common video generation utilities for VideoWordData.
Shared functions for creating videos with gradual text display.
"""

import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path

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
PROMPT_FONT_SIZE = 28
RESPONSE_FONT_SIZE = 28
CHARS_PER_LINE = 38  # Adjusted for monospace font


def get_font_path():
    """Get the path to the font file"""
    return Path(__file__).resolve().parent.parent / "fonts" / "DejaVuSansMono.ttf"


def add_sentence_newlines(text):
    """Add newline after each sentence for better readability"""
    # Clean up extra whitespace (but preserve existing newlines)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    
    # Add newline after each sentence (. ! ? followed by space)
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    
    return text


def create_video_with_gradual_text(prompt_text, response_text, output_path, font_path=None):
    """
    Create video with gradual text display and page-turning.
    
    Args:
        prompt_text: Text to display above separator (question/intro)
        response_text: Text to display below separator, revealed word by word
        output_path: Path to save the video file
        font_path: Optional path to font file (uses default if not provided)
    
    Returns:
        True if video was created, False if skipped (already exists)
    """
    if font_path is None:
        font_path = get_font_path()
    
    # Split response into tokens (words and newlines)
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
        p_font = ImageFont.truetype(str(font_path), PROMPT_FONT_SIZE)
        r_font = ImageFont.truetype(str(font_path), RESPONSE_FONT_SIZE)
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
    return True
