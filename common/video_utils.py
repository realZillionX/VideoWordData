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

# Font settings
PROMPT_FONT_SIZE = 28
RESPONSE_FONT_SIZE = 28
CHARS_PER_LINE = 34  # For English text
CHINESE_CHARS_PER_LINE = 18  # For Chinese text (wider characters)


def get_font_path():
    """Get the path to the default (English) font file"""
    return Path(__file__).resolve().parent.parent / "fonts" / "DejaVuSansMono.ttf"


def get_chinese_font_path():
    """Get the path to the Chinese font file"""
    return Path(__file__).resolve().parent.parent / "fonts" / "DroidSansFallbackFull.ttf"


def is_chinese_char(char):
    """Check if a character is Chinese"""
    cp = ord(char)
    return (0x4E00 <= cp <= 0x9FFF or  # CJK Unified Ideographs
            0x3400 <= cp <= 0x4DBF or  # CJK Unified Ideographs Extension A
            0x20000 <= cp <= 0x2A6DF or  # CJK Unified Ideographs Extension B
            0x2A700 <= cp <= 0x2B73F or  # CJK Unified Ideographs Extension C
            0x2B740 <= cp <= 0x2B81F or  # CJK Unified Ideographs Extension D
            0x2B820 <= cp <= 0x2CEAF or  # CJK Unified Ideographs Extension E
            0xF900 <= cp <= 0xFAFF or  # CJK Compatibility Ideographs
            0x3000 <= cp <= 0x303F or  # CJK Punctuation
            0xFF00 <= cp <= 0xFFEF)  # Halfwidth and Fullwidth Forms


# Characters that should NOT appear at the beginning of a line (no break before)
NO_BREAK_BEFORE = set(
    '。，！？；：、）」』】》\"\'..,!?;:)]}>\'"'  # Closing punctuation
    '）】》〉」』'  # Chinese closing brackets
    '％‰'  # Percentage signs
    '\u201d'  # Right curly quote "
)

# Characters that should NOT appear at the end of a line (no break after)  
NO_BREAK_AFTER = set(
    '（「『【《\"\'.([{<\'"'  # Opening punctuation
    '（【《〈「『'  # Chinese opening brackets
    '¥$￥'  # Currency symbols
    '\u201c'  # Left curly quote "
)

# Math operators that should stay with adjacent numbers
MATH_OPERATORS = set('+-*/×÷=≠<>≤≥±')


def should_break_before(char):
    """Check if we can break line before this character"""
    return char not in NO_BREAK_BEFORE


def should_break_after(char):
    """Check if we can break line after this character"""
    return char not in NO_BREAK_AFTER


def is_math_token(token):
    """Check if token is a math operator or part of math expression"""
    if len(token) == 1 and token in MATH_OPERATORS:
        return True
    return False


def get_text_width_pixels(text, font_en, font_cn):
    """Calculate actual pixel width of text using font metrics with fallback"""
    total_width = 0
    for char in text:
        if is_chinese_char(char):
            font = font_cn
        else:
            font = font_en
        
        if hasattr(font, 'getlength'):
            total_width += font.getlength(char)
        else:
            total_width += font.getsize(char)[0]
    return total_width


def wrap_text_by_pixels(text, max_width, font_en, font_cn):
    """
    Wrap text based on pixel width, respecting line break rules.
    Returns list of lines.
    """
    if not text:
        return ['']
    
    lines = []
    current_line = ""
    current_width = 0
    
    # Split text into characters for Chinese, words for English
    # Simple approach: iterate character by character
    i = 0
    while i < len(text):
        char = text[i]
        
        # Handle newlines
        if char == '\n':
            lines.append(current_line)
            current_line = ""
            current_width = 0
            i += 1
            continue
        
        # Get character width
        if is_chinese_char(char):
            char_width = font_cn.getlength(char) if hasattr(font_cn, 'getlength') else font_cn.getsize(char)[0]
        else:
            char_width = font_en.getlength(char) if hasattr(font_en, 'getlength') else font_en.getsize(char)[0]
        
        # Check if adding this char would exceed max width
        if current_width + char_width > max_width and current_line:
            # Need to wrap - but check line break rules
            # Check if current char is NO_BREAK_BEFORE (shouldn't start a line)
            if not should_break_before(char):
                # This char should stay with previous, add it anyway
                current_line += char
                current_width += char_width
                i += 1
                # Then wrap after
                lines.append(current_line)
                current_line = ""
                current_width = 0
                continue
            
            # Check if previous char is NO_BREAK_AFTER (shouldn't end a line)
            if current_line and not should_break_after(current_line[-1]):
                # Previous char shouldn't end line, add current char anyway
                current_line += char
                current_width += char_width
                i += 1
                continue
            
            # Normal wrap
            lines.append(current_line)
            current_line = char
            current_width = char_width
        else:
            current_line += char
            current_width += char_width
        
        i += 1
    
    if current_line:
        lines.append(current_line)
    
    return lines if lines else ['']


def draw_text_with_fallback(draw, position, text, fill, font_en, font_cn):
    """
    Draw text with character-level font fallback.
    Uses English font for ASCII, Chinese font for CJK characters.
    Adjusts vertical position for English chars to align baselines.
    """
    x, y = position
    
    # Calculate vertical offset for English font to align with Chinese font baseline
    # Chinese fonts typically have lower baseline, so we need to shift English chars down
    en_offset = 2  # Pixels to shift English chars down for baseline alignment
    
    for char in text:
        if is_chinese_char(char):
            font = font_cn
            char_y = y
        else:
            font = font_en
            char_y = y + en_offset  # Shift English chars down slightly
        
        # Get character width
        if hasattr(font, 'getlength'):
            char_width = font.getlength(char)
        else:
             # Fallback for older Pillow
            char_width = font.getsize(char)[0]
        
        # Draw the character
        draw.text((x, char_y), char, fill=fill, font=font)
        x += char_width


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
        font_path: Optional path to font file (ignored, uses system fonts with fallback)
    
    Returns:
        True if video was created, False if skipped (already exists)
    """
    # Always load both fonts for fallback support
    try:
        font_en = ImageFont.truetype(str(get_font_path()), PROMPT_FONT_SIZE)
        font_cn = ImageFont.truetype(str(get_chinese_font_path()), PROMPT_FONT_SIZE)
    except Exception as e:
        print(f"Warning: Failed to load fonts: {e}. using default.")
        font_en = ImageFont.load_default()
        font_cn = ImageFont.load_default()
    
    # Analyze text to determine wrapping width
    has_chinese_prompt = any(is_chinese_char(c) for c in prompt_text)
    has_chinese_response = any(is_chinese_char(c) for c in response_text)
    
    # Calculate max text width in pixels (accounting for margins)
    margin = 15
    max_text_width = VIDEO_WIDTH - 2 * margin

    # Split response into tokens (words and newlines)
    # For Chinese, we might want to split by character if there are no spaces, 
    # but for now let's stick to the existing logic which respects spaces if present.
    # If the text is purely Chinese without spaces, we might need to insert spaces?
    # No, gradual display usually works by "token". If Chinese text has no spaces, 
    # it might come as one giant token.
    # Hack for pure Chinese text without spaces: insert spaces between characters for splitting, then remove them?
    # Or just treat each character as a token if no spaces found.
    
    if has_chinese_response:
        # Treat every character as a token for dense Chinese text
        # But group: 1) no-break punctuation, 2) numbers/decimals/operators
        tokens = []
        current_group = ""
        in_number_mode = False  # Track if we're accumulating numbers
        
        def is_numeric_or_operator(c):
            """Check if char is digit, decimal point, or math operator"""
            return c.isdigit() or c in '.+-*/×÷=≠<>≤≥±%'
        
        for i, char in enumerate(response_text):
            if char == '\n':
                if current_group:
                    tokens.append(current_group)
                    current_group = ""
                in_number_mode = False
                tokens.append('\n')
            elif is_numeric_or_operator(char):
                # If switching from non-number to number, flush previous group
                if not in_number_mode and current_group:
                    tokens.append(current_group)
                    current_group = ""
                # Numbers and operators should stay together
                current_group += char
                in_number_mode = True
            elif not should_break_before(char):
                # This char should stay with previous (e.g., closing punctuation)
                current_group += char
                in_number_mode = False
            elif not should_break_after(char):
                # This char should stay with next (e.g., opening punctuation)
                if current_group:
                    tokens.append(current_group)
                current_group = char
                in_number_mode = False
            else:
                # Normal character - flush previous group and start new
                if current_group:
                    tokens.append(current_group)
                current_group = char
                in_number_mode = False
        if current_group:
            tokens.append(current_group)
    else:
        # Standard splitting
        tokens = re.split(r'(\n+)', response_text)
    
    response_parts = []  # List of (word, newline_after)
    
    if has_chinese_response:
        # Handling for grouped Chinese tokens
        for token in tokens:
            if token == '\n':
                if response_parts:
                    response_parts[-1] = (response_parts[-1][0], '\n')
            elif token.strip():
                response_parts.append((token, ''))
    else:
        # Standard processing
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

    # Calculate frame counts - exactly 193 frames
    prompt_only_frames = 1
    
    words_at_frame = []
    for frame_num in range(TOTAL_FRAMES):
        if frame_num < prompt_only_frames:
            words_at_frame.append(0)
        else:
            words_to_show = min(frame_num - prompt_only_frames + 1, total_words)
            words_at_frame.append(words_to_show)
    
    # Initialize video writer

    
    # Layout parameters
    margin = 15
    line_height = RESPONSE_FONT_SIZE + 4
    max_y = VIDEO_HEIGHT - margin
    
    # Wrap prompt lines using pixel-based wrapping
    prompt_lines = []
    for line in prompt_text.split('\n'):
        if line.strip():
            wrapped = wrap_text_by_pixels(line, max_text_width, font_en, font_cn)
            prompt_lines.extend(wrapped if wrapped else [''])
        else:
            prompt_lines.append('')
    
    # Calculate prompt height to check if there's enough space for response
    prompt_height = len(prompt_lines) * line_height
    separator_height = 20  # 5px gap + 2px line + 15px gap
    available_for_response = VIDEO_HEIGHT - margin - prompt_height - separator_height - margin
    min_response_lines = 6  # Require at least 6 lines for response to avoid cramping
    
    # print(f"DEBUG: prompt_lines={len(prompt_lines)}, prompt_height={prompt_height}, available={available_for_response}, min_req={min_response_lines*line_height}")
    
    if available_for_response < min_response_lines * line_height:
        print(f"Skipping sample: prompt too long ({len(prompt_lines)} lines), only {available_for_response}px available for response")
        return False
    
    # Validated: Output video initialization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Track pagination
    current_page_start_word = 0
    
    # Generate frames
    for frame_num in range(TOTAL_FRAMES):
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        
        y = margin
        current_word_count = words_at_frame[frame_num]
        
        # --- Draw Prompt (Always visible) ---
        for line in prompt_lines:
            draw_text_with_fallback(draw, (margin, y), line, PROMPT_COLOR, font_en, font_cn)
            y += line_height
        
        # Draw Separator
        y += 5
        draw.line([(margin, y), (VIDEO_WIDTH - margin, y)], fill=(0, 0, 0), width=2)
        y += 15
        separator_y = y
        
        # --- Draw Response (Gradual) ---
        # Determine content for this page
        # We need to simulate wrapping from the very beginning to find where the current view starts
        
        # Simulation to find page start
        sim_y = separator_y
        sim_start_word_idx = 0
        
        # Logic: We simulate drawing ALL words up to current_word_count.
        # If text goes off screen, we clear and start a new page.
        # The page start index is updated accordingly.
        
        page_content_words = [] # List of words on current page
        
        cursor_x = margin
        
        # Re-calculating layout for every frame is inefficient but robust.
        # Given small text size (max 200 words), it's negligible.
        
        current_line_words = []
        current_line_len = 0
        
        last_page_start_index = 0
        
        # We need to reconstruct lines to wrap correctly
        # This is tricky because we are revealing word by word.
        # Formatting (wrapping) should remain stable as words appear.
        
        # Construct the full text up to current_word_count to determine layout
        # But wait, wrapping depends on future words? 
        # No, textwrap usually works paragraph by paragraph.
        # We simply reconstruct the "visible" text and wrap it?
        # No, that would cause words to jump lines as new words appear.
        # We must wrap the ENTIRE response first, then display only visible parts.
        
        # 1. Reconstruct full response text preserving newlines
        # Actually we have response_parts = [(word, newline_suffix), ...]
        
        full_visible_lines = []
        current_line = []
        current_line_char_count = 0
        
        # We need to wrap the WHOLE text to ensure stable layout
        # then only draw what is visible.
        
        all_lines_struct = [] # List of [ (word_index, word_text), ... ] represents one line
        
        curr_line_struct = []
        curr_line_width = 0  # Track width in pixels
        is_pure_chinese = has_chinese_response
        
        # Get space width for English text
        if hasattr(font_en, 'getlength'):
            space_width = font_en.getlength(' ')
        else:
            space_width = font_en.getsize(' ')[0]
        
        for idx, (word, suffix) in enumerate(response_parts):
            # Calculate word width in pixels
            word_width = get_text_width_pixels(word, font_en, font_cn)
            
            # Calculate space between words (0 for pure Chinese)
            if is_pure_chinese:
                space_before = 0
            else:
                space_before = space_width if curr_line_width > 0 else 0
            
            # Check if this word would exceed line width in pixels
            would_exceed = curr_line_width + space_before + word_width > max_text_width

            
            if would_exceed and curr_line_struct:
                # The current word causes overflow.
                # Check if we should move previous words to the next line to keep them together (sticky rules).
                
                # Start defining the cluster of words that MUST stay together
                cluster = [(idx, word)]
                
                # Backtrack to pull previous words if they are bonded
                while curr_line_struct:
                    last_idx, last_word_text = curr_line_struct[-1][0], curr_line_struct[-1][1]
                    next_word_text = cluster[0][1] # The first word of the cluster we are building
                    
                    last_char_of_prev = last_word_text[-1] if last_word_text else ''
                    first_char_of_curr = next_word_text[0] if next_word_text else ''
                    
                    is_bonded = False
                    
                    # Rule 1: Don't break if current token starts with no-break-before char
                    if first_char_of_curr and not should_break_before(first_char_of_curr):
                        is_bonded = True
                    
                    # Rule 2: Don't break if previous token ends with no-break-after char
                    elif last_char_of_prev and not should_break_after(last_char_of_prev):
                        is_bonded = True
                        
                    # Rule 3: Math expressions
                    elif is_math_token(next_word_text) or is_math_token(last_word_text):
                        is_bonded = True
                        
                    if is_bonded:
                        # Move last word from current line to cluster
                        cluster.insert(0, curr_line_struct.pop())
                    else:
                        # No bond, valid break point found
                        break
                
                # If curr_line_struct is effectively empty after backtracking,
                # it means the whole line was one big cluster (or we are at start).
                # We interpret this as "Current line is finished/empty", start new line with cluster.
                
                # Flush current line if anything remains
                if curr_line_struct:
                    all_lines_struct.append(curr_line_struct)
                
                # Start new line with the cluster
                curr_line_struct = cluster
                
                # Recalculate width for the new line
                curr_line_width = 0
                for i, (_, w_text) in enumerate(curr_line_struct):
                    w_px = get_text_width_pixels(w_text, font_en, font_cn)
                    sp = 0
                    if i > 0:
                        sp = 0 if is_pure_chinese else space_width
                    curr_line_width += sp + w_px
                
                continue
            
            curr_line_struct.append((idx, word))
            curr_line_width += space_before + word_width
            
            if '\n' in suffix:
                all_lines_struct.append(curr_line_struct)
                curr_line_struct = []
                curr_line_width = 0
        
        if curr_line_struct:
            all_lines_struct.append(curr_line_struct)
            
        # calculate pagination based on all lines
        lines_per_page = int((max_y - separator_y) / line_height)
        if lines_per_page < 1: lines_per_page = 1
        
        # Find which page includes the last visible word (current_word_count - 1)
        # However, we only show words up to current_word_count.
        
        visible_lines_to_draw = []
        start_line_idx = 0
        
        # Locate the line containing the last visible word
        if current_word_count > 0:
            last_visible_idx = current_word_count - 1
            
            # Find line containing this index
            target_line_idx = 0
            for i, line_data in enumerate(all_lines_struct):
                for w_idx, _ in line_data:
                    if w_idx == last_visible_idx:
                        target_line_idx = i
                        break
                else:
                    continue
                break
            
            # Calculate page start line
            page_num = target_line_idx // lines_per_page
            start_line_idx = page_num * lines_per_page
        
        # Draw visible lines for current page
        y = separator_y
        for i in range(start_line_idx, min(start_line_idx + lines_per_page, len(all_lines_struct))):
            line_data = all_lines_struct[i]
            x = margin
            
            for w_idx, word in line_data:
                if w_idx < current_word_count:
                    draw_text_with_fallback(draw, (x, y), word, RESPONSE_COLOR, font_en, font_cn)
                    
                    # Calculate advance width using fallback logic
                    # This is an approximation since we don't return width from draw
                    # We need to re-calculate width since draw_text_with_fallback doesn't return it
                    w_width = 0
                    for char in word:
                        f = font_cn if is_chinese_char(char) else font_en
                        if hasattr(f, 'getlength'):
                            w_width += f.getlength(char)
                        else:
                            w_width += f.getsize(char)[0]
                    
                    x += w_width
                    
                    # Add space if not last in line
                    if line_data.index((w_idx, word)) < len(line_data) - 1:
                         if not has_chinese_response:
                             if hasattr(font_en, 'getlength'):
                                 space_w = font_en.getlength(' ')
                             else:
                                 space_w = font_en.getsize(' ')[0]
                             x += space_w
            y += line_height

        # Save frame
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert for OpenCV
        out.write(frame)
    
    out.release()
    return True
