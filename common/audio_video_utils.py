"""
Audio Video Generation Utilities for VideoWordData.

This module provides functions to create 10-second videos with:
- Text-to-Speech (TTS) audio using edge-tts
- Synchronized subtitles at the bottom of the video
- Clean white background throughout

All videos are fixed at 10 seconds duration.
Subtitles display word-by-word in sync with audio.

Dependencies:
    pip install edge-tts moviepy
"""

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MoviePy 2.x compatibility
try:
    from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, ColorClip
except ImportError:
    from moviepy.editor import (
        ImageClip, AudioFileClip, CompositeVideoClip, ColorClip
    )

# Import shared utilities from video_utils
from .video_utils import (
    VIDEO_WIDTH, VIDEO_HEIGHT, BACKGROUND_COLOR,
    get_font_path, get_chinese_font_path,
    is_chinese_char, wrap_text_by_pixels, draw_text_with_fallback,
    get_text_width_pixels, VIDEO_DURATION
)

# Audio video specific settings
AUDIO_FPS = 24  # Standard video FPS
AUDIO_VIDEO_DURATION = 10.0  # Fixed 10-second video duration

# Subtitle settings - WHITE background with BLACK text
SUBTITLE_AREA_HEIGHT = 100  # Height of subtitle area at bottom
SUBTITLE_FONT_SIZE = 36     # Large font for subtitles
SUBTITLE_BG_COLOR = (255, 255, 255)  # White background (same as video)
SUBTITLE_TEXT_COLOR = (0, 0, 0)  # Black text
SUBTITLE_PADDING = 15
SUBTITLE_MAX_LINES = 1  # Single line subtitle

# TTS Voice configurations
TTS_VOICES = {
    "en": "en-US-AriaNeural",
    "en-male": "en-US-GuyNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "zh-male": "zh-CN-YunxiNeural",
}

# Speaking rates (words/chars per second) - tuned for Piper TTS
ENGLISH_WPS = 3.5  # Piper TTS speaks faster (~3.5 words per second)
CHINESE_CPS = 5.0  # Piper TTS Chinese is also faster


def get_tts_voice(language: str = "en", gender: str = "female") -> str:
    """Get the TTS voice ID for the given language and gender."""
    key = f"{language}-{gender}" if gender == "male" else language
    return TTS_VOICES.get(key, TTS_VOICES["en"])


def estimate_audio_duration(text: str, language: str = "en") -> float:
    """Estimate the audio duration for a given text."""
    if language == "zh":
        char_count = len([c for c in text if is_chinese_char(c)])
        return char_count / CHINESE_CPS
    else:
        word_count = len(text.split())
        return word_count / ENGLISH_WPS


def split_text_into_sentences(text: str, language: str = "en") -> List[str]:
    """Split text into sentences."""
    if language == "zh":
        pattern = r'(?<=[。！？；])\s*'
    else:
        pattern = r'(?<=[.!?])\s+'
    
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def find_sentences_within_duration(
    sentences: List[str], 
    max_duration: float,
    language: str = "en"
) -> int:
    """Find how many sentences can fit within the given duration."""
    total_duration = 0.0
    count = 0
    
    for s in sentences:
        duration = estimate_audio_duration(s, language)
        if total_duration + duration <= max_duration:
            total_duration += duration
            count += 1
        else:
            break
    
    return count


def can_fit_in_duration(text: str, max_duration: float, language: str = "en") -> bool:
    """Check if text can be read within the given duration."""
    return estimate_audio_duration(text, language) <= max_duration


def calculate_subtitle_capacity(
    font_en: ImageFont.FreeTypeFont,
    font_cn: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int = SUBTITLE_MAX_LINES
) -> int:
    """
    Calculate approximately how many words can fit in the subtitle area.
    Returns estimated word capacity.
    """
    # Estimate average word width (rough estimate)
    avg_word_width = font_en.getlength("average ") if hasattr(font_en, 'getlength') else 80
    words_per_line = int(max_width / avg_word_width)
    return words_per_line * max_lines


# Piper TTS model paths (must be downloaded beforehand)
# Download from: https://github.com/rhasspy/piper/releases
# English: en_US-lessac-medium.onnx
# Chinese: zh_CN-huayan-medium.onnx
PIPER_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "piper"

PIPER_MODELS = {
    "en": "en_US-lessac-medium.onnx",
    "zh": "zh_CN-huayan-medium.onnx",
}

# Global piper voice cache
_piper_voice = None
_piper_voice_language = None


def get_piper_voice(language: str = "en"):
    """
    Get or initialize the Piper TTS voice.
    
    Models must be downloaded beforehand to PIPER_MODEL_DIR.
    """
    global _piper_voice, _piper_voice_language
    
    # Return cached voice if language matches
    if _piper_voice is not None and _piper_voice_language == language:
        return _piper_voice
    
    from piper import PiperVoice
    
    model_name = PIPER_MODELS.get(language, PIPER_MODELS["en"])
    model_path = PIPER_MODEL_DIR / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper model not found: {model_path}\n"
            f"Please download from https://github.com/rhasspy/piper/releases\n"
            f"and place in {PIPER_MODEL_DIR}"
        )
    
    print(f"Loading Piper TTS model: {model_path}")
    _piper_voice = PiperVoice.load(str(model_path))
    _piper_voice_language = language
    
    return _piper_voice


def generate_tts_with_word_timestamps_sync(
    text: str,
    output_audio_path: str,
    voice: str = None,
    language: str = "en"
) -> Tuple[List[Tuple[float, float, str]], float]:
    """
    Generate TTS audio using Piper TTS (offline, CPU/GPU).
    
    Returns:
        Tuple of (word_timestamps list, actual duration)
        Each timestamp is (start_time, end_time, word)
    """
    import wave
    
    try:
        piper_voice = get_piper_voice(language)
        
        # Generate audio - piper outputs WAV
        wav_path = output_audio_path.replace('.mp3', '.wav')
        
        # Piper synthesize yields AudioChunk objects
        audio_chunks = list(piper_voice.synthesize(text))
        
        if not audio_chunks:
            print("No audio chunks generated")
            return [], 0
        
        # Join all audio data
        all_audio = b''.join(c.audio_int16_bytes for c in audio_chunks)
        sample_rate = audio_chunks[0].sample_rate
        sample_width = audio_chunks[0].sample_width
        sample_channels = audio_chunks[0].sample_channels
        
        # Write WAV file with proper configuration
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(sample_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(all_audio)
        
        # wav_path is now set and will be used below
        
    except Exception as e:
        print(f"Error generating TTS: {e}")
        import traceback
        traceback.print_exc()
        return [], 0
    
    # Get actual audio duration (use wav_path, not output_audio_path)
    try:
        audio_clip = AudioFileClip(wav_path)
        actual_duration = audio_clip.duration
        audio_clip.close()
    except Exception as e:
        print(f"Error loading audio: {e}")
        return [], 0
    
    # Split text into words and distribute timestamps evenly
    if language == "zh":
        # For Chinese, split by characters
        words = [c for c in text if c.strip()]
    else:
        # For English, split by words
        words = text.split()
    
    if not words:
        return [], actual_duration
    
    # Distribute timestamps evenly across audio duration
    # Leave small padding at start and larger padding at end
    start_padding = 0.05
    end_padding = 0.3  # Adjusted for Piper TTS
    usable_duration = actual_duration - start_padding - end_padding
    
    if usable_duration <= 0:
        usable_duration = actual_duration
        start_padding = 0
    
    word_duration = usable_duration / len(words)
    
    word_timestamps = []
    for i, word in enumerate(words):
        w_start = start_padding + i * word_duration
        w_end = w_start + word_duration
        word_timestamps.append((w_start, w_end, word))
    
    return word_timestamps, actual_duration


def group_words_for_display(
    word_timestamps: List[Tuple[float, float, str]],
    max_width: int,
    font_en: ImageFont.FreeTypeFont,
    font_cn: ImageFont.FreeTypeFont,
    max_lines: int = SUBTITLE_MAX_LINES,
    language: str = "en"
) -> List[Tuple[float, float, str]]:
    """
    Group words into display chunks that fit in the subtitle area.
    
    Returns list of (start_time, end_time, text_to_display) for each chunk.
    """
    if not word_timestamps:
        return []
    
    chunks = []
    current_words = []
    current_start = None
    current_end = None
    
    # Calculate line height
    subtitle_line_height = SUBTITLE_FONT_SIZE + 6
    
    for start, end, word in word_timestamps:
        # Try adding this word
        test_words = current_words + [word]
        if language == "zh":
            test_text = ''.join(test_words)
        else:
            test_text = ' '.join(test_words)
        
        # Check if it fits
        test_lines = wrap_text_by_pixels(test_text, max_width, font_en, font_cn)
        
        if len(test_lines) <= max_lines:
            # Fits - add the word
            current_words.append(word)
            if current_start is None:
                current_start = start
            current_end = end
        else:
            # Doesn't fit - save current chunk and start new one
            if current_words:
                if language == "zh":
                    chunk_text = ''.join(current_words)
                else:
                    chunk_text = ' '.join(current_words)
                chunks.append((current_start, current_end, chunk_text))
            
            # Start new chunk with this word
            current_words = [word]
            current_start = start
            current_end = end
    
    # Don't forget the last chunk
    if current_words:
        if language == "zh":
            chunk_text = ''.join(current_words)
        else:
            chunk_text = ' '.join(current_words)
        chunks.append((current_start, current_end, chunk_text))
    
    return chunks


def create_subtitle_frame(
    subtitle_text: str,
    frame_size: Tuple[int, int] = (VIDEO_WIDTH, VIDEO_HEIGHT),
    subtitle_font_en: ImageFont.FreeTypeFont = None,
    subtitle_font_cn: ImageFont.FreeTypeFont = None,
) -> np.ndarray:
    """
    Create a frame with subtitle at bottom.
    White background throughout, black text for subtitles.
    """
    width, height = frame_size
    
    # Load fonts if not provided
    if subtitle_font_en is None:
        try:
            subtitle_font_en = ImageFont.truetype(str(get_font_path()), SUBTITLE_FONT_SIZE)
        except:
            subtitle_font_en = ImageFont.load_default()
    if subtitle_font_cn is None:
        try:
            subtitle_font_cn = ImageFont.truetype(str(get_chinese_font_path()), SUBTITLE_FONT_SIZE)
        except:
            subtitle_font_cn = subtitle_font_en
    
    # Create white background (same color throughout)
    img = Image.new('RGB', (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Subtitle area position (bottom of video)
    subtitle_area_top = height - SUBTITLE_AREA_HEIGHT
    
    # Draw subtitle text (centered, black on white)
    if subtitle_text:
        subtitle_max_width = width - 2 * SUBTITLE_PADDING
        subtitle_line_height = SUBTITLE_FONT_SIZE + 6
        
        subtitle_lines = wrap_text_by_pixels(
            subtitle_text, subtitle_max_width, 
            subtitle_font_en, subtitle_font_cn
        )
        
        # Limit to max lines (no truncation/ellipsis - text should already fit)
        subtitle_lines = subtitle_lines[:SUBTITLE_MAX_LINES]
        
        total_subtitle_height = len(subtitle_lines) * subtitle_line_height
        subtitle_y = subtitle_area_top + (SUBTITLE_AREA_HEIGHT - total_subtitle_height) // 2
        
        for line in subtitle_lines:
            line_width = get_text_width_pixels(line, subtitle_font_en, subtitle_font_cn)
            subtitle_x = (width - line_width) // 2
            
            draw_text_with_fallback(
                draw, (subtitle_x, subtitle_y), line, 
                SUBTITLE_TEXT_COLOR, subtitle_font_en, subtitle_font_cn
            )
            subtitle_y += subtitle_line_height
    
    return np.array(img)


def create_video_with_audio_subtitles(
    text_to_speak: str,
    output_path: str,
    language: str = "en",
    voice: str = None,
    fps: int = AUDIO_FPS,
    target_duration: float = AUDIO_VIDEO_DURATION,
) -> bool:
    """
    Create a 10-second video with TTS audio and synchronized subtitles.
    
    Subtitles are displayed word-by-word, grouped by subtitle area capacity.
    White background with black text.
    
    Args:
        text_to_speak: Text to speak and display as subtitles
        output_path: Path to save the output video
        language: Language code ('en' or 'zh')
        voice: TTS voice to use
        fps: Video frame rate
        target_duration: Target video duration (default 10 seconds)
    
    Returns:
        True if video was created successfully, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load fonts
    try:
        subtitle_font_en = ImageFont.truetype(str(get_font_path()), SUBTITLE_FONT_SIZE)
        subtitle_font_cn = ImageFont.truetype(str(get_chinese_font_path()), SUBTITLE_FONT_SIZE)
    except Exception as e:
        print(f"Warning: Failed to load fonts: {e}")
        subtitle_font_en = ImageFont.load_default()
        subtitle_font_cn = subtitle_font_en
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")  # Piper generates WAV
        
        try:
            word_timestamps, actual_duration = generate_tts_with_word_timestamps_sync(
                text_to_speak, audio_path, voice, language
            )
        except Exception as e:
            print(f"Error generating TTS: {e}")
            return False
        
        if not word_timestamps:
            print("No word timestamps generated from TTS")
            return False
        
        # Check if audio is too long
        if actual_duration > target_duration + 0.5:
            print(f"Audio too long ({actual_duration:.1f}s > {target_duration}s), skipping")
            return False
        
        # Load audio
        try:
            audio_clip = AudioFileClip(audio_path)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
        
        video_duration = target_duration
        subtitle_max_width = VIDEO_WIDTH - 2 * SUBTITLE_PADDING
        
        # Group words into display chunks based on subtitle area capacity
        display_chunks = group_words_for_display(
            word_timestamps, subtitle_max_width,
            subtitle_font_en, subtitle_font_cn,
            SUBTITLE_MAX_LINES, language
        )
        
        # Create video frames
        clips = []
        
        # Initial blank frame if first chunk doesn't start at 0
        if display_chunks and display_chunks[0][0] > 0.05:
            initial_frame = create_subtitle_frame(
                "",
                subtitle_font_en=subtitle_font_en,
                subtitle_font_cn=subtitle_font_cn
            )
            initial_clip = ImageClip(initial_frame).with_duration(display_chunks[0][0])
            clips.append(initial_clip)
        
        # Frames for each display chunk
        for i, (start_time, end_time, text) in enumerate(display_chunks):
            if start_time >= video_duration:
                break
            
            # Duration until next chunk or end
            if i + 1 < len(display_chunks):
                next_start = min(display_chunks[i + 1][0], video_duration)
                duration = next_start - start_time
            else:
                duration = video_duration - start_time
            
            if duration <= 0:
                duration = 0.1
            
            frame = create_subtitle_frame(
                text,
                subtitle_font_en=subtitle_font_en,
                subtitle_font_cn=subtitle_font_cn
            )
            clip = ImageClip(frame).with_start(start_time).with_duration(duration)
            clips.append(clip)
        
        # Final blank frame if needed
        if display_chunks:
            last_end = display_chunks[-1][1]
            if last_end < video_duration:
                final_frame = create_subtitle_frame(
                    "",
                    subtitle_font_en=subtitle_font_en,
                    subtitle_font_cn=subtitle_font_cn
                )
                final_clip = ImageClip(final_frame).with_start(last_end).with_duration(video_duration - last_end)
                clips.append(final_clip)
        
        # Composite
        if clips:
            final_video = CompositeVideoClip(clips, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
            final_video = final_video.with_duration(video_duration)
        else:
            frame = create_subtitle_frame("", subtitle_font_en=subtitle_font_en, subtitle_font_cn=subtitle_font_cn)
            final_video = ImageClip(frame).with_duration(video_duration)
        
        # Add audio
        final_video = final_video.with_audio(audio_clip)
        
        # Write video
        try:
            final_video.write_videofile(
                str(output_path),
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                logger=None,
                threads=4,
            )
        except Exception as e:
            print(f"Error writing video: {e}")
            return False
        finally:
            audio_clip.close()
            final_video.close()
    
    return True


def create_video_with_audio_subtitles_fast(
    text_to_speak: str,
    output_path: str,
    language: str = "en",
    voice: str = None,
    fps: int = AUDIO_FPS,
    target_duration: float = AUDIO_VIDEO_DURATION,
) -> bool:
    """
    FAST version using ffmpeg directly instead of moviepy.
    Creates a 10-second video with TTS audio and synchronized subtitles.
    
    This is 5-10x faster than the moviepy version.
    """
    import subprocess
    import wave
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load fonts
    try:
        subtitle_font_en = ImageFont.truetype(str(get_font_path()), SUBTITLE_FONT_SIZE)
        subtitle_font_cn = ImageFont.truetype(str(get_chinese_font_path()), SUBTITLE_FONT_SIZE)
    except Exception:
        subtitle_font_en = ImageFont.load_default()
        subtitle_font_cn = subtitle_font_en
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        # Generate TTS audio
        try:
            piper_voice = get_piper_voice(language)
            audio_chunks = list(piper_voice.synthesize(text_to_speak))
            
            if not audio_chunks:
                return False
            
            all_audio = b''.join(c.audio_int16_bytes for c in audio_chunks)
            sample_rate = audio_chunks[0].sample_rate
            sample_width = audio_chunks[0].sample_width
            sample_channels = audio_chunks[0].sample_channels
            
            with wave.open(audio_path, 'wb') as wav_file:
                wav_file.setnchannels(sample_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(all_audio)
                
            actual_duration = len(all_audio) / (sample_rate * sample_width)
        except Exception as e:
            print(f"Error generating TTS: {e}")
            return False
        
        if actual_duration > target_duration + 0.5:
            print(f"Audio too long ({actual_duration:.1f}s > {target_duration}s), skipping")
            return False
        
        # Split text into words for timestamp distribution
        if language == "zh":
            words = [c for c in text_to_speak if c.strip()]
        else:
            words = text_to_speak.split()
        
        if not words:
            return False
        
        # Calculate word timestamps
        start_padding = 0.05
        end_padding = 0.3
        usable_duration = actual_duration - start_padding - end_padding
        if usable_duration <= 0:
            usable_duration = actual_duration
            start_padding = 0
        
        word_duration = usable_duration / len(words)
        word_timestamps = []
        for i, word in enumerate(words):
            w_start = start_padding + i * word_duration
            w_end = w_start + word_duration
            word_timestamps.append((w_start, w_end, word))
        
        # Group words for display
        subtitle_max_width = VIDEO_WIDTH - 2 * SUBTITLE_PADDING
        display_chunks = group_words_for_display(
            word_timestamps, subtitle_max_width,
            subtitle_font_en, subtitle_font_cn,
            SUBTITLE_MAX_LINES, language
        )
        
        # Generate frames as images
        frame_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frame_dir)
        
        total_frames = int(target_duration * fps)
        current_chunk_idx = 0
        
        for frame_num in range(total_frames):
            t = frame_num / fps
            
            # Find current subtitle chunk
            current_text = ""
            while current_chunk_idx < len(display_chunks):
                chunk_start, chunk_end, chunk_text = display_chunks[current_chunk_idx]
                if t < chunk_start:
                    break
                elif t <= chunk_end or (current_chunk_idx + 1 < len(display_chunks) and t < display_chunks[current_chunk_idx + 1][0]):
                    current_text = chunk_text
                    break
                else:
                    current_chunk_idx += 1
            
            if current_chunk_idx > 0 and current_chunk_idx < len(display_chunks):
                current_text = display_chunks[current_chunk_idx][2] if t >= display_chunks[current_chunk_idx][0] else display_chunks[current_chunk_idx - 1][2]
            
            # Create frame
            frame = create_subtitle_frame(
                current_text,
                subtitle_font_en=subtitle_font_en,
                subtitle_font_cn=subtitle_font_cn
            )
            
            # Save frame as PNG
            frame_path = os.path.join(frame_dir, f"frame_{frame_num:05d}.png")
            Image.fromarray(frame).save(frame_path, optimize=True)
        
        # Use ffmpeg to combine frames + audio
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frame_dir, 'frame_%05d.png'),
                '-i', audio_path,
                '-c:v', 'libopenh264',  # More compatible codec
                '-c:a', 'aac',
                '-shortest',
                '-t', str(target_duration),
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                # Fallback to default codec
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(frame_dir, 'frame_%05d.png'),
                    '-i', audio_path,
                    '-c:a', 'aac',
                    '-shortest',
                    '-t', str(target_duration),
                    '-pix_fmt', 'yuv420p',
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
    
    return True


def test_audio_video_generation():
    """Test the audio video generation."""
    text = "Once upon a time, there was a little rabbit. The rabbit loved to hop around the meadow. One day, it met a friendly fox who became its best friend."
    output_path = "test_audio_video.mp4"
    
    print(f"Generating test video: {output_path}")
    print(f"Text: {text}")
    print(f"Estimated duration: {estimate_audio_duration(text, 'en'):.1f}s")
    
    success = create_video_with_audio_subtitles(
        text, output_path, language="en"
    )
    
    if success:
        print(f"✓ Video generated successfully: {output_path}")
    else:
        print("✗ Video generation failed")
    
    return success


if __name__ == "__main__":
    test_audio_video_generation()
