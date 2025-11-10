#!/usr/bin/env python3
"""
Video wall effect generator for multiple timelapses.
Merges timelapses by day, syncs them by timestamp, and creates a video wall
with zoom effect (starts zoomed in on first timelapse, then zooms out to show all).
"""

import cv2
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
import tempfile
import shutil
from collections import defaultdict
import math

# Import YoutubeDownloader
sys.path.insert(0, str(Path(__file__).parent / 'YouTubeDownloader'))
from youtube_downloader import YoutubeDownloader


def find_capture_dirs_for_date(date_str, captures_root='captures'):
    """
    Find all capture directories for a given date.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        captures_root: Root directory containing capture folders (default: 'captures')
    
    Returns:
        List of Path objects for capture directories matching the date
    """
    captures_root = Path(captures_root)
    if not captures_root.exists():
        return []
    
    # Find all directories that start with the date
    matching_dirs = []
    for item in captures_root.iterdir():
        if item.is_dir() and item.name.startswith(date_str):
            images_dir = item / 'images'
            if images_dir.exists():
                matching_dirs.append(item)
    
    return sorted(matching_dirs)


def extract_timestamp(frame_path):
    """Extract timestamp from frame filename."""
    filename = frame_path.stem  # Get filename without extension
    parts = filename.split('_')
    if len(parts) >= 4:
        # Format: frame_000254_2025-11-05_10-43-05
        date_str = parts[2]  # YYYY-MM-DD
        time_str = parts[3]   # HH-MM-SS
        try:
            return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H-%M-%S')
        except ValueError:
            return None
    return None


def collect_frames_from_dirs(capture_dirs):
    """
    Collect all frames from multiple capture directories and sort them chronologically.
    
    Args:
        capture_dirs: List of Path objects for capture directories
    
    Returns:
        List of Path objects for image files, sorted chronologically by timestamp
    """
    all_frames = []
    
    for capture_dir in capture_dirs:
        images_dir = capture_dir / 'images'
        frames = list(images_dir.glob('frame_*.jpg'))
        all_frames.extend(frames)
    
    # Sort frames by timestamp
    all_frames.sort(key=lambda p: (extract_timestamp(p) or datetime.min, p.name))
    
    return all_frames


def get_frames_for_timelapse(timelapse_dir, captures_root='captures'):
    """
    Get all frames for a timelapse directory, merging all recordings from the same day.
    
    Args:
        timelapse_dir: Path to a timelapse directory (or a date string)
        captures_root: Root directory containing capture folders
    
    Returns:
        List of Path objects for image files, sorted chronologically
    """
    timelapse_dir = Path(timelapse_dir)
    
    # Extract date from directory name (format: YYYY-MM-DD_HH-MM-SS)
    if timelapse_dir.is_dir():
        dir_name = timelapse_dir.name
        date_str = dir_name.split('_')[0]  # Get YYYY-MM-DD part
    else:
        # Assume it's a date string
        date_str = str(timelapse_dir)
    
    # Find all capture directories for this date
    captures_root = Path(captures_root)
    matching_dirs = find_capture_dirs_for_date(date_str, captures_root)
    
    if not matching_dirs:
        print(f"Warning: No capture directories found for date {date_str}")
        return []
    
    # Collect and sort all frames chronologically
    all_frames = collect_frames_from_dirs(matching_dirs)
    
    return all_frames


def sync_frames_by_timestamp(frame_lists, max_time_diff_seconds=5):
    """
    Sync multiple frame lists by timestamp (hour, minute, second).
    Returns a list of frame groups, where each group contains frames
    from all timelapses that match the same hour:minute:second.
    
    Smart frame selection:
    - If a timelapse hasn't started yet, use the first frame
    - If a timelapse has ended, use the last frame
    - Otherwise, use the closest frame within max_time_diff_seconds
    
    Args:
        frame_lists: List of lists, where each inner list contains frame Paths
        max_time_diff_seconds: Maximum time difference in seconds to consider frames synced (default: 5)
    
    Returns:
        List of tuples, where each tuple contains (timestamp, [frame1, frame2, ...])
    """
    # Build timestamp list for each timelapse and get first/last frames
    timelapse_timestamps = []
    first_frames = []
    last_frames = []
    
    for frames in frame_lists:
        timestamps = []
        for frame in frames:
            ts = extract_timestamp(frame)
            if ts:
                timestamps.append((ts, frame))
        # Sort by timestamp
        timestamps.sort(key=lambda x: x[0])
        timelapse_timestamps.append(timestamps)
        
        # Store first and last frames
        if timestamps:
            first_frames.append(timestamps[0][1])  # First frame
            last_frames.append(timestamps[-1][1])  # Last frame
        else:
            first_frames.append(None)
            last_frames.append(None)
    
    # Find all unique time keys (hour, minute, second) across all timelapses
    all_time_keys = set()
    for timestamps in timelapse_timestamps:
        for ts, _ in timestamps:
            time_key = (ts.hour, ts.minute, ts.second)
            all_time_keys.add(time_key)
    
    # Sort time keys
    sorted_time_keys = sorted(all_time_keys)
    
    # Build synced frame groups
    synced_groups = []
    for time_key in sorted_time_keys:
        frames_at_time = []
        
        # Convert time_key to datetime for comparison
        # Use a reference date (doesn't matter which, we only care about time)
        ref_time = datetime(2025, 1, 1, time_key[0], time_key[1], time_key[2])
        
        for idx, timestamps in enumerate(timelapse_timestamps):
            if not timestamps:
                # No frames for this timelapse
                frames_at_time.append(None)
                continue
            
            # Get first and last timestamps for this timelapse
            first_ts = timestamps[0][0]
            last_ts = timestamps[-1][0]
            
            first_time = datetime(2025, 1, 1, first_ts.hour, first_ts.minute, first_ts.second)
            last_time = datetime(2025, 1, 1, last_ts.hour, last_ts.minute, last_ts.second)
            ref_time = datetime(2025, 1, 1, time_key[0], time_key[1], time_key[2])
            
            # Check if timelapse hasn't started yet
            if ref_time < first_time:
                # Use first frame
                frames_at_time.append(first_frames[idx])
                continue
            
            # Check if timelapse has ended
            if ref_time > last_time:
                # Use last frame
                frames_at_time.append(last_frames[idx])
                continue
            
            # Find closest frame within max_time_diff_seconds
            best_frame = None
            min_diff = float('inf')
            
            for ts, frame in timestamps:
                # Calculate time difference (ignoring date, only time)
                frame_time = datetime(2025, 1, 1, ts.hour, ts.minute, ts.second)
                diff = abs((frame_time - ref_time).total_seconds())
                
                if diff < min_diff and diff <= max_time_diff_seconds:
                    min_diff = diff
                    best_frame = frame
            
            # If no frame found within threshold, use closest one anyway
            if best_frame is None:
                for ts, frame in timestamps:
                    frame_time = datetime(2025, 1, 1, ts.hour, ts.minute, ts.second)
                    diff = abs((frame_time - ref_time).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        best_frame = frame
            
            frames_at_time.append(best_frame)
        
        synced_groups.append((time_key, frames_at_time))
    
    return synced_groups


def add_film_grain_effect(image_path, output_path, grain_intensity=0.15, blur_radius=1.5):
    """
    Add film grain and filter effects to an image to create a vintage look
    and reduce text clarity.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with effects
        grain_intensity: Intensity of film grain (0.0-1.0, default: 0.15)
        blur_radius: Blur radius to reduce text clarity (default: 1.5)
    """
    # Open image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to float for processing
    img_float = img_array.astype(np.float32) / 255.0
    
    # Apply slight blur to reduce text clarity
    if blur_radius > 0:
        # Convert PIL to OpenCV format (RGB to BGR)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Apply Gaussian blur
        img_cv = cv2.GaussianBlur(img_cv, (0, 0), blur_radius)
        # Convert back to RGB
        img_array = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_float = img_array.astype(np.float32) / 255.0
    
    # Add film grain noise
    if grain_intensity > 0:
        # Generate random noise
        noise = np.random.normal(0, grain_intensity, img_float.shape).astype(np.float32)
        # Add noise to image
        img_float = img_float + noise
        # Clip values to valid range
        img_float = np.clip(img_float, 0, 1)
    
    # Apply subtle color grading for vintage look
    # Slight desaturation and warm tone
    # Convert to HSV for color manipulation (using BGR since we're working with OpenCV)
    img_bgr = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Slightly desaturate (reduce saturation by 10%)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.9
    # Slight warm tone shift (increase hue slightly for warmer look)
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + 5, 0, 179)  # OpenCV uses 0-179 for hue
    # Convert back to BGR then RGB
    img_bgr_processed = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img_processed = cv2.cvtColor(img_bgr_processed, cv2.COLOR_BGR2RGB)
    
    # Convert back to PIL Image
    img_final = Image.fromarray(img_processed)
    
    # Save with slightly reduced quality to further reduce clarity
    img_final.save(output_path, 'JPEG', quality=85)


def add_frame_overlay(frame_path, day_name=None, time_str=None, font_size=30):
    """
    Add day name (top left) and time (bottom right) overlay to a frame.
    
    Args:
        frame_path: Path to the frame image
        day_name: Day name string (e.g., "Thursday")
        time_str: Time string in HH:MM format (e.g., "09:00")
        font_size: Font size for text (default: 30)
    
    Returns:
        PIL Image with overlay, or None if frame couldn't be loaded
    """
    try:
        img = Image.open(frame_path)
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = None
        try:
            helvetica_paths = [
                '/System/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/Supplemental/Helvetica.ttc',
                '/Library/Fonts/Helvetica.ttc',
                '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                '/Library/Fonts/Arial Bold.ttf',
                '/Library/Fonts/Arial.ttf',
            ]
            for font_path in helvetica_paths:
                if os.path.exists(font_path):
                    try:
                        if font_path.endswith('.ttc'):
                            try:
                                font = ImageFont.truetype(font_path, font_size, index=1)
                                break
                            except:
                                try:
                                    font = ImageFont.truetype(font_path, font_size, index=0)
                                    break
                                except:
                                    continue
                        elif 'Bold' in font_path:
                            font = ImageFont.truetype(font_path, font_size)
                            break
                        else:
                            font = ImageFont.truetype(font_path, font_size)
                            break
                    except (OSError, IOError):
                        continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        img_width, img_height = img.size
        padding = 10
        
        # Add day name to top left
        if day_name:
            draw.text((padding, padding), day_name, fill='white', font=font)
        
        # Add time to bottom right
        if time_str:
            bbox = draw.textbbox((0, 0), time_str, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = img_width - padding - text_width
            y = img_height - padding - text_height
            draw.text((x, y), time_str, fill='white', font=font)
        
        return img
    except Exception as e:
        print(f"Warning: Could not add overlay to {frame_path}: {e}")
        return None


def create_video_wall_frame(frame_paths, grid_cols, grid_rows, output_size=(1920, 1080), 
                           zoom_factor=1.0, focus_x=None, focus_y=None, focus_index=0, last_frames=None):
    """
    Create a video wall frame from multiple images.
    
    Args:
        frame_paths: List of image paths (can contain None for missing frames)
        grid_cols: Number of columns in the grid
        grid_rows: Number of rows in the grid
        output_size: Output frame size (width, height)
        zoom_factor: Zoom factor (1.0 = full view, >1.0 = zoomed in)
        focus_x: X coordinate to focus on when zoomed (if None, uses focus_index)
        focus_y: Y coordinate to focus on when zoomed (if None, uses focus_index)
        focus_index: Index of the timelapse to focus on when zoomed in (used if focus_x/focus_y not provided)
        last_frames: Dictionary to track last valid frame for each index (for missing frames)
    
    Returns:
        Tuple of (PIL Image of the video wall, last_frames dictionary)
    """
    width, height = output_size
    
    # Calculate cell size
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    # Create base image
    wall_image = Image.new('RGB', (width, height), (0, 0, 0))
    
    # Track last valid frames
    if last_frames is None:
        last_frames = {}
    
    # Place each frame in the grid
    for idx, frame_path in enumerate(frame_paths):
        # Use last valid frame if current is None
        if frame_path is None:
            if idx in last_frames:
                frame_path = last_frames[idx]
            else:
                continue
        
        if not Path(frame_path).exists():
            if idx in last_frames:
                frame_path = last_frames[idx]
            else:
                continue
        
        row = idx // grid_cols
        col = idx % grid_cols
        
        try:
            # Load frame
            img = Image.open(frame_path)
            original_width, original_height = img.size
            
            # Crop and resize to fill cell completely (maintain aspect ratio, crop to fit)
            # Calculate scaling factor to fill the cell
            scale_w = cell_width / original_width
            scale_h = cell_height / original_height
            scale = max(scale_w, scale_h)  # Use larger scale to ensure we fill the cell
            
            # Calculate new size
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to cell dimensions (center crop)
            left = (new_width - cell_width) // 2
            top = (new_height - cell_height) // 2
            right = left + cell_width
            bottom = top + cell_height
            img = img.crop((left, top, right, bottom))
            
            # Extract timestamp for overlay
            ts = extract_timestamp(frame_path)
            time_str = None
            if ts:
                time_str = ts.strftime('%H:%M')  # Time in HH:MM format
            
            # Calculate font size based on cell size (increased for better visibility)
            # Use about 5-6% of the smaller dimension for font size
            img_width, img_height = img.size
            base_font_size = max(20, int(min(img_width, img_height) * 0.055))
            
            # Add overlay to resized frame
            if time_str:
                draw = ImageDraw.Draw(img)
                
                # Load font
                font = None
                try:
                    helvetica_paths = [
                        '/System/Library/Fonts/Helvetica.ttc',
                        '/System/Library/Fonts/Supplemental/Helvetica.ttc',
                        '/Library/Fonts/Helvetica.ttc',
                        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                        '/Library/Fonts/Arial Bold.ttf',
                        '/Library/Fonts/Arial.ttf',
                    ]
                    for font_path in helvetica_paths:
                        if os.path.exists(font_path):
                            try:
                                if font_path.endswith('.ttc'):
                                    try:
                                        font = ImageFont.truetype(font_path, base_font_size, index=1)
                                        break
                                    except:
                                        try:
                                            font = ImageFont.truetype(font_path, base_font_size, index=0)
                                            break
                                        except:
                                            continue
                                elif 'Bold' in font_path:
                                    font = ImageFont.truetype(font_path, base_font_size)
                                    break
                                else:
                                    font = ImageFont.truetype(font_path, base_font_size)
                                    break
                            except (OSError, IOError):
                                continue
                    if font is None:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                padding = max(5, int(min(img_width, img_height) * 0.01))
                
                # Add time to bottom right
                if time_str:
                    bbox = draw.textbbox((0, 0), time_str, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    x = img_width - padding - text_width
                    y = img_height - padding - text_height
                    draw.text((x, y), time_str, fill='white', font=font)
            
            # Paste onto wall (image is already exactly cell_width x cell_height)
            x = col * cell_width
            y = row * cell_height
            wall_image.paste(img, (x, y))
            
            # Update last valid frame
            last_frames[idx] = frame_path
        except Exception as e:
            # Try to use last valid frame
            if idx in last_frames:
                try:
                    last_frame = last_frames[idx]
                    img = Image.open(last_frame)
                    original_width, original_height = img.size
                    
                    # Crop and resize to fill cell completely
                    scale_w = cell_width / original_width
                    scale_h = cell_height / original_height
                    scale = max(scale_w, scale_h)
                    
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Crop to cell dimensions
                    left = (new_width - cell_width) // 2
                    top = (new_height - cell_height) // 2
                    right = left + cell_width
                    bottom = top + cell_height
                    img = img.crop((left, top, right, bottom))
                    
                    ts = extract_timestamp(last_frame)
                    time_str = None
                    if ts:
                        time_str = ts.strftime('%H:%M')
                    
                    # Add overlay
                    if time_str:
                        img_width, img_height = img.size
                        base_font_size = max(20, int(min(img_width, img_height) * 0.055))
                        draw = ImageDraw.Draw(img)
                        
                        # Load font
                        font = None
                        try:
                            helvetica_paths = [
                                '/System/Library/Fonts/Helvetica.ttc',
                                '/System/Library/Fonts/Supplemental/Helvetica.ttc',
                                '/Library/Fonts/Helvetica.ttc',
                                '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
                                '/Library/Fonts/Arial Bold.ttf',
                                '/Library/Fonts/Arial.ttf',
                            ]
                            for font_path in helvetica_paths:
                                if os.path.exists(font_path):
                                    try:
                                        if font_path.endswith('.ttc'):
                                            try:
                                                font = ImageFont.truetype(font_path, base_font_size, index=1)
                                                break
                                            except:
                                                try:
                                                    font = ImageFont.truetype(font_path, base_font_size, index=0)
                                                    break
                                                except:
                                                    continue
                                        elif 'Bold' in font_path:
                                            font = ImageFont.truetype(font_path, base_font_size)
                                            break
                                        else:
                                            font = ImageFont.truetype(font_path, base_font_size)
                                            break
                                    except (OSError, IOError):
                                        continue
                            if font is None:
                                font = ImageFont.load_default()
                        except:
                            font = ImageFont.load_default()
                        
                        padding = max(5, int(min(img_width, img_height) * 0.01))
                        
                        if time_str:
                            bbox = draw.textbbox((0, 0), time_str, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            x = img_width - padding - text_width
                            y = img_height - padding - text_height
                            draw.text((x, y), time_str, fill='white', font=font)
                    
                    # Paste onto wall (image is already exactly cell_width x cell_height)
                    x = col * cell_width
                    y = row * cell_height
                    wall_image.paste(img, (x, y))
                except:
                    pass
            continue
    
    # Apply zoom effect
    if zoom_factor > 1.0:
        # Use provided focus coordinates, or calculate from focus_index
        if focus_x is None or focus_y is None:
            if focus_index < len(frame_paths):
                # Calculate focus position in the grid
                focus_row = focus_index // grid_cols
                focus_col = focus_index % grid_cols
                focus_x = focus_col * cell_width + cell_width // 2
                focus_y = focus_row * cell_height + cell_height // 2
            else:
                # Default to center if focus_index is invalid
                focus_x = width // 2
                focus_y = height // 2
        
        # Convert to numpy for zoom
        wall_array = np.array(wall_image)
        
        # Calculate zoom
        zoom_width = int(width / zoom_factor)
        zoom_height = int(height / zoom_factor)
        
        # Calculate crop region centered on focus point
        crop_x = max(0, min(int(focus_x - zoom_width // 2), width - zoom_width))
        crop_y = max(0, min(int(focus_y - zoom_height // 2), height - zoom_height))
        
        # Crop and resize
        cropped = wall_array[crop_y:crop_y+zoom_height, crop_x:crop_x+zoom_width]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        
        wall_image = Image.fromarray(zoomed)
    
    return wall_image, last_frames


def calculate_grid_layout(num_timelapses):
    """
    Calculate optimal grid layout for given number of timelapses.
    
    Args:
        num_timelapses: Number of timelapses to display
    
    Returns:
        Tuple of (rows, cols)
    """
    if num_timelapses == 0:
        return (1, 1)
    
    # Calculate square-ish grid
    cols = int(math.ceil(math.sqrt(num_timelapses)))
    rows = int(math.ceil(num_timelapses / cols))
    
    return (rows, cols)


def generate_video_wall(timelapse_dirs, output_path, fps=24, zoom_duration=3.0, 
                       output_size=(1920, 1080), captures_root='captures', debug_limit=None,
                       apply_film_grain=True, audio_file=None, zoom_delay_frames=0, frame_interval=1):
    """
    Generate a video wall from multiple timelapses.
    
    Args:
        timelapse_dirs: List of timelapse directory paths or date strings
        output_path: Path for output video
        fps: Frames per second
        zoom_duration: Duration in seconds for zoom transition (default: 3.0)
        output_size: Output video size (width, height)
        captures_root: Root directory containing capture folders
        debug_limit: Optional limit on number of frames for faster iteration (default: None)
        apply_film_grain: Apply film grain and filter effects (default: True)
        audio_file: Optional path to audio file to combine with video (default: None)
        zoom_delay_frames: Number of frames to remain zoomed in before starting zoom out (default: 0)
        frame_interval: Sample every N frames (default: 1, meaning every frame)
    
    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    
    print("\n" + "="*60)
    print("Video Wall Generator")
    print("="*60)
    
    # Get frames for each timelapse (merged by day)
    print("\nCollecting frames from timelapses...")
    frame_lists = []
    for i, timelapse_dir in enumerate(timelapse_dirs):
        print(f"  Processing timelapse {i+1}/{len(timelapse_dirs)}: {timelapse_dir}")
        frames = get_frames_for_timelapse(timelapse_dir, captures_root)
        frame_lists.append(frames)
        print(f"    Found {len(frames)} frames")
    
    # Filter out empty timelapses
    valid_timelapses = [(i, frames) for i, frames in enumerate(frame_lists) if frames]
    if not valid_timelapses:
        print("Error: No frames found in any timelapse")
        return False
    
    # Rebuild frame_lists with only valid timelapses
    frame_lists = [frames for _, frames in valid_timelapses]
    timelapse_dirs = [timelapse_dirs[i] for i, _ in valid_timelapses]
    
    print(f"\nValid timelapses: {len(frame_lists)}")
    
    # Sync frames by timestamp
    print("\nSyncing frames by timestamp...")
    synced_groups = sync_frames_by_timestamp(frame_lists, max_time_diff_seconds=5)
    print(f"Synced {len(synced_groups)} time groups")
    
    # Filter out groups where no frames are available
    synced_groups = [(key, frames) for key, frames in synced_groups if any(f is not None for f in frames)]
    print(f"After filtering empty groups: {len(synced_groups)} time groups")
    
    if not synced_groups:
        print("Error: No synced frames found")
        return False
    
    # Find the first timestamp of the first timelapse and latest timestamp across all timelapses
    start_time_key = None
    end_time_key = None
    
    if frame_lists and frame_lists[0]:
        first_timelapse_first_frame = frame_lists[0][0]
        first_timestamp = extract_timestamp(first_timelapse_first_frame)
        if first_timestamp:
            start_time_key = (first_timestamp.hour, first_timestamp.minute, first_timestamp.second)
            print(f"\nFirst timestamp of first video: {first_timestamp.strftime('%H:%M:%S')}")
    
    # Find the latest time of day (hour:minute:second) across all timelapses
    # We compare only the time portion, ignoring the date
    latest_time_key = None
    latest_timelapse_idx = None
    latest_full_timestamp = None
    
    for idx, timelapse_frames in enumerate(frame_lists):
        if timelapse_frames:
            # Check all frames in this timelapse, not just the last one
            # (in case frames aren't perfectly sorted)
            for frame in timelapse_frames:
                frame_timestamp = extract_timestamp(frame)
                if frame_timestamp:
                    # Compare only time of day (hour, minute, second)
                    time_key = (frame_timestamp.hour, frame_timestamp.minute, frame_timestamp.second)
                    
                    if latest_time_key is None or time_key > latest_time_key:
                        latest_time_key = time_key
                        latest_full_timestamp = frame_timestamp
                        latest_timelapse_idx = idx
    
    if latest_time_key:
        end_time_key = latest_time_key
        print(f"Latest time of day across all videos: {latest_time_key[0]:02d}:{latest_time_key[1]:02d}:{latest_time_key[2]:02d}")
        if latest_timelapse_idx is not None and latest_full_timestamp:
            print(f"  Found in timelapse {latest_timelapse_idx + 1}: {timelapse_dirs[latest_timelapse_idx]}")
            print(f"  Full timestamp: {latest_full_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Filter synced groups to start from first video and end at latest timestamp
    if start_time_key or end_time_key:
        filtered_groups = []
        start_found = not start_time_key  # If no start time, start from beginning
        
        for key, frames in synced_groups:
            # Check if we've reached the start
            if start_time_key and not start_found:
                if key >= start_time_key:
                    start_found = True
                else:
                    continue  # Skip frames before start
            
            # Check if we've passed the end
            if end_time_key and key > end_time_key:
                break  # Stop at end
            
            if start_found:
                filtered_groups.append((key, frames))
        
        if filtered_groups:
            original_count = len(synced_groups)
            synced_groups = filtered_groups
            if start_time_key and end_time_key:
                print(f"Filtered to start from first video and end at latest timestamp: {len(synced_groups)} time groups (from {original_count})")
            elif start_time_key:
                print(f"Filtered to start from first video: {len(synced_groups)} time groups (from {original_count})")
            elif end_time_key:
                print(f"Filtered to end at latest timestamp: {len(synced_groups)} time groups (from {original_count})")
        else:
            print("Warning: Could not filter groups, using all groups")
    
    # Apply frame interval sampling
    if frame_interval > 1:
        original_count = len(synced_groups)
        synced_groups = synced_groups[::frame_interval]
        print(f"Sampling every {frame_interval} frames: {len(synced_groups)} time groups (from {original_count})")
    
    # Apply debug limit if specified
    if debug_limit and debug_limit > 0:
        original_count = len(synced_groups)
        synced_groups = synced_groups[:debug_limit]
        print(f"\nDebug mode: Limited to {len(synced_groups)} frames (from {original_count})")
    
    # Calculate grid layout
    num_timelapses = len(frame_lists)
    grid_rows, grid_cols = calculate_grid_layout(num_timelapses)
    print(f"\nGrid layout: {grid_rows} rows x {grid_cols} cols")
    
    # Create temporary directory for processed frames
    temp_dir = output_path.parent / 'temp_video_wall'
    temp_dir.mkdir(exist_ok=True)
    
    print("\nGenerating video wall frames...")
    
    # Calculate zoom parameters
    # Increase zoom duration by 100% for slower zoom out (2x duration)
    zoom_frames = int(zoom_duration * fps * 2.0)
    total_frames = len(synced_groups)
    total_zoom_frames = zoom_delay_frames + zoom_frames
    
    # Initial zoom factor - zoom in enough so first video fills screen
    # For a grid, we need to zoom in by at least the grid size
    # Use a higher zoom factor (4.0) to ensure full screen coverage
    initial_zoom = 4.0
    final_zoom = 1.0
    
    # Calculate focus positions
    # Start: center of first video
    # End: center of entire video wall
    cell_width = output_size[0] // grid_cols
    cell_height = output_size[1] // grid_rows
    
    # First video position (top-left cell)
    first_video_row = 0
    first_video_col = 0
    start_focus_x = first_video_col * cell_width + cell_width // 2
    start_focus_y = first_video_row * cell_height + cell_height // 2
    
    # Center of video wall
    end_focus_x = output_size[0] // 2
    end_focus_y = output_size[1] // 2
    
    # Generate frames
    last_frames = {}  # Track last valid frames for missing frame handling
    for frame_idx, (time_key, frame_paths) in enumerate(synced_groups):
        # Calculate zoom factor
        if frame_idx < zoom_delay_frames:
            # Stay zoomed in during delay period
            zoom_factor = initial_zoom
            # Focus on first video during delay
            focus_x = start_focus_x
            focus_y = start_focus_y
        elif frame_idx < total_zoom_frames:
            # Zoom out with easing (ease-in-out curve)
            zoom_progress = (frame_idx - zoom_delay_frames) / zoom_frames
            
            # Ease-in-out cubic: smooth start, fast middle, smooth end
            # Formula: t < 0.5 ? 4 * t^3 : 1 - pow(-2 * t + 2, 3) / 2
            if zoom_progress < 0.5:
                eased_progress = 4 * pow(zoom_progress, 3)
            else:
                eased_progress = 1.0 - pow(-2 * zoom_progress + 2, 3) / 2
            
            # Interpolate from initial_zoom to final_zoom
            zoom_range = initial_zoom - final_zoom
            zoom_factor = initial_zoom - (eased_progress * zoom_range)
            
            # Interpolate focus point from first video to center of wall
            focus_x = start_focus_x + (end_focus_x - start_focus_x) * eased_progress
            focus_y = start_focus_y + (end_focus_y - start_focus_y) * eased_progress
        else:
            # Fully zoomed out
            zoom_factor = final_zoom
            focus_x = end_focus_x
            focus_y = end_focus_y
        
        # Create video wall frame
        wall_frame, last_frames = create_video_wall_frame(
            frame_paths, 
            grid_cols, 
            grid_rows, 
            output_size, 
            zoom_factor=zoom_factor,
            focus_x=focus_x,
            focus_y=focus_y,
            last_frames=last_frames
        )
        
        # Save frame (temporary, before film grain if enabled)
        frame_temp = temp_dir / f"wall_frame_temp_{frame_idx:06d}.jpg"
        wall_frame.save(str(frame_temp), 'JPEG', quality=95)
        
        # Apply film grain effect if enabled
        if apply_film_grain:
            frame_path = temp_dir / f"wall_frame_{frame_idx:06d}.jpg"
            add_film_grain_effect(str(frame_temp), str(frame_path))
            # Remove temporary file
            frame_temp.unlink()
        else:
            # Rename temp file to final file
            frame_path = temp_dir / f"wall_frame_{frame_idx:06d}.jpg"
            frame_temp.rename(frame_path)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Generated {frame_idx + 1}/{total_frames} frames")
    
    if apply_film_grain:
        print(f"Generated {total_frames} video wall frames with film grain effects")
    else:
        print(f"Generated {total_frames} video wall frames")
    
    # Create video using FFmpeg
    print(f"\nCreating video at {fps} FPS...")
    
    # Create input list file for FFmpeg
    list_file = temp_dir / 'input_list.txt'
    with open(list_file, 'w') as f:
        for i in range(total_frames):
            frame_path = temp_dir / f"wall_frame_{i:06d}.jpg"
            f.write(f"file '{frame_path.absolute()}'\n")
            f.write(f"duration {1/fps}\n")
        # Add last frame again
        last_frame_path = temp_dir / f"wall_frame_{total_frames-1:06d}.jpg"
        f.write(f"file '{last_frame_path.absolute()}'\n")
    
    # FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
    ]
    
    # Add audio input (YouTube audio or silent track)
    if audio_file and Path(audio_file).exists():
        print(f"Using audio from: {audio_file}")
        ffmpeg_cmd.extend(['-i', str(audio_file)])
    else:
        ffmpeg_cmd.extend(['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo'])  # Generate silent audio track
    
    # Add video filter for additional blur if film grain is enabled
    if apply_film_grain:
        ffmpeg_cmd.extend(['-vf', 'gblur=sigma=0.8'])
    
    # Continue with encoding settings
    ffmpeg_cmd.extend([
        '-c:v', 'libx264',  # H.264 codec
        '-profile:v', 'high',  # High profile for better quality
        '-level', '4.0',
        '-pix_fmt', 'yuv420p',  # Twitter-compatible pixel format
        '-crf', '23' if apply_film_grain else '18',  # Higher CRF = lower quality (23 is good balance, 18 is higher quality)
        '-preset', 'medium',  # Encoding preset
        '-c:a', 'aac',  # AAC audio codec
        '-b:a', '128k',  # Audio bitrate
        '-shortest',  # Match audio length to video length
        '-movflags', '+faststart',  # Enable streaming
        '-r', str(fps),  # Set frame rate
        str(output_path)
    ])
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        print(f"\n✓ Video wall generated successfully: {output_path}")
        print(f"  Timelapses: {num_timelapses}")
        print(f"  Frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        return True
        
    except FileNotFoundError:
        print("\nError: FFmpeg not found. Please install FFmpeg:")
        print("  brew install ffmpeg")
        return False
    except Exception as e:
        print(f"\nError generating video: {e}")
        return False
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a video wall effect from multiple timelapses.\n'
                    'Merges all recordings from each specified day and syncs them by timestamp.\n'
                    'Starts zoomed in on the first timelapse, then zooms out to reveal all.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video wall from multiple timelapse directories
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 captures/2025-11-07_08-37-45
  
  # Specify output file
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 -o video_wall.mp4
  
  # Custom FPS and zoom duration
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --fps 30 --zoom-duration 5.0
  
  # Custom output resolution
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --output-size 3840 2160
  
  # Debug mode: limit to 200 frames for faster iteration
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --debug 200
  
  # Generate with YouTube audio
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --youtube-audio "https://youtube.com/watch?v=..."
  
  # Disable film grain effects
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --no-film-grain
  
  # Add delay before zoom out (stay zoomed in for 72 frames = 3 seconds at 24fps)
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --delay 72
  
  # Sample every 5 frames (faster processing, lower frame rate)
  python video_wall.py captures/2025-11-05_10-47-27 captures/2025-11-06_08-45-44 --interval 5
        """
    )
    
    parser.add_argument(
        'timelapses',
        nargs='+',
        metavar='TIMELAPSE',
        help='One or more timelapse directory paths or date strings (YYYY-MM-DD). '
             'Each timelapse will be merged by day (all recordings from the same date).'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='video_wall.mp4',
        metavar='FILE',
        help='Output video file path (default: video_wall.mp4)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        metavar='FRAMES',
        help='Frames per second for the output video (default: 24)'
    )
    
    parser.add_argument(
        '--zoom-duration',
        type=float,
        default=3.0,
        metavar='SECONDS',
        help='Duration in seconds for the zoom-out transition (default: 3.0)'
    )
    
    parser.add_argument(
        '--output-size',
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=('WIDTH', 'HEIGHT'),
        help='Output video resolution (default: 1920 1080)'
    )
    
    parser.add_argument(
        '--captures-root',
        type=str,
        default='captures',
        metavar='DIR',
        help='Root directory containing capture folders (default: captures)'
    )
    
    parser.add_argument(
        '--debug',
        type=int,
        metavar='FRAMES',
        help='Debug mode: Limit to specified number of frames for faster iteration (e.g., 200)'
    )
    
    parser.add_argument(
        '--youtube-audio',
        type=str,
        metavar='URL',
        help='Optional YouTube video URL to download audio from and combine with the generated video. '
             'The audio will be downloaded and merged with the timelapse video.'
    )
    
    parser.add_argument(
        '--no-film-grain',
        action='store_true',
        help='Disable film grain and filter effects. By default, these effects are enabled '
             'to create a vintage look and reduce text clarity in the video.'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=0,
        metavar='FRAMES',
        help='Number of frames to remain zoomed in on the first timelapse before starting zoom out animation (default: 0)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        metavar='N',
        help='Sample every N frames (default: 1, meaning every frame). '
             'For example, --interval 5 will use every 5th frame.'
    )
    
    args = parser.parse_args()
    
    # Validate output size
    if len(args.output_size) != 2 or args.output_size[0] <= 0 or args.output_size[1] <= 0:
        print("Error: Invalid output size. Must be two positive integers (width height)")
        sys.exit(1)
    
    output_size = tuple(args.output_size)
    
    apply_film_grain = not args.no_film_grain
    
    # Download YouTube audio if specified
    audio_file = None
    temp_audio_path = None
    if args.youtube_audio:
        print("\n" + "="*60)
        print("Downloading YouTube audio...")
        print("="*60)
        try:
            downloader = YoutubeDownloader()
            # Create temporary file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_audio_path = temp_audio.name
            temp_audio.close()
            audio_file = downloader.download_audio(args.youtube_audio, temp_audio_path)
            if not audio_file:
                print("Warning: Failed to download YouTube audio. Proceeding without audio.")
                audio_file = None
                # Clean up temp file if download failed
                if temp_audio_path and Path(temp_audio_path).exists():
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
        except Exception as e:
            print(f"Warning: Error downloading YouTube audio: {e}")
            print("Proceeding without audio.")
            audio_file = None
            # Clean up temp file if exception occurred
            if temp_audio_path and Path(temp_audio_path).exists():
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
    
    # Validate interval
    if args.interval < 1:
        print("Error: --interval must be at least 1")
        sys.exit(1)
    
    # Generate video wall
    success = generate_video_wall(
        args.timelapses,
        args.output,
        fps=args.fps,
        zoom_duration=args.zoom_duration,
        output_size=output_size,
        captures_root=args.captures_root,
        debug_limit=args.debug,
        apply_film_grain=apply_film_grain,
        audio_file=audio_file,
        zoom_delay_frames=args.delay,
        frame_interval=args.interval
    )
    
    # Clean up temporary audio file if it was created
    cleanup_path = audio_file if audio_file else temp_audio_path
    if cleanup_path and Path(cleanup_path).exists():
        try:
            os.unlink(cleanup_path)
        except:
            pass
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

