#!/usr/bin/env python3
"""
Timelapse capture and video generation script for Logitech C925e webcam.
Supports Twitter-compatible video encoding with timestamp overlays.
"""

import cv2
import os
import sys
import argparse
import signal
import time
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess


class TimelapseCapture:
    def __init__(self, interval=30, quality='720p', output_dir=None):
        """
        Initialize timelapse capture.
        
        Args:
            interval: Seconds between captures (default: 30)
            quality: Video quality - '720p' or '1080p' (default: '720p')
            output_dir: Optional output directory path
        """
        self.interval = interval
        self.quality = quality
        self.running = False
        self.frames_captured = 0
        
        # Set resolution based on quality
        self.resolutions = {
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        self.width, self.height = self.resolutions.get(quality, (1280, 720))
        
        # Create output directory with timestamp
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_dir = Path('captures') / timestamp
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Images will be saved to: {self.images_dir}")
        
    def setup_camera(self):
        """Initialize the webcam."""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera. Is the Logitech C925e connected?")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verify resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized at {actual_width}x{actual_height}")
        
        # Warm up the camera
        for _ in range(5):
            self.cap.read()
        
    def capture_frame(self):
        """Capture a single frame and save it with timestamp."""
        ret, frame = self.cap.read()
        
        if not ret:
            print("Failed to capture frame")
            return None
        
        # Get current timestamp
        now = datetime.now()
        timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save frame
        filename = f"frame_{self.frames_captured:06d}_{timestamp_str}.jpg"
        filepath = self.images_dir / filename
        
        # Save with high quality JPEG
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        self.frames_captured += 1
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Captured frame {self.frames_captured}: {filename}")
        
        return filepath, now
        
    def start_capture(self):
        """Start the timelapse capture loop."""
        self.setup_camera()
        self.running = True
        
        print(f"\nStarting timelapse capture:")
        print(f"  Interval: {self.interval} seconds")
        print(f"  Quality: {self.quality}")
        print(f"  Press Ctrl+C to stop and generate video\n")
        
        try:
            while self.running:
                self.capture_frame()
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        print(f"\nTotal frames captured: {self.frames_captured}")
        
    def stop(self):
        """Stop the capture loop."""
        self.running = False


def add_timestamp_overlay(image_path, output_path):
    """
    Add timestamp overlay to an image.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with overlay
    """
    # Extract timestamp from filename
    filename = Path(image_path).stem
    parts = filename.split('_')
    
    if len(parts) >= 4:
        date_str = parts[2]  # YYYY-MM-DD
        time_str = parts[3]  # HH-MM-SS
        
        # Parse the date and time
        date_obj = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H-%M-%S')
        
        # Format for overlay
        day_line = date_obj.strftime('%a %b %d')  # e.g., "Fri Oct 10"
        time_line = date_obj.strftime('%H:%M')     # e.g., "20:13"
    else:
        day_line = "Unknown"
        time_line = "00:00"
    
    # Open image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        # Try common font locations on macOS
        font_paths = [
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
            '/System/Library/Fonts/Helvetica.ttc',
            '/Library/Fonts/Arial.ttf',
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 40)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Position in top right corner
    padding = 20
    
    # Draw text with black outline for better visibility
    for offset_x in [-2, 0, 2]:
        for offset_y in [-2, 0, 2]:
            if offset_x != 0 or offset_y != 0:
                draw.text((img.width - 200 + offset_x, padding + offset_y), 
                         day_line, fill='black', font=font)
                draw.text((img.width - 200 + offset_x, padding + 50 + offset_y), 
                         time_line, fill='black', font=font)
    
    # Draw white text on top
    draw.text((img.width - 200, padding), day_line, fill='white', font=font)
    draw.text((img.width - 200, padding + 50), time_line, fill='white', font=font)
    
    # Save image
    img.save(output_path, 'JPEG', quality=95)


def generate_video(images_dir, output_path, fps=30):
    """
    Generate a Twitter-compatible MP4 video from captured images with overlays.
    
    Args:
        images_dir: Directory containing the captured images
        output_path: Path for the output video file
        fps: Frames per second for the output video (default: 30)
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    
    # Get all image files sorted by filename
    image_files = sorted(images_dir.glob('frame_*.jpg'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return False
    
    print(f"\nGenerating video from {len(image_files)} frames...")
    
    # Create temporary directory for overlay images
    temp_dir = images_dir.parent / 'temp_overlay'
    temp_dir.mkdir(exist_ok=True)
    
    print("Adding timestamp overlays to frames...")
    overlay_files = []
    for i, img_file in enumerate(image_files):
        output_file = temp_dir / f"overlay_{i:06d}.jpg"
        add_timestamp_overlay(img_file, output_file)
        overlay_files.append(output_file)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} frames")
    
    print(f"All frames processed with overlays")
    
    # Create video using FFmpeg with Twitter-compatible settings
    print(f"Creating MP4 video at {fps} FPS...")
    
    # Create a text file with list of images for FFmpeg
    list_file = temp_dir / 'input_list.txt'
    with open(list_file, 'w') as f:
        for overlay_file in overlay_files:
            f.write(f"file '{overlay_file.absolute()}'\n")
            f.write(f"duration {1/fps}\n")
        # Add the last image again without duration to ensure it's included
        f.write(f"file '{overlay_files[-1].absolute()}'\n")
    
    # FFmpeg command for Twitter-compatible video
    # H.264 video codec, AAC audio codec, yuv420p pixel format
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'concat',
        '-safe', '0',
        '-i', str(list_file),
        '-f', 'lavfi',
        '-i', 'anullsrc=r=44100:cl=stereo',  # Generate silent audio track
        '-c:v', 'libx264',  # H.264 codec
        '-profile:v', 'high',  # High profile for better quality
        '-level', '4.0',
        '-pix_fmt', 'yuv420p',  # Twitter-compatible pixel format
        '-c:a', 'aac',  # AAC audio codec
        '-b:a', '128k',  # Audio bitrate
        '-shortest',  # Match audio length to video length
        '-movflags', '+faststart',  # Enable streaming
        '-r', str(fps),  # Set frame rate
        str(output_path)
    ]
    
    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        print(f"\n✓ Video generated successfully: {output_path}")
        print(f"  Frames: {len(image_files)}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {len(image_files)/fps:.2f} seconds")
        
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
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Timelapse capture and video generation for Logitech C925e',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start capturing with default settings (30s interval, 720p)
  python timelapse.py
  
  # Capture with custom interval and quality
  python timelapse.py --interval 60 --quality 1080p
  
  # Generate video from existing capture folder
  python timelapse.py --generate-video captures/2025-10-10_14-30-00
  
  # Generate video with custom FPS
  python timelapse.py --generate-video captures/2025-10-10_14-30-00 --fps 60
        """
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Seconds between captures (default: 30)'
    )
    
    parser.add_argument(
        '--quality',
        choices=['720p', '1080p'],
        default='720p',
        help='Video quality: 720p or 1080p (default: 720p)'
    )
    
    parser.add_argument(
        '--generate-video',
        type=str,
        metavar='FOLDER',
        help='Generate video from existing capture folder (skips capture)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output video (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Mode 1: Generate video from existing folder
    if args.generate_video:
        capture_dir = Path(args.generate_video)
        
        if not capture_dir.exists():
            print(f"Error: Directory not found: {capture_dir}")
            sys.exit(1)
        
        images_dir = capture_dir / 'images'
        if not images_dir.exists():
            print(f"Error: Images directory not found: {images_dir}")
            sys.exit(1)
        
        output_video = capture_dir / 'timelapse.mp4'
        success = generate_video(images_dir, output_video, fps=args.fps)
        
        sys.exit(0 if success else 1)
    
    # Mode 2: Capture mode
    timelapse = TimelapseCapture(
        interval=args.interval,
        quality=args.quality
    )
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nStopping capture...")
        timelapse.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start capturing
        timelapse.start_capture()
        
        # After capture is done, generate video
        if timelapse.frames_captured > 0:
            print("\n" + "="*60)
            print("Generating video...")
            print("="*60)
            
            output_video = timelapse.output_dir / 'timelapse.mp4'
            success = generate_video(timelapse.images_dir, output_video, fps=args.fps)
            
            if success:
                print("\n" + "="*60)
                print("Timelapse complete!")
                print("="*60)
            else:
                print("\nVideo generation failed. You can try again with:")
                print(f"  python timelapse.py --generate-video {timelapse.output_dir}")
        else:
            print("\nNo frames captured.")
            
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

