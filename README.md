# Timelapse Capture Tool

A Python script for capturing timelapse videos using a Logitech C925e webcam with timestamp overlays and Twitter-compatible encoding.

## Features

- 📸 Automatic frame capture at configurable intervals
- 🎥 Twitter-compatible MP4 video generation with H.264 codec
- ⏰ Timestamp overlay (date and time) on each frame
- 📁 Organized folder structure with dated captures
- 🎬 Separate video generation from existing captures
- ⚡ High-quality JPEG compression for efficient storage

## Prerequisites

### 1. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. FFmpeg

FFmpeg is required for video generation. Install it using Homebrew:

```bash
brew install ffmpeg
```

## Usage

### Basic Timelapse Capture

Start capturing with default settings (30-second interval, 720p quality):

```bash
python timelapse.py
```

Press `Ctrl+C` to stop capturing. The video will be automatically generated.

### Custom Capture Settings

Capture with custom interval and quality:

```bash
# Capture every 60 seconds in 1080p
python timelapse.py --interval 60 --quality 1080p

# Capture every 10 seconds in 720p
python timelapse.py --interval 10 --quality 720p
```

### Arguments

- `--interval`: Seconds between captures (default: 30)
- `--quality`: Video quality - `720p` or `1080p` (default: `720p`)
- `--fps`: Frames per second for output video (default: 30)
- `--generate-video`: Generate video from existing capture folder (skips capture)

### Regenerate Video from Existing Captures

If you want to regenerate the video with different settings:

```bash
# Generate video with default 30 FPS
python timelapse.py --generate-video captures/2025-10-10_14-30-00

# Generate video with 60 FPS
python timelapse.py --generate-video captures/2025-10-10_14-30-00 --fps 60
```

## Output Structure

The script creates the following folder structure:

```
timelapse/
├── captures/
│   └── 2025-10-10_14-30-00/        # Timestamped capture session
│       ├── images/                  # Original captured frames
│       │   ├── frame_000000_2025-10-10_14-30-00.jpg
│       │   ├── frame_000001_2025-10-10_14-30-30.jpg
│       │   └── ...
│       └── timelapse.mp4           # Generated video
```

## Video Features

### Timestamp Overlay

Each frame in the video includes a timestamp overlay in the top-right corner showing:

- **Line 1**: Day and date (e.g., "Fri Oct 10")
- **Line 2**: Time in 24-hour format (e.g., "20:13")

### Twitter Compatibility

The generated videos are optimized for Twitter with:

- H.264 video codec (High profile)
- AAC audio codec (silent audio track)
- yuv420p pixel format
- Fast-start enabled for streaming
- MP4 container format

## Examples

### Example 1: Daily Work Timelapse

Capture your workspace every 30 seconds throughout the day:

```bash
python timelapse.py --interval 30 --quality 1080p
```

### Example 2: Plant Growth Timelapse

Capture plant growth every 5 minutes (300 seconds):

```bash
python timelapse.py --interval 300 --quality 720p
```

### Example 3: Sunset Timelapse

Capture sunset every 10 seconds in high quality:

```bash
python timelapse.py --interval 10 --quality 1080p
```

## Troubleshooting

### Camera Not Found

If you see "Could not open camera", ensure:

1. The Logitech C925e is connected via USB
2. No other application is using the camera
3. Camera permissions are granted to Terminal/Python

### FFmpeg Not Found

If video generation fails with "FFmpeg not found":

```bash
brew install ffmpeg
```

### Low Disk Space

Each capture session can use significant disk space:

- 720p JPEG: ~200-500 KB per frame
- 1080p JPEG: ~500 KB - 1 MB per frame

For a 1-hour timelapse with 30-second intervals (120 frames):

- 720p: ~24-60 MB
- 1080p: ~60-120 MB

## Tips

1. **Test First**: Run a short test capture to verify camera settings before long sessions
2. **Lighting**: Ensure consistent lighting for best results
3. **Stability**: Mount the camera on a stable surface or tripod
4. **Storage**: Monitor disk space for long captures
5. **Preview**: Check the first few captured frames in the `images/` folder

## License

MIT License - Feel free to modify and use as needed!
