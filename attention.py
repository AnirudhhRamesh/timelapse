#!/usr/bin/env python3
"""
Attention tracking system that monitors screen activity and classifies tasks.
Uses Ollama vision-language models to analyze screenshots and track focus.
"""

import mss
import argparse
import time
import csv
import os
import sys
import json
import base64
import requests
from datetime import datetime
from pathlib import Path
from PIL import Image
import io
import subprocess
import threading
import pandas as pd
import matplotlib.pyplot as plt


class AttentionTracker:
    def __init__(self, interval=300, model="qwen3-vl:4b-instruct-q4_K_M", ollama_url="http://localhost:11434"):
        """
        Initialize attention tracker.
        
        Args:
            interval: Seconds between screen captures (default: 300, i.e., 5 minutes)
            model: Ollama model name for vision (default: "qwen3-vl:4b-instruct-q4_K_M")
            ollama_url: URL of Ollama API (default: "http://localhost:11434")
        """
        self.interval = interval
        self.model = model
        self.ollama_url = ollama_url
        self.running = False
        
        # Create attention log directory structure
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.base_dir = Path('attention') / timestamp
        self.log_dir = self.base_dir / 'log'
        self.screenshots_dir = self.base_dir / 'screenshots'
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / 'attention_log.csv'
        
        # Initialize CSV file
        self._init_csv()
        
        print(f"Attention tracker initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Screenshots directory: {self.screenshots_dir}")
        print(f"CSV log: {self.csv_path}")
        print(f"Model: {self.model}")
        print(f"Interval: {self.interval} seconds")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'date', 'time', 'type', 'category', 'description', 'screenshot'])
    
    def get_monitors(self):
        """Get list of all available monitors, identifying primary vs external."""
        with mss.mss() as sct:
            # Index 0 is all monitors combined, 1+ are individual monitors
            # Index 1 is typically the primary/internal monitor
            monitors = []
            for i in range(1, len(sct.monitors)):
                is_primary = (i == 1)  # Index 1 is usually primary/internal
                monitors.append({
                    'index': i,
                    'monitor': sct.monitors[i],
                    'name': f"Monitor {i}",
                    'is_primary': is_primary
                })
            return monitors
    
    def capture_screen(self, monitor_info):
        """
        Capture screenshot of a specific monitor.
        
        Args:
            monitor_info: dict with 'index' and 'monitor' keys
        
        Returns:
            PIL Image object
        """
        with mss.mss() as sct:
            screenshot = sct.grab(monitor_info['monitor'])
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
    
    def combine_monitors(self, monitor_images):
        """
        Combine multiple monitor screenshots into a single image.
        External displays are given priority (placed first and larger).
        
        Args:
            monitor_images: List of tuples (img, monitor_info) where monitor_info has 'is_primary' flag
        
        Returns:
            Combined PIL Image
        """
        if len(monitor_images) == 1:
            return monitor_images[0][0]
        
        # Separate primary and external monitors
        primary_imgs = []
        external_imgs = []
        
        for img, monitor_info in monitor_images:
            if monitor_info['is_primary']:
                primary_imgs.append((img, monitor_info))
            else:
                external_imgs.append((img, monitor_info))
        
        # Prioritize external displays - place them first
        ordered_monitors = external_imgs + primary_imgs
        
        # Calculate target height (scale down if monitors are very large)
        max_height = max(img.height for img, _ in ordered_monitors)
        # Cap at reasonable size for API (keep total width reasonable too)
        target_height = min(max_height, 1200)  # Cap at 1200px to avoid huge images
        
        # Resize all images, maintaining aspect ratio
        # External displays get 1.3x larger than primary to emphasize their importance
        resized_imgs = []
        total_width = 0
        
        for img, monitor_info in ordered_monitors:
            # External displays get larger size (1.3x weight)
            # Primary gets normal size (1.0x weight)
            scale_factor = 1.3 if not monitor_info['is_primary'] else 1.0
            scaled_height = int(target_height * scale_factor)
            
            # Maintain aspect ratio
            aspect_ratio = img.width / img.height
            scaled_width = int(scaled_height * aspect_ratio)
            
            resized_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            resized_imgs.append(resized_img)
            total_width += scaled_width
        
        # If combined width is too large, scale everything down proportionally
        max_combined_width = 4000  # Reasonable max width for API
        if total_width > max_combined_width:
            scale_down = max_combined_width / total_width
            for i in range(len(resized_imgs)):
                new_width = int(resized_imgs[i].width * scale_down)
                new_height = int(resized_imgs[i].height * scale_down)
                resized_imgs[i] = resized_imgs[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            total_width = max_combined_width
        
        # Create combined image (horizontal layout)
        combined_height = max(img.height for img in resized_imgs)
        combined_img = Image.new('RGB', (total_width, combined_height))
        
        x_offset = 0
        for img in resized_imgs:
            # Center vertically if heights differ
            y_offset = (combined_height - img.height) // 2
            combined_img.paste(img, (x_offset, y_offset))
            x_offset += img.width
        
        return combined_img
    
    def save_screenshot(self, img, timestamp_str, monitor_index=None):
        """
        Save screenshot to screenshots directory with compression.
        Uses JPEG format with quality optimized for text readability.
        
        Args:
            img: PIL Image object
            timestamp_str: Timestamp string for filename (format: YYYY-MM-DD_HH-MM-SS)
            monitor_index: Optional monitor number for filename
        
        Returns:
            Path to saved screenshot
        """
        # Convert to RGB if needed (JPEG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if image is very large to reduce file size further
        # Keep high resolution for text readability but cap at reasonable size
        max_width = 2560  # Keep detail but reduce storage
        if img.width > max_width:
            aspect_ratio = img.height / img.width
            new_height = int(max_width * aspect_ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        if monitor_index is not None:
            filename = f"screenshot_{timestamp_str}_monitor{monitor_index}.jpg"
        else:
            filename = f"screenshot_{timestamp_str}.jpg"
        filepath = self.screenshots_dir / filename
        
        # Save as JPEG with quality 85 (good balance: readable text, good compression)
        # optimize=True reduces file size further
        img.save(filepath, 'JPEG', quality=85, optimize=True)
        return filepath
    
    def image_to_base64(self, img):
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def classify_screen(self, img):
        """
        Classify screen content using Ollama vision model.
        
        Returns:
            dict with 'type', 'category', 'description'
        """
        # Convert image to base64
        img_b64 = self.image_to_base64(img)
        
        # Define valid categories
        course_categories = [
            "Mixed Reality",
            "Physically-based Simulation in Computer Graphics",
            "Deep Learning",
            "Macroeconomics",
            "Information Security Lab",
            "Data Management Systems",
            "Computer Architecture Seminar"
        ]
        
        distraction_categories = [
            "LinkedIn",
            "Twitter",
            "YouTube (if not course related)",
            "Other",
            "Not Working"
        ]
        
        # Create classification prompt
        prompt = f"""Analyze this screenshot and classify what the user is working on.

Respond with ONLY a JSON object in this exact format:
{{
  "type": "Course" OR "Distraction",
  "category": "specific category name",
  "description": "brief description of the activity"
}}

VALID TYPES:
- "Course": for academic coursework
- "Distraction": for non-work activities

VALID CATEGORIES FOR "Course":
{chr(10).join(f'- "{cat}"' for cat in course_categories)}

VALID CATEGORIES FOR "Distraction":
{chr(10).join(f'- "{cat}"' for cat in distraction_categories)}

Category descriptions:

Course categories:
- "Mixed Reality": working on VLAs/OpenPi/egocentric datasets, coding in Python, reading robotics papers
- "Physically-based Simulation in Computer Graphics": reading CG/ODEs/physics lectures, coding in C++/C, running simulations/visualizations
- "Deep Learning": reading papers on 3D Gaussian Splatting, LoRAs, Stable Diffusion, coding in Python/PyTorch, gaussian splat visualizations
- "Macroeconomics": reading economics/macro lecture notes, reading about companies/wikipedia
- "Information Security Lab": reading infosec lectures (buffer overflows, crypto, vulnerabilities), doing security exercises/CTF
- "Data Management Systems": reading system design/DB lectures, PDFs on large-scale data systems, databases
- "Computer Architecture Seminar": Moodle quiz on computer architecture, watching Onur Mutlu YouTube lectures on hardware

Distraction categories:
- "LinkedIn": LinkedIn website/app
- "Twitter": Twitter/X website/app
- "YouTube (if not course related)": YouTube videos that are NOT educational/course-related
- "Other": any other distraction (specify in description)
- "Not Working": idle screen, not actively working

Description examples:
- "Implementing model architecture"
- "Studying lecture notes"
- "Doomscrolling"
- "Reading research papers"
- "Writing code for simulation"

IMPORTANT: You MUST use one of the exact category names listed above. Do not create new categories. If uncertain, choose the closest match from the list."""

        # Prepare API request
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            response_text = result.get('response', '{}')
            
            # Try to extract JSON from response
            try:
                # Clean up response if it has markdown code blocks
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                classification = json.loads(response_text)
                
                # Validate and normalize classification
                validated = self._validate_classification(classification, course_categories, distraction_categories)
                
                return validated
            except json.JSONDecodeError:
                # Fallback: try to extract from text
                print(f"Warning: Could not parse JSON response: {response_text[:200]}")
                return {
                    'type': 'Unknown',
                    'category': 'Unknown',
                    'description': response_text[:100]
                }
                
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return {
                'type': 'Error',
                'category': 'API Error',
                'description': str(e)
            }
    
    def _validate_classification(self, classification, course_categories, distraction_categories):
        """
        Validate and normalize classification to ensure it matches allowed types and categories.
        
        Args:
            classification: dict with 'type', 'category', 'description'
            course_categories: list of valid course category names
            distraction_categories: list of valid distraction category names
        
        Returns:
            Validated classification dict
        """
        # Normalize type
        type_val = classification.get('type', '').strip()
        type_lower = type_val.lower()
        
        # Map various type formats to valid types
        if 'course' in type_lower or 'studies' in type_lower or 'study' in type_lower:
            validated_type = 'Course'
        elif 'distraction' in type_lower or 'distract' in type_lower:
            validated_type = 'Distraction'
        else:
            # Default to Course if unclear
            validated_type = 'Course'
        
        # Normalize category
        category_val = classification.get('category', '').strip()
        category_lower = category_val.lower()
        
        # Find matching category (case-insensitive, partial match)
        validated_category = None
        
        if validated_type == 'Course':
            # Try exact match first
            for cat in course_categories:
                if category_val == cat:
                    validated_category = cat
                    break
            
            # Try case-insensitive match
            if not validated_category:
                for cat in course_categories:
                    if category_lower == cat.lower():
                        validated_category = cat
                        break
            
            # Try partial match
            if not validated_category:
                for cat in course_categories:
                    if cat.lower() in category_lower or category_lower in cat.lower():
                        validated_category = cat
                        break
            
            # Default to first course category if no match
            if not validated_category:
                validated_category = course_categories[0]
        else:  # Distraction
            # Try exact match first
            for cat in distraction_categories:
                if category_val == cat:
                    validated_category = cat
                    break
            
            # Try case-insensitive match
            if not validated_category:
                for cat in distraction_categories:
                    if category_lower == cat.lower():
                        validated_category = cat
                        break
            
            # Try partial match (handle "Other: {specify}" or "Other" variations)
            if not validated_category:
                # Check if it starts with "other" (case-insensitive)
                if category_lower.startswith('other'):
                    validated_category = "Other"
                else:
                    # Try partial match for other categories
                    for cat in distraction_categories:
                        cat_base = cat.split(':')[0].strip().lower()
                        if cat_base in category_lower or category_lower in cat_base:
                            validated_category = cat
                            break
            
            # Default to first distraction category if no match
            if not validated_category:
                validated_category = distraction_categories[0]
        
        # Get description (truncate if too long)
        description = classification.get('description', '').strip()
        if len(description) > 200:
            description = description[:200]
        
        return {
            'type': validated_type,
            'category': validated_category,
            'description': description if description else f"Working on {validated_category}"
        }
    
    def show_distraction_popup(self, category):
        """Show popup window warning about distraction using macOS native alert."""
        # Use osascript for macOS native alert (works reliably from any thread)
        # Escape quotes for AppleScript
        escaped_category = category.replace('"', '\\"').replace('\\', '\\\\')
        message_line1 = f"You're distracted by: {escaped_category}"
        message_line2 = "Get back to work!"
        
        # AppleScript uses "return" for newlines
        script = f'display dialog "{message_line1}\\n\\n{message_line2}" with title "Focus Alert! 🚨" buttons {{"OK"}} default button "OK" with icon caution'
        
        try:
            # Use osascript to show native macOS alert
            # No timeout - dialog stays open until user clicks OK
            # Don't capture output to avoid interfering with dialog display
            subprocess.run(
                ['osascript', '-e', script],
                check=False,
                capture_output=False,
                stderr=subprocess.DEVNULL  # Suppress stderr to avoid noise
            )
        except Exception as e:
            # Fallback: just print to console if alert fails
            print(f"⚠️  Focus Alert: You're distracted by {category}!")
    
    def log_classification(self, classification, screenshot_filename):
        """Log classification to CSV file."""
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        date = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                date,
                time_str,
                classification['type'],
                classification['category'],
                classification['description'],
                screenshot_filename
            ])
        
        print(f"[{time_str}] {classification['type']} - {classification['category']}")
    
    def check_distraction(self, classification):
        """Check if classification indicates distraction and show popup."""
        if classification['type'] == 'Distraction':
            # Show popup in separate thread to avoid blocking
            # Using osascript, so it's safe to call from a thread
            popup_thread = threading.Thread(
                target=self.show_distraction_popup,
                args=(classification['category'],),
                daemon=True
            )
            popup_thread.start()
            return True
        return False
    
    def start_tracking(self):
        """Start the attention tracking loop."""
        self.running = True
        
        # Get all available monitors
        monitors = self.get_monitors()
        if not monitors:
            print("Error: No monitors detected!")
            return
        
        print(f"\nStarting attention tracking...")
        print(f"Detected {len(monitors)} monitor(s):")
        external_count = 0
        for monitor_info in monitors:
            monitor_type = "Primary/Internal" if monitor_info['is_primary'] else "External"
            print(f"  - {monitor_info['name']} ({monitor_type})")
            if not monitor_info['is_primary']:
                external_count += 1
        
        if external_count > 0:
            print(f"\nOnly external display(s) will be used for classification.")
            print(f"All monitors will be captured and saved, but classification is based on external display only.")
        else:
            print(f"\nWarning: No external monitor detected. Will use primary monitor for classification.")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Capture all monitors
                capture_start_time = time.time()
                now = datetime.now()
                timestamp_str = now.strftime('%Y-%m-%d_%H-%M-%S')
                print(f"\nCapturing all screens at {now.strftime('%H:%M:%S')}...")
                
                # Calculate next capture time (300 seconds from now, regardless of processing time)
                next_capture_time = capture_start_time + self.interval
                
                # Capture all monitors and save individually
                monitor_data = []  # List of (screenshot, monitor_info, screenshot_path)
                external_monitor = None
                external_screenshot_path = None
                
                for monitor_info in monitors:
                    print(f"  Capturing {monitor_info['name']}...")
                    screenshot = self.capture_screen(monitor_info)
                    
                    # Save individual screenshot
                    screenshot_path = self.save_screenshot(screenshot, timestamp_str, monitor_info['index'])
                    print(f"    Saved: {screenshot_path.name}")
                    
                    monitor_data.append((screenshot, monitor_info, screenshot_path))
                    
                    # Identify external monitor (first non-primary monitor)
                    if not monitor_info['is_primary'] and external_monitor is None:
                        external_monitor = (screenshot, monitor_info)
                        external_screenshot_path = screenshot_path
                
                # If no external monitor found, use primary monitor as fallback
                if external_monitor is None:
                    print("  Warning: No external monitor found, using primary monitor for classification")
                    # Find primary monitor
                    for screenshot, monitor_info, screenshot_path in monitor_data:
                        if monitor_info['is_primary']:
                            external_monitor = (screenshot, monitor_info)
                            external_screenshot_path = screenshot_path
                            break
                
                if external_monitor is None:
                    print("  Error: No monitors available for classification")
                    # Sleep until next capture time
                    sleep_duration = next_capture_time - time.time()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    continue
                
                # Classify only the external monitor
                external_img, external_info = external_monitor
                print(f"  Classifying {external_info['name']} (external display) content...")
                
                # Time the classification
                classification_start_time = time.time()
                classification = self.classify_screen(external_img)
                classification_duration = time.time() - classification_start_time
                print(f"  Classification completed in {classification_duration:.2f} seconds")
                
                # Get the screenshot filename for the external monitor
                external_screenshot_filename = external_screenshot_path.name
                
                # Log classification
                self.log_classification(classification, external_screenshot_filename)
                
                # Check for distractions
                self.check_distraction(classification)
                
                # Wait until next capture time (ensures captures happen every 5 minutes)
                current_time = time.time()
                sleep_duration = next_capture_time - current_time
                
                if sleep_duration > 0:
                    print(f"Waiting {sleep_duration:.1f} seconds until next capture...\n")
                    time.sleep(sleep_duration)
                else:
                    print(f"Warning: Processing took {current_time - capture_start_time:.1f} seconds, next capture is overdue!\n")
                
        except KeyboardInterrupt:
            print("\n\nAttention tracking stopped by user")
        finally:
            self.running = False
            print(f"\nTracking session complete.")
            print(f"Log saved to: {self.csv_path}")
            print(f"Screenshots saved to: {self.screenshots_dir}")
    
    def stop(self):
        """Stop the tracking loop."""
        self.running = False


def analyze_attention_log(csv_path):
    """
    Analyze attention log and generate pie chart visualization.
    
    Args:
        csv_path: Path to CSV log file
    """
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found: {csv_path}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("Error: Log file is empty")
        return
    
    # Calculate total time (assuming each entry is 300 seconds by default)
    # You can adjust this based on your actual interval
    interval = 300  # Default interval in seconds (5 minutes)
    total_minutes = len(df) * (interval / 60)
    
    print(f"\nTotal tracking time: {total_minutes:.1f} minutes")
    print(f"Total entries: {len(df)}")
    
    # Group by type and category
    type_counts = df['type'].value_counts()
    category_counts = df['category'].value_counts()
    
    print("\n=== Time Breakdown by Type ===")
    for type_name, count in type_counts.items():
        minutes = count * (interval / 60)
        percentage = (count / len(df)) * 100
        print(f"{type_name}: {minutes:.1f} minutes ({percentage:.1f}%)")
    
    print("\n=== Time Breakdown by Category ===")
    for category, count in category_counts.items():
        minutes = count * (interval / 60)
        percentage = (count / len(df)) * 100
        print(f"{category}: {minutes:.1f} minutes ({percentage:.1f}%)")
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Pie chart by type
    ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Focus Distribution by Type')
    
    # Pie chart by category (top 10)
    top_categories = category_counts.head(10)
    ax2.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Top 10 Categories')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(csv_path).parent / 'attention_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    # Show plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Attention tracking system using vision-language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start tracking with default 5 minute (300 second) interval
  python attention.py
  
  # Track with custom interval (2 minutes = 120 seconds)
  python attention.py --interval 120
  
  # Use different Ollama model
  python attention.py --model gemma3:4b
  python attention.py --model gemma3:12b
  python attention.py --model llava:7b
  
  # Analyze existing log file
  python attention.py --analyze attention/2025-11-05_09-00-00/log/attention_log.csv
        """
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Seconds between screen captures (default: 300, i.e., 5 minutes)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen3-vl:4b-instruct-q4_K_M',
        help='Ollama vision model name (default: qwen3-vl:4b-instruct-q4_K_M). '
             'Other options: qwen3-vl:8b-instruct-q4_K_M, gemma3:4b, gemma3:12b, gemma3:27b, llava, llava:7b, llava:13b, bakllava, etc.'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--analyze',
        type=str,
        metavar='CSV_PATH',
        help='Analyze existing log file and generate visualization'
    )
    
    args = parser.parse_args()
    
    # Mode 1: Analyze mode
    if args.analyze:
        analyze_attention_log(args.analyze)
        sys.exit(0)
    
    # Mode 2: Tracking mode
    tracker = AttentionTracker(
        interval=args.interval,
        model=args.model,
        ollama_url=args.ollama_url
    )
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nStopping attention tracker...")
        tracker.stop()
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        tracker.start_tracking()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

