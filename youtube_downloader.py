import pytubefix
from pytubefix import YouTube
import subprocess
import os

class YoutubeDownloader:
    def __init__(self):
        self.check_ffmpeg()

    def check_ffmpeg(self):
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("ffmpeg is not installed. Installing ffmpeg...")
            try:
                # This assumes you're on a Debian-based system. Adjust if needed.
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)
                print("ffmpeg installed successfully.")
            except subprocess.CalledProcessError:
                print("Failed to install ffmpeg. Please install it manually.")
                exit(1)

    def download(self, url, start_time=None, end_time=None):
        try:
            yt = YouTube(url)
            print(f"Title: {yt.title}")
            print(f"Views: {yt.views}")
            
            # Get the highest resolution video stream (1080p or 720p)
            video_stream = yt.streams.filter(progressive=False, file_extension='mp4').order_by('resolution').desc().first()
            if video_stream.resolution not in ['1080p', '720p']:
                video_stream = yt.streams.filter(progressive=False, file_extension='mp4', resolution='720p').first()
            
            # Get the highest quality audio stream
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
            
            print(f"Selected video resolution: {video_stream.resolution}")
            print(f"Selected audio bitrate: {audio_stream.abr}")
            
            # Download video
            print("Downloading video...")
            video_file = video_stream.download(filename_prefix="video_")
            
            # Download audio
            print("Downloading audio...")
            audio_file = audio_stream.download(filename_prefix="audio_")
            
            # Merge video and audio using ffmpeg
            print("Merging video and audio...")
            output_file = f"{yt.title.replace('/', '_')}.mp4"  # Sanitize filename
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_file,
                "-i", audio_file,
                "-c:v", "libx264",  # Use H.264 codec for video
                "-c:a", "aac",  # Use AAC codec for audio
                "-strict", "experimental",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p"  # Ensure pixel format is compatible
            ]

            # Add time range parameters if specified
            if start_time:
                ffmpeg_command.extend(["-ss", start_time])
            if end_time:
                ffmpeg_command.extend(["-to", end_time])

            ffmpeg_command.append(output_file)
            
            subprocess.run(ffmpeg_command, check=True)
            
            # Remove temporary files
            os.remove(video_file)
            os.remove(audio_file)
            
            print("Download and merge completed!")
            return output_file
        except pytubefix.exceptions.PytubeError as e:
            print(f"An error occurred: {str(e)}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running ffmpeg: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        return None

    def download_audio(self, url, output_path=None):
        """
        Download only the audio from a YouTube video as MP3.
        
        Args:
            url: YouTube video URL
            output_path: Optional path for output file. If None, uses video title.
        
        Returns:
            Path to downloaded audio file, or None if failed
        """
        try:
            yt = YouTube(url)
            print(f"Title: {yt.title}")
            print(f"Views: {yt.views}")
            
            # Get the highest quality audio stream
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
            
            if not audio_stream:
                print("No audio stream found")
                return None
            
            print(f"Selected audio bitrate: {audio_stream.abr}")
            
            # Download audio
            print("Downloading audio...")
            audio_file = audio_stream.download(filename_prefix="audio_")
            
            # Convert to MP3 using ffmpeg
            if output_path is None:
                output_file = f"{yt.title.replace('/', '_').replace('|', '_')}.mp3"
            else:
                output_file = output_path
            
            print("Converting to MP3...")
            ffmpeg_command = [
                "ffmpeg",
                "-i", audio_file,
                "-vn",  # No video
                "-acodec", "libmp3lame",  # MP3 codec
                "-ab", "192k",  # Audio bitrate
                "-ar", "44100",  # Sample rate
                "-y",  # Overwrite output file
                output_file
            ]
            
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Remove temporary file
            os.remove(audio_file)
            
            print(f"Audio downloaded successfully: {output_file}")
            return output_file
            
        except pytubefix.exceptions.PytubeError as e:
            print(f"An error occurred: {str(e)}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running ffmpeg: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        return None

# Usage example:
# yt_downloader = YoutubeDownloader()
# yt_downloader.download(url, start_time="00:01:30", end_time="00:03:45")
# yt_downloader.download_audio(url, "output.mp3")

