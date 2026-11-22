import yt_dlp
from pydub import AudioSegment
import os
DOWNLOAD_DIR= 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

#capturing the audio of a youtube video in wav fromat using yt_dlp
def download_youtube_audio(url : str)->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

#print(download_youtube_audio('https://www.youtube.com/watch?v=4LINhsEZEDI'))

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav" 
    audio = AudioSegment.from_file(input_path) #detecting and converting any form like .mp3, .mp4 to wav
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz #dual audio converted into mono(so that same audio speed in both earphones)
    audio.export(output_path, format="wav")
    return output_path