
import yt_dlp
from pydub import AudioSegment
import os
import subprocess

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# Download YouTube audio
def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        filename = os.path.splitext(
            ydl.prepare_filename(info)
        )[0] + ".wav"

    return filename


# Convert local audio/video to WAV
def convert_to_wav(input_path: str) -> str:

    output_path = (
        os.path.splitext(input_path)[0] + "_converted.wav"
    )

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


# Audio enhancement and noise reduction
def enhance_audio(input_path: str) -> str:

    print("Enhancing audio and reducing background noise...")

    output_path = (
        os.path.splitext(input_path)[0] + "_enhanced.wav"
    )

    audio_filters = (
        "highpass=f=80,"
        "lowpass=f=7600,"
        "afftdn=nr=10:nf=-25,"
        "loudnorm=I=-20:TP=-2:LRA=11"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", audio_filters,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    print("Audio enhancement completed.")

    return output_path


# Split audio into chunks
def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        chunk_path = f"{wav_path}_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


# Main audio processing pipeline
def process_input(source: str) -> list:

    if source.startswith(("http://", "https://")):

        print("Detected YouTube URL. Downloading audio...")

        wav_path = download_youtube_audio(source)

    else:

        print("Detected local file. Converting to WAV...")

        wav_path = convert_to_wav(source)

    # Apply noise reduction and audio enhancement
    enhanced_path = enhance_audio(wav_path)

    # Split enhanced audio into chunks
    print("Chunking audio...")

    chunks = chunk_audio(enhanced_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")

    return chunks
