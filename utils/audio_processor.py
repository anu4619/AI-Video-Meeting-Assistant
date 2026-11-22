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

#data = download_youtube_audio('https://www.youtube.com/watch?v=XN3xNJvWXsc')


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav" 
    audio = AudioSegment.from_file(input_path) #detecting and converting any form like .mp3, .mp4 to wav
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz #dual audio converted into mono(so that same audio speed in both earphones)
    audio.export(output_path, format="wav")
    return output_path

#data_final = convert_to_wav(data)

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 #in ms

    chunks = [] #storing all the video chunks

    for i, start in enumerate(range(0,len(audio),chunk_ms)): #chunk larger videos into chunks of 10 min videos
        chunk = audio[start : start + chunk_ms] #from 0th ms to 600000 ms(10mins)
        chunk_path = f"{wav_path}_chunk_{i}.wav" #adding .wav to all
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

#print(chunk_audio(data_final))

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...") 
        wav_path = convert_to_wav(source)  #processing local files

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks