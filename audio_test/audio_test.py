
import os
import subprocess
import time
import json
import re

import torch
import whisper
from jiwer import wer, process_words
from pydub import AudioSegment

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paste your YouTube URL here
YOUTUBE_URL = "https://www.youtube.com/shorts/t46_JOiVPuk"

# Exact manually verified transcript
REFERENCE_FILE = os.path.join(BASE_DIR, "reference.txt")

FFMPEG_OUTPUT = os.path.join(BASE_DIR, "ffmpeg_enhanced.wav")
DEEPFILTER_OUTPUT = os.path.join(BASE_DIR, "deepfilter_enhanced.wav")

RESULTS_FILE = os.path.join(BASE_DIR, "wer_results.txt")
JSON_RESULTS_FILE = os.path.join(BASE_DIR, "wer_results.json")
WHISPER_MODEL = "small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = DEVICE == "cuda"

import yt_dlp

def download_youtube_audio(url):

    print("\nDownloading YouTube audio...")

    output_template = os.path.join(
        BASE_DIR, "downloaded_audio.%(ext)s"
    )

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = os.path.splitext(
            ydl.prepare_filename(info)
        )[0] + ".wav"

    print(f"Audio downloaded: {downloaded_file}")

    return downloaded_file
# ============================================================
# AUDIO STANDARDIZATION
# ============================================================

def convert_to_standard_wav(input_path):
    """
    Convert audio to:
    - WAV
    - Mono
    - 16 kHz
    - PCM 16-bit
    """

    output_path = os.path.join(BASE_DIR, "standard_audio.wav")

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)

    audio.export(output_path, format="wav")

    print(f"Standard audio created: {output_path}")

    return output_path


# ============================================================
# CASE 1: ORIGINAL AUDIO
# ============================================================

def process_original_audio(input_path):
    print("\n[CASE 1] Using original audio...")

    return input_path


# ============================================================
# CASE 2: FFMPEG ENHANCEMENT
# ============================================================

def enhance_with_ffmpeg(input_path):

    print("\n[CASE 2] Applying FFmpeg noise reduction...")

    filters = (
        "highpass=f=80,"
        "lowpass=f=7600,"
        "afftdn=nr=10:nf=-25,"
        "loudnorm=I=-20:TP=-2:LRA=11"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", filters,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        FFMPEG_OUTPUT
    ]

    subprocess.run(command, check=True)

    print("FFmpeg enhancement completed.")

    return FFMPEG_OUTPUT


# ============================================================
# CASE 3: DEEPFILTERNET ENHANCEMENT
# ============================================================


def enhance_with_deepfilternet(input_path):

    print("\n[CASE 3] Applying DeepFilterNet noise reduction...")

    import soundfile as sf
    import numpy as np
    import torch

    from df.enhance import enhance, init_df, save_audio

    # Initialize DeepFilterNet
    model, df_state, _ = init_df()

    # Read audio directly using soundfile
    audio, sample_rate = sf.read(
        input_path,
        dtype="float32"
    )

    # Convert stereo to mono if necessary
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Convert NumPy array to Torch tensor
    audio = torch.from_numpy(audio).unsqueeze(0)

    # Resample to DeepFilterNet's expected sample rate
    if sample_rate != df_state.sr():
        import torchaudio.functional as AF

        audio = AF.resample(
            audio,
            orig_freq=sample_rate,
            new_freq=df_state.sr()
        )

    # Apply enhancement
    enhanced_audio = enhance(
        model,
        df_state,
        audio
    )

    # Save enhanced audio
    save_audio(
        DEEPFILTER_OUTPUT,
        enhanced_audio,
        df_state.sr()
    )

    print("DeepFilterNet enhancement completed.")

    return DEEPFILTER_OUTPUT




def mix_original_and_deepfilter(original_path, enhanced_path):

    print("\n[CASE 4] Mixing original and DeepFilterNet audio...")

    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from math import gcd

    # Read original audio
    original, original_sr = sf.read(
        original_path,
        dtype="float32"
    )

    # Read DeepFilterNet audio
    enhanced, enhanced_sr = sf.read(
        enhanced_path,
        dtype="float32"
    )

    # Convert stereo to mono if needed
    if original.ndim == 2:
        original = np.mean(original, axis=1)

    if enhanced.ndim == 2:
        enhanced = np.mean(enhanced, axis=1)

    print(f"Original sample rate: {original_sr}")
    print(f"Enhanced sample rate: {enhanced_sr}")

    # Resample enhanced audio to original sample rate
    if enhanced_sr != original_sr:

        common = gcd(enhanced_sr, original_sr)

        enhanced = resample_poly(
            enhanced,
            original_sr // common,
            enhanced_sr // common
        )

        print("Enhanced audio resampled successfully.")

    # Align audio lengths
    min_length = min(
        len(original),
        len(enhanced)
    )

    original = original[:min_length]
    enhanced = enhanced[:min_length]

    # Mix 50% original + 50% enhanced
    mixed = (0.5 * original) + (0.5 * enhanced)

    # Prevent clipping
    mixed = np.clip(mixed, -1.0, 1.0)

    output_path = os.path.join(
        BASE_DIR,
        "mixed_audio.wav"
    )

    sf.write(
        output_path,
        mixed,
        original_sr
    )

    print(f"Mixed audio saved: {output_path}")
    print(f"Final sample rate: {original_sr}")
    print(f"Final duration: {min_length / original_sr:.2f} seconds")

    return output_path

# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(model, audio_path):

    start_time = time.perf_counter()

    result = model.transcribe(
        audio_path,
        task="transcribe",
        fp16=FP16,
        verbose=False
    )

    end_time = time.perf_counter()

    transcription_time = end_time - start_time

    return result["text"].strip(), transcription_time


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    # Remove punctuation but retain apostrophes.
    text = re.sub(r"[^\w\s']", " ", text)

    # Remove extra spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# WER CALCULATION
# ============================================================

def evaluate_transcription(reference, hypothesis):

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    error_rate = wer(reference, hypothesis) * 100

    details = process_words(reference, hypothesis)

    return {
        "wer": error_rate,
        "substitutions": details.substitutions,
        "deletions": details.deletions,
        "insertions": details.insertions,
        "reference_words": details.hits
        + details.substitutions
        + details.deletions
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:

        file.write("=" * 65 + "\n")
        file.write("             AUDIO ENHANCEMENT WER COMPARISON\n")
        file.write("=" * 65 + "\n\n")

        for result in results:

            file.write(f"CASE: {result['case']}\n")
            file.write("-" * 65 + "\n")

            file.write(f"WER: {result['wer']:.2f}%\n")
            file.write(f"Transcription time: {result['time']:.2f} seconds\n")
            file.write(f"Substitutions: {result['substitutions']}\n")
            file.write(f"Deletions: {result['deletions']}\n")
            file.write(f"Insertions: {result['insertions']}\n")

            file.write("\nTRANSCRIPTION:\n")
            file.write(result["transcript"] + "\n\n")

            file.write("=" * 65 + "\n\n")

        best = min(results, key=lambda x: x["wer"])

        file.write("\nBEST PERFORMING METHOD:\n")
        file.write(f"{best['case']} with WER {best['wer']:.2f}%\n")

    with open(JSON_RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"JSON results saved to: {JSON_RESULTS_FILE}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("          AUDIO ENHANCEMENT EVALUATION")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1: Validate YouTube URL and reference transcript
    # --------------------------------------------------------

    if not YOUTUBE_URL or "YOUR_VIDEO_ID" in YOUTUBE_URL:
        raise ValueError("Please provide a valid YouTube URL.")

    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"Reference transcript not found: {REFERENCE_FILE}"
        )

    with open(REFERENCE_FILE, "r", encoding="utf-8") as file:
        reference = file.read().strip()

    if not reference:
        raise ValueError("Reference transcript is empty.")

    # --------------------------------------------------------
    # STEP 2: Download audio from YouTube
    # --------------------------------------------------------

    print("\n[STEP 1] Downloading YouTube audio...")

    downloaded_audio = download_youtube_audio(YOUTUBE_URL)

    if not os.path.exists(downloaded_audio):
        raise FileNotFoundError(
            f"Downloaded audio not found: {downloaded_audio}"
        )

    print(f"Downloaded successfully: {downloaded_audio}")

    # --------------------------------------------------------
    # STEP 3: Load Whisper model
    # --------------------------------------------------------

    print(f"\nDevice: {DEVICE}")
    print(f"Loading Whisper model: {WHISPER_MODEL}")

    model = whisper.load_model(
        WHISPER_MODEL,
        device=DEVICE
    )

    print("Whisper model loaded successfully.")

    # --------------------------------------------------------
    # STEP 4: Standardize downloaded audio
    # --------------------------------------------------------

    print("\n[STEP 2] Standardizing audio...")

    standard_audio = convert_to_standard_wav(downloaded_audio)

    # --------------------------------------------------------
    # STEP 5: Prepare three audio variants
    # --------------------------------------------------------

    print("\n[STEP 3] Preparing audio variants...")

    # Prepare all four test cases
    # Generate enhanced audio first
    ffmpeg_audio = enhance_with_ffmpeg(standard_audio)

    deepfilter_audio = enhance_with_deepfilternet(standard_audio)

    # Mix original and DeepFilterNet output
    mixed_audio = mix_original_and_deepfilter(
    standard_audio,
    deepfilter_audio
    )
    audio_cases = [
    (
        "Original Audio",
        process_original_audio(standard_audio)
    ),
    (
        "FFmpeg Enhanced",
        ffmpeg_audio
    ),
    (
        "DeepFilterNet Enhanced",
        deepfilter_audio
    ),
    (
        "Original + DeepFilterNet",
        mixed_audio
    )
    ]


    results = []

    # --------------------------------------------------------
    # STEP 6: Transcribe and evaluate each variant
    # --------------------------------------------------------

    for case_name, audio_path in audio_cases:

        print("\n" + "=" * 65)
        print(f"TESTING: {case_name}")
        print("=" * 65)

        hypothesis, transcription_time = transcribe_audio(
            model,
            audio_path
        )

        metrics = evaluate_transcription(
            reference,
            hypothesis
        )

        result = {
            "case": case_name,
            "wer": metrics["wer"],
            "time": transcription_time,
            "substitutions": metrics["substitutions"],
            "deletions": metrics["deletions"],
            "insertions": metrics["insertions"],
            "reference_words": metrics["reference_words"],
            "transcript": hypothesis
        }

        results.append(result)

        print("\nTRANSCRIPTION:")
        print(hypothesis)

        print(f"\nWER: {metrics['wer']:.2f}%")
        print(f"Substitutions: {metrics['substitutions']}")
        print(f"Deletions: {metrics['deletions']}")
        print(f"Insertions: {metrics['insertions']}")
        print(f"Transcription time: {transcription_time:.2f} seconds")

    # --------------------------------------------------------
    # STEP 7: Final comparison
    # --------------------------------------------------------

    print("\n\n" + "=" * 75)
    print("                    FINAL WER COMPARISON")
    print("=" * 75)

    print(
        f"{'METHOD':<27}"
        f"{'WER':<13}"
        f"{'TIME (s)':<13}"
        f"{'S':<6}"
        f"{'D':<6}"
        f"{'I':<6}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['case']:<27}"
            f"{result['wer']:<13.2f}"
            f"{result['time']:<13.2f}"
            f"{result['substitutions']:<6}"
            f"{result['deletions']:<6}"
            f"{result['insertions']:<6}"
        )

    print("-" * 75)

    # --------------------------------------------------------
    # STEP 8: Identify best enhancement method
    # --------------------------------------------------------

    best = min(results, key=lambda x: x["wer"])

    print(f"\nBEST METHOD: {best['case']}")
    print(f"LOWEST WER: {best['wer']:.2f}%")

    # --------------------------------------------------------
    # STEP 9: Save results
    # --------------------------------------------------------

    save_results(results)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
