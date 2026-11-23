
import os
import torch
import numpy as np

from dotenv import load_dotenv
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing in .env file")

# Supports OGG, MP3, WAV, etc.
AUDIO_PATH = r"technova_meet.ogg"

if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")


# STEP 1: Load audio using pydub
print("Loading audio...")

audio = AudioSegment.from_file(AUDIO_PATH)

# Convert to mono, 16 kHz, 16-bit
audio = (
    audio
    .set_channels(1)
    .set_frame_rate(16000)
    .set_sample_width(2)
)

# Convert audio samples into a PyTorch tensor
samples = np.array(
    audio.get_array_of_samples(),
    dtype=np.float32
)

waveform = torch.from_numpy(samples).unsqueeze(0) / 32768.0

print(f"Audio duration: {len(audio) / 1000:.2f} seconds")


# STEP 2: Load pyannote model
print("\nLoading pyannote diarization model...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

if pipeline is None:
    raise RuntimeError("Could not load diarization model.")


# STEP 3: Run diarization
print("\nStarting speaker diarization...")

audio_input = {
    "waveform": waveform,
    "sample_rate": 16000
}

with ProgressHook() as hook:
    output = pipeline(
        audio_input,
        num_speakers=3,
        hook=hook
    )


# STEP 4: Display results
print("\nSPEAKER DIARIZATION RESULTS")
print("-" * 50)

for turn, speaker in output.speaker_diarization:
    print(
        f"{speaker}: "
        f"{turn.start:.2f}s - {turn.end:.2f}s"
    )


# STEP 5: Count speakers
speakers = {
    speaker
    for _, speaker in output.speaker_diarization
}

print("\nDetected speakers:")

for speaker in sorted(speakers):
    print(speaker)

print(f"\nTotal speakers detected: {len(speakers)}")
