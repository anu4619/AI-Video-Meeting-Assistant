
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

_pipeline = None


def load_diarization_model():

    global _pipeline

    if _pipeline is None:

        print("\nLoading pyannote Community-1 model...")

        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=HF_TOKEN
        )

        if _pipeline is None:
            raise RuntimeError("Could not load pyannote model.")

        print("Diarization model loaded successfully.")

    return _pipeline


def diarize_audio(audio_path: str, num_speakers: int) -> list:

    print("\nLoading complete audio for diarization...")

    audio = AudioSegment.from_file(audio_path)

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
        .set_sample_width(2)
    )

    samples = np.array(
        audio.get_array_of_samples(),
        dtype=np.float32
    )

    waveform = torch.from_numpy(samples).unsqueeze(0) / 32768.0

    audio_input = {
        "waveform": waveform,
        "sample_rate": 16000
    }

    print(f"Audio duration: {len(audio) / 1000:.2f} seconds")
    print(f"Expected participants: {num_speakers}")

    pipeline = load_diarization_model()

    print("\nStarting speaker diarization...")

    with ProgressHook() as hook:

        output = pipeline(
            audio_input,
            num_speakers=num_speakers,
            hook=hook
        )

    diarization = []

    # Exclusive diarization avoids overlapping speaker assignments.
    annotation = output.exclusive_speaker_diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):

        diarization.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    print("\nDiarization completed.")

    detected_speakers = sorted({
        item["speaker"] for item in diarization
    })

    print("\nDetected speaker IDs:")

    for speaker in detected_speakers:
        print(speaker)

    print(f"\nTotal detected speakers: {len(detected_speakers)}")

    return diarization


def assign_speakers(transcript_segments: list, diarization: list) -> str:

    """
    Match Whisper transcript segments with pyannote speaker turns
    using maximum temporal overlap.

    Segments without sufficient overlap are marked UNKNOWN.
    """

    final_transcript = []

    for segment in transcript_segments:

        segment_start = segment["start"]
        segment_end = segment["end"]

        segment_duration = max(
            segment_end - segment_start,
            0
        )

        best_speaker = "UNKNOWN"
        best_overlap = 0

        for turn in diarization:

            overlap = max(
                0,
                min(segment_end, turn["end"])
                - max(segment_start, turn["start"])
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_speaker = turn["speaker"]

        # Avoid assigning a speaker based on a tiny overlap.
        if (
            segment_duration <= 0
            or best_overlap < 0.1
            or best_overlap / segment_duration < 0.2
        ):
            best_speaker = "UNKNOWN"

        start_time = (
            f"{int(segment_start // 60):02d}:"
            f"{int(segment_start % 60):02d}"
        )

        end_time = (
            f"{int(segment_end // 60):02d}:"
            f"{int(segment_end % 60):02d}"
        )

        final_transcript.append(
            f"[{start_time}-{end_time}] "
            f"[{best_speaker}] "
            f"{segment['text'].strip()}"
        )

    return "\n".join(final_transcript)
