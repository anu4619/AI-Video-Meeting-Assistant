
import os
import torch
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None

def load_model():
    global _model

    if _model is None:
        print(f"Loading Whisper {WHISPER_MODEL} on {DEVICE}...", flush=True)

        _model = whisper.load_model(
            WHISPER_MODEL,
            device=DEVICE
        )

        print("Whisper model loaded successfully!", flush=True)

    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        chunk_path,
        task=task,
        fp16=(DEVICE == "cuda"),
        verbose=True
    )

    return result["text"]


def transcribe_all(chunks: list, translate: bool = False):
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}", flush=True)

        text = transcribe_chunk(chunk, translate=translate)
        full_transcript += text + " "

    print("Transcription completed!", flush=True)

    return full_transcript
