
from dotenv import load_dotenv

from utils.audio_processor import process_input

from core.transcriber import (
    transcribe_all,
    transcribe_all_with_timestamps
)

from core.diarizer import (
    diarize_audio,
    assign_speakers
)

from core.summarize import summarize, generate_title

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)

from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(
    source: str,
    language: str = "english",
    recording_type: str = "youtube",
    num_speakers: int = None
) -> dict:

    print("\nStarting AI Video / Meeting Assistant")

    # Get chunks and complete enhanced audio path
    chunks, enhanced_audio_path = process_input(
        source,
        return_audio_path=True
    )

    # MEETING PIPELINE
    if recording_type == "meeting":

        # Current Sarvam function returns plain text,
        # so timestamp alignment is not available through it.
        if language.lower() == "hinglish":
            raise ValueError(
                "Meeting diarization currently requires Whisper "
                "timestamped transcription. Please select English "
                "for this test. Hinglish timestamp support can be "
                "added separately."
            )

        diarization = diarize_audio(
            enhanced_audio_path,
            num_speakers
        )

        transcript_segments = transcribe_all_with_timestamps(
            chunks,
            chunk_duration_seconds=600
        )

        transcript = assign_speakers(
            transcript_segments,
            diarization
        )

    # YOUTUBE PIPELINE
    else:

        transcript = transcribe_all(
            chunks,
            language=language
        )

    print(
        f"\nRaw transcription (first 300 characters):\n"
        f"{transcript[:300]}"
    )

    # EXISTING SUMMARIZATION PIPELINE

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    print("\n========== AI VIDEO / MEETING ASSISTANT ==========")

    # 1. Select recording type
    while True:

        recording_type = input(
            "\nSelect input type:\n"
            "1. YouTube video\n"
            "2. Meeting recording\n"
            "Enter choice: "
        ).strip()

        if recording_type in ["1", "2"]:
            break

        print("Invalid choice.")

    recording_type = (
        "youtube" if recording_type == "1"
        else "meeting"
    )

    # 2. Get source
    if recording_type == "youtube":

        source = input("\nEnter YouTube URL: ").strip()
        num_speakers = None

    else:

        source = input(
            "\nEnter meeting audio/video file path: "
        ).strip()

        while True:
            try:
                num_speakers = int(
                    input("Number of meeting participants: ")
                )

                if num_speakers >= 2:
                    break

                print("Enter at least 2 participants.")

            except ValueError:
                print("Enter a valid number.")

    # 3. Language
    language = input(
        "\nLanguage (english/hinglish): "
    ).strip().lower() or "english"

    # 4. Process recording
    result = run_pipeline(
        source=source,
        language=language,
        recording_type=recording_type,
        num_speakers=num_speakers
    )

    # 5. Display generated outputs
    print("\n" + "=" * 60)
    print("RECORDING ANALYSIS")
    print("=" * 60)

    print(f"\nTitle:\n{result['title']}")

    print(f"\nTranscript:\n{result['transcript']}")

    print(f"\nSummary:\n{result['summary']}")

    print(f"\nAction Items:\n{result['action_items']}")

    print(f"\nKey Decisions:\n{result['key_decisions']}")

    print(f"\nOpen Questions:\n{result['open_questions']}")

    print("\n" + "=" * 60)

    # 6. Interactive RAG session
    rag_chain = result["rag_chain"]

    print("\n========== CHAT WITH YOUR RECORDING ==========")
    print("Ask any question about the recording.")
    print("Type 'exit' to end the conversation.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        if not question:
            continue

        try:

            answer = ask_question(rag_chain, question)

            print(f"\nAssistant: {answer}\n")

        except Exception as e:

            print(f"\nError answering question: {e}\n")
