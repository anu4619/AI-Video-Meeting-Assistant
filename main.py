from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

def run_pipeline(sourse:str, language:str="english")->dict:
    print("starting AI video Assistant")
    chunks=process_input(sourse)

    transcript = transcribe_all(chunks,language=language)

    print(f"raw transcription (first 300 characters) {transcript[:300]}")

    title=generate_title(transcript)

    summary=summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions= extract_key_decisions(transcript)
    questions= extract_questions(transcript)

    rag_chain=build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    # STEP 1: Get meeting input
    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = input(
        "Language (english/hinglish): "
    ).strip() or "english"

    # STEP 2: Run the complete meeting pipeline
    result = run_pipeline(source, language)

    # STEP 3: Extract RAG chain
    rag_chain = result["rag_chain"]

    # STEP 4: Test RAG retrieval
    test_result = rag_chain.invoke(
        "What were the main decisions discussed?"
    )

    print("\n========== RAG EVALUATION TEST ==========")

    print("\nGENERATED ANSWER:")
    print(test_result["answer"])

    print("\nRETRIEVED CONTEXTS:")

    for i, context in enumerate(
        test_result["contexts"], start=1
    ):
        print(f"\nChunk {i}:")
        print(context)

    print("\nDOCUMENT METADATA:")

    if test_result["documents"]:
        print(test_result["documents"][0].metadata)
    else:
        print("No documents retrieved.")

    print("\n=========================================")

    # STEP 5: Display meeting outputs
    print("\n" + "=" * 60)

    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")

    print("=" * 60)

    # STEP 6: Chat with meeting using RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant: {answer}\n")
