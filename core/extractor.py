
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def extract_information(transcript: str, system_prompt: str) -> str:

    llm = get_llm()
    parser = StrOutputParser()

    # Step 1: Extract information from individual chunks
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    extraction_chain = extraction_prompt | llm | parser

    chunks = split_transcript(transcript)

    extracted_results = [
        extraction_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Step 2: Combine results from all chunks
    combine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting analyst. Combine the extracted "
            "information into one accurate, organized list. "
            "Remove duplicates while preserving important details. "
            "Do not invent missing information."
        ),
        ("human", "{text}")
    ])

    combine_chain = combine_prompt | llm | parser

    combined = "\n\n".join(extracted_results)

    return combine_chain.invoke({"text": combined})


def extract_action_items(transcript: str) -> str:

    return extract_information(
        transcript,
        "You are an expert meeting analyst. Extract all action items "
        "from this transcript. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, otherwise write 'Not specified')\n\n"
        "Format as a numbered list. If none are found, say "
        "'No action items found.'"
    )


def extract_key_decisions(transcript: str) -> str:

    return extract_information(
        transcript,
        "You are an expert meeting analyst. Extract all key decisions "
        "made in this portion of the transcript. Include relevant context "
        "where useful. Do not include suggestions that were not finalized. "
        "Format as a numbered list. If none are found, say "
        "'No key decisions found.'"
    )


def extract_questions(transcript: str) -> str:

    return extract_information(
        transcript,
        "You are an expert meeting analyst. Extract all unresolved questions "
        "and topics requiring follow-up from this portion of the transcript. "
        "Do not include questions that were already answered. "
        "Format as a numbered list. If none are found, say "
        "'No open questions found.'"
    )
