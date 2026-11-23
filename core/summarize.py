
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


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


def summarize(transcript: str) -> str:
    llm = get_llm()
    parser = StrOutputParser()

    # Step 1: Summarize individual chunks (Map)
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting summarizer.

            Summarize this portion of the meeting transcript concisely.

            Rules:
            - Capture the main topics, discussions, outcomes and important details.
            - Do not invent information or assume anything that is not stated.
            - Do not include anonymous speaker labels such as SPEAKER_00,
              SPEAKER_01 or SPEAKER_02.
            - Do not identify a person using a speaker ID.
            - Mention a person's real name only when that name is
              explicitly available in the transcript.
            - Do not attribute statements or decisions to people
              when their identity is unknown.
            - Preserve important numbers, dates, names and decisions.
            - Avoid repeating the same information.
            """
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | parser

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    # Step 2: Combine chunk summaries (Reduce)
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert meeting analyst.

            Combine the partial summaries into one accurate,
            professional meeting summary using bullet points.

            Rules:
            - Merge related discussion points.
            - Remove repeated information.
            - Preserve important decisions, dates, numbers and outcomes.
            - Do not introduce information that is absent from the
              partial summaries.
            - Do not mention anonymous speaker IDs such as SPEAKER_00.
            - Do not invent or guess people's names.
            - Do not attribute statements to unidentified participants.
            - Keep the summary concise and easy to understand.
            """
        ),
        ("human", "{text}")
    ])

    combined_chain = combined_prompt | llm | parser

    combined = "\n\n".join(chunk_summaries)

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Generate a short, professional title for this meeting.
            Maximum 8 words.
            Return only the title.
            Do not invent the meeting topic.
            """
        ),
        ("human", "{text}")
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke({"text": transcript[:2000]})
