
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
            """
            You are an expert meeting analyst.

            Combine the extracted information into one accurate,
            organized list.

            Rules:
            - Remove duplicates.
            - Preserve important details.
            - Do not invent missing information.
            - Never treat anonymous speaker IDs as real names.
            - Never guess a person's identity from a speaker ID.
            - Follow the specific extraction rules given in the
              extracted information.
            """
        ),
        ("human", "{text}")
    ])

    combine_chain = combine_prompt | llm | parser

    combined = "\n\n".join(extracted_results)

    return combine_chain.invoke({"text": combined})


def extract_action_items(transcript: str) -> str:

    return extract_information(
        transcript,
        """
        You are an expert meeting analyst.

        Extract action items from this meeting transcript.

        STRICT OWNER REQUIREMENT:

        Include an action item ONLY if its responsible owner is
        explicitly identifiable by a real person's name.

        Examples:

        1. "Riya, please prepare the presentation."
           -> Include the task with Riya as the owner.

        2. "[SPEAKER_01] I will prepare the presentation."
           -> Exclude this task because the real person's name
              is not known.

        3. "[SPEAKER_02] Riya will send the report."
           -> Include the task with Riya as the owner, because
              the transcript explicitly names the responsible person.

        4. "Someone should prepare the report."
           -> Exclude this task because the owner is unknown.

        Additional rules:
        - Never use SPEAKER_00, SPEAKER_01, etc. as an owner.
        - Never guess a person's name from a speaker ID.
        - Do not assume the speaker is responsible for a task
          unless the transcript explicitly establishes their
          real identity and responsibility.
        - Do not include tasks with unknown or ambiguous owners.
        - Do not invent deadlines.
        - If a deadline is not mentioned, write "Not specified".

        For every qualifying action item, provide:
        - Task description
        - Owner (real person's name)
        - Deadline

        Format as a numbered list.

        If no action items have an identifiable real-name owner,
        return exactly:
        No action items found.
        """
    )


def extract_key_decisions(transcript: str) -> str:

    return extract_information(
        transcript,
        """
        You are an expert meeting analyst.

        Extract all finalized key decisions from this portion
        of the meeting transcript.

        Rules:
        - Include only decisions that were actually agreed upon
          or finalized.
        - Do not include suggestions, possibilities or proposals
          that were not accepted.
        - Preserve relevant dates, numbers and context.
        - Do not mention anonymous speaker IDs.
        - Do not guess or invent people's names.
        - Do not attribute decisions to unidentified participants.

        Format as a numbered list.

        If no finalized decisions are found, return exactly:
        No key decisions found.
        """
    )


def extract_questions(transcript: str) -> str:

    return extract_information(
        transcript,
        """
        You are an expert meeting analyst.

        Extract unresolved questions and topics requiring
        follow-up from this portion of the meeting.

        Rules:
        - Include only questions that remain unanswered.
        - Exclude questions that were answered later in the
          available transcript.
        - Exclude topics that were already resolved.
        - Include pending clarifications and decisions that
          still require follow-up.
        - Do not invent questions.
        - Do not mention anonymous speaker IDs.

        Format as a numbered list.

        If no unresolved questions are found, return exactly:
        No open questions found.
        """
    )
