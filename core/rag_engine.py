
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)

load_dotenv()


def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    )


def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


# Generate alternative queries
def generate_query_variations(question: str, llm) -> list:

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert search query generator.

            Generate 3 alternative search queries for the user's
            question to retrieve relevant information from a
            meeting transcript.

            Use different wording, synonyms and related concepts.
            Preserve the original intent.

            Return only the alternative queries, one per line.
            Do not answer the question."""
        ),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"question": question})

    variations = []

    for line in result.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove numbering and bullet points
        line = line.lstrip("0123456789.-) ").strip()

        if line:
            variations.append(line)

    # Keep original question and remove duplicate variations
    queries = [question] + variations

    unique_queries = list(dict.fromkeys(
        q.strip() for q in queries if q.strip()
    ))

    return unique_queries[:4]


# Multi-query retrieval with Reciprocal Rank Fusion
def multi_query_retrieve(
    question: str,
    retriever,
    llm,
    final_k: int = 6
) -> list:

    queries = generate_query_variations(question, llm)

    print("\nGenerated search queries:")
    for query in queries:
        print("-", query)

    documents = {}
    scores = {}

    # Search using every generated query
    for query in queries:

        results = retriever.invoke(query)

        for rank, doc in enumerate(results, start=1):

            # Use page content to identify duplicate chunks
            key = doc.page_content.strip()

            documents[key] = doc

            # Reciprocal Rank Fusion score
            scores[key] = scores.get(key, 0) + (
                1 / (60 + rank)
            )

    # Rank documents using their combined RRF scores
    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    retrieved_docs = [
        documents[key]
        for key in ranked_keys[:final_k]
    ]

    print(f"\nRetrieved {len(retrieved_docs)} unique chunks.")

    return retrieved_docs


# Create the RAG chain

def create_rag_chain(vector_store):

    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

            Answer the user's question based ONLY on the
            meeting transcript context provided below.

            If the answer is not found in the context, say:
            "I could not find this information in the meeting transcript."

            Always be concise and precise.
            If quoting someone, mention it clearly.

            Context from meeting transcript:
            {context}"""
        ),
        ("human", "{question}")
    ])

    def retrieve_with_trace(question):

        docs = multi_query_retrieve(
            question=question,
            retriever=retriever,
            llm=llm,
            final_k=6
        )

        return {
            "question": question,
            "documents": docs,
            "contexts": [
                doc.page_content for doc in docs
            ],
        }

    def add_formatted_context(data):

        return {
            **data,
            "context": format_docs(data["documents"])
        }

    rag_chain = (
        RunnableLambda(retrieve_with_trace)
        | RunnableLambda(add_formatted_context)
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:

    print(f"\nQuestion: {question}")

    result = rag_chain.invoke(question)

    answer = result["answer"]

    print(f"\nAnswer: {answer}")

    return answer

    


# Build vector store for a new meeting
def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    return create_rag_chain(vector_store)


# Load an already existing vector store
def load_rag_chain():

    vector_store = load_vector_store()

    return create_rag_chain(vector_store)



