
import os
import pandas as pd

from dotenv import load_dotenv

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from core.rag_engine import get_llm
from core.vector_store import get_embeddings

load_dotenv()


def evaluate_rag(rag_chain, test_questions):
    """
    Evaluate an existing RAG chain using RAGAS.

    Each test case must contain:
        question: User's question
        reference: Expected correct answer
    """

    print("\n========== RAGAS EVALUATION ==========")

    # 1. Generate responses using your actual RAG pipeline
    evaluation_data = []

    for i, test_case in enumerate(test_questions, start=1):

        question = test_case["question"]
        reference = test_case["reference"]

        print(f"\nEvaluating question {i}/{len(test_questions)}")
        print(f"Question: {question}")

        # Invoke your existing RAG chain
        result = rag_chain.invoke(question)

        answer = result["answer"]
        contexts = result["contexts"]

        evaluation_data.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })

        print(f"Answer: {answer}")
        print(f"Retrieved chunks: {len(contexts)}")

    # 2. Create RAGAS dataset
    dataset = EvaluationDataset.from_list(evaluation_data)

    # 3. Configure evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(get_llm())

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        get_embeddings()
    )

    # 4. Select evaluation metrics
    metrics = [
        #Faithfulness(),
        ResponseRelevancy(),
        #LLMContextPrecisionWithoutReference(),
        #LLMContextRecall(),
        
    ]

    # 5. Run evaluation
    print("\nRunning RAGAS metrics...")

    results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=True
)

    # 6. Display results
    print("\n========== RAGAS RESULTS ==========")
    print(results)

    # 7. Save individual evaluation scores
    results_df = results.to_pandas()

    os.makedirs("evaluation_results", exist_ok=True)

    output_path = "evaluation_results/rag_evaluation.csv"

    results_df.to_csv(output_path, index=False)

    print(f"\nDetailed results saved to: {output_path}")

    return results
