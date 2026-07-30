import argparse
import json
import sys
import time
from pathlib import Path
import os

# Suppress the deprecation warnings so the logs stay clean
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from loguru import logger
import requests
from ragas import evaluate
from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig

from openai import OpenAI
from ragas.llms import llm_factory

# Use LangChain's native embeddings component directly to satisfy the legacy metric loop
from langchain_ollama import OllamaEmbeddings

from ragas.metrics import (
    Faithfulness,  # measures whether LLM's response is faithful to the retrieved contexts
    AnswerRelevancy,  # measures whether LLM's response is relevant to the user's question
    ContextRecall,  # measures whether the retrieved contexts contain the answer to the user's question
    LLMContextPrecisionWithReference,  # measures whether the retrieved contexts are relevant to the user's question and the LLM's response
)

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "llm_context_precision_with_reference",
]

DEFAULT_CORRECTNESS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
]


def get_active_ollama_models(model_info_url: str) -> dict:
    """Fetch the currently active chat/embedding ollama models from the app's own endpoint."""
    resp = requests.get(model_info_url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_model_list_url(model_info_url: str) -> str:
    return model_info_url.rsplit("/", maxsplit=1)[0] + "/list"


def get_app_model_names(model_info_url: str) -> list[str]:
    """Return exact model tags reported by the app's model management API."""
    resp = requests.get(get_model_list_url(model_info_url), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return sorted(model["name"] for model in data.get("models", []))


def get_local_ollama_model_names(ollama_base_url: str) -> list[str]:
    """Return exact model tags available in the local Ollama server."""
    tags_url = f"{ollama_base_url.rstrip('/')}/api/tags"
    resp = requests.get(tags_url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return sorted(model["name"] for model in data.get("models", []))


def validate_ollama_model_available(model_name: str, ollama_base_url: str) -> None:
    available_models = get_local_ollama_model_names(ollama_base_url)
    if model_name in available_models:
        return

    family = model_name.split(":", maxsplit=1)[0]
    similar_models = [name for name in available_models if name.startswith(family)]
    suggestions = similar_models or available_models
    suggestion_text = ", ".join(suggestions[:12]) if suggestions else "none"
    raise ValueError(
        f"Model '{model_name}' is not available in local Ollama. "
        f"Use one of the exact tags shown by `ollama list`. "
        f"Available/similar models: {suggestion_text}"
    )


def validate_judge_model_available(
    model_name: str,
    model_info_url: str,
    ollama_base_url: str,
) -> None:
    try:
        available_models = get_app_model_names(model_info_url)
    except requests.RequestException:
        validate_ollama_model_available(model_name, ollama_base_url)
        return

    if model_name in available_models:
        return

    family = model_name.split(":", maxsplit=1)[0]
    similar_models = [name for name in available_models if name.startswith(family)]
    suggestions = similar_models or available_models
    suggestion_text = ", ".join(suggestions[:12]) if suggestions else "none"
    raise ValueError(
        f"Model '{model_name}' is not reported by the app model API. "
        f"Use one of the exact tags from {get_model_list_url(model_info_url)}. "
        f"Available/similar models: {suggestion_text}"
    )


def build_judge(args):
    models = get_active_ollama_models(args.model_info_url)
    active_chat_model = models["chat_model"]
    judge_model = args.judge_model or active_chat_model
    embedding_model = models["embedding_model"]

    validate_judge_model_available(
        judge_model,
        args.model_info_url,
        args.ollama_base_url,
    )

    logger.info(f"Active chat model: {active_chat_model}")
    logger.info(f"Judge LLM: {judge_model}")
    logger.info(f"Judge Embeddings: {embedding_model}")

    openai_client = OpenAI(
        api_key="ollama",
        base_url=f"{args.ollama_base_url}/v1",
        timeout=args.judge_timeout,
    )

    if not args.skip_judge_preflight:
        logger.info("Checking judge model with a tiny preflight request...")
        start = time.perf_counter()
        openai_client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: OK",
                }
            ],
            max_tokens=16,
            temperature=0,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"  Judge preflight completed in {elapsed:.2f}s")

    llm = llm_factory(
        model=judge_model,
        client=openai_client,
    )

    # Force max_tokens configuration adjustments on the raw model dictionary
    # to eliminate the IncompleteOutputException
    llm.model_args["max_tokens"] = 2048
    # defines the randomness of the output; 0 means deterministic output
    llm.model_args["temperature"] = 0

    # Passing raw LangChain objects matches the legacy namespace metric signature requirements perfectly
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=args.ollama_base_url,
    )

    return llm, embeddings


def load_test_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def query_rag_api(api_url: str, question: str, timeout: int = 120) -> dict:
    """Call the live RAG endpoint and return {response, contexts}."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "include_contexts": True,
    }
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def limit_contexts(
    contexts: list[str],
    max_contexts: int | None,
    max_context_chars: int | None,
) -> list[str]:
    if max_contexts is not None:
        contexts = contexts[:max_contexts]

    if max_context_chars is None:
        return contexts

    return [context[:max_context_chars] for context in contexts]


def build_eval_dataset(
    test_cases: list[dict],
    args,
) -> tuple[EvaluationDataset, list[dict]]:
    samples = []
    sample_metadata = []
    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        ground_truth = case.get("ground_truth", "")

        logger.info(f"[{i}/{len(test_cases)}] Querying: {question[:80]}...")
        try:
            start = time.perf_counter()
            api_result = query_rag_api(args.api_url, question)
            elapsed = time.perf_counter() - start
            logger.info(f"  API response received in {elapsed:.2f}s")
        except requests.RequestException as e:
            logger.info(f"  !! API call failed for this question: {e}", file=sys.stderr)
            continue

        answer = api_result.get("response", "")
        retrieved_contexts = api_result.get("contexts") or []
        retrieved_contexts = limit_contexts(
            retrieved_contexts,
            args.max_eval_contexts,
            args.max_eval_context_chars,
        )
        expected_contexts = case.get("contexts") or []

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            )
        )
        sample_metadata.append(
            {
                "query_id": i,
                "api_latency_seconds": elapsed,
                "retrieved_context_count": len(retrieved_contexts),
                "expected_context_count": len(expected_contexts),
                "max_eval_contexts": args.max_eval_contexts,
                "max_eval_context_chars": args.max_eval_context_chars,
            }
        )

        # if i == 3:  # Evaluating 5 test items for verification
        #     break

        time.sleep(0.2)

    return EvaluationDataset(samples=samples), sample_metadata


def add_derived_correctness(df, metric_cols: list[str], threshold: float):
    missing_cols = [col for col in metric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Cannot derive correctness because these metrics are missing: "
            + ", ".join(missing_cols)
        )

    df = df.copy()
    df["correctness_score"] = df[metric_cols].mean(axis=1, skipna=False)
    df["is_correct"] = df[metric_cols].ge(threshold).all(axis=1)
    df["correctness_threshold"] = threshold
    df["correctness_metrics"] = ",".join(metric_cols)
    df["outcome"] = df["is_correct"].map({True: "Correct", False: "Needs review"})
    return df


def main():
    dataset_path = Path(__file__).parent / "datasets" / "goldens.json"
    results_path = Path(__file__).parent / "results_v2" / "ragas_results_llama_3_1_8b.csv"

    parser = argparse.ArgumentParser(
        description="Run RAGAs evaluation against the live RAG API"
    )

    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Ollama model used as the RAGAS judge. "
            "Defaults to the app's active chat model."
        ),
    )

    parser.add_argument(
        "--dataset", default=dataset_path, help="Path to test_cases.json"
    )
    parser.add_argument(
        "--output",
        default=results_path,
        help="Path where the per-sample CSV results should be written",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001/api/v1/rag/local",
        help="Full URL of the RAG chat endpoint",
    )

    parser.add_argument(
        "--model-info-url",
        default="http://localhost:8001/api/v1/model/active-models",
        help="Endpoint that reports the currently active models",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama server",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each local judge model call",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        help="Maximum concurrent RAGAS evaluation workers",
    )
    parser.add_argument(
        "--skip-judge-preflight",
        action="store_true",
        help="Skip the small test request to the judge model before RAGAS starts",
    )
    parser.add_argument(
        "--max-eval-contexts",
        type=int,
        default=3,
        help="Maximum retrieved contexts to pass into RAGAS per query",
    )
    parser.add_argument(
        "--max-eval-context-chars",
        type=int,
        default=2500,
        help="Maximum characters per retrieved context passed into RAGAS",
    )
    parser.add_argument(
        "--raise-eval-exceptions",
        action="store_true",
        help="Raise RAGAS exceptions instead of recording failed metric calls as NaN",
    )
    parser.add_argument(
        "--correctness-threshold",
        type=float,
        default=0.75,
        help="Minimum score required on each selected metric to mark a query correct",
    )
    parser.add_argument(
        "--correctness-metrics",
        nargs="+",
        default=DEFAULT_CORRECTNESS_METRICS,
        choices=METRIC_COLS,
        help="Metrics used to derive the is_correct column",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRIC_COLS,
        choices=METRIC_COLS,
        help="RAGAS metrics to run",
    )
    args = parser.parse_args()

    if not 0 <= args.correctness_threshold <= 1:
        parser.error("--correctness-threshold must be between 0 and 1")
    missing_correctness_metrics = set(args.correctness_metrics) - set(args.metrics)
    if missing_correctness_metrics:
        parser.error(
            "--correctness-metrics must be included in --metrics. Missing: "
            + ", ".join(sorted(missing_correctness_metrics))
        )

    logger.info(f"Loading dataset from {args.dataset}...")
    test_cases = load_test_cases(args.dataset)
    logger.info(f"Loaded {len(test_cases)} test cases.\n")

    logger.info("Querying live API for each question...")
    eval_dataset, sample_metadata = build_eval_dataset(test_cases, args)

    if len(eval_dataset) == 0:
        logger.info(
            "No samples were successfully collected. Aborting.", file=sys.stderr
        )
        sys.exit(1)

    logger.info(
        f"\nCollected {len(eval_dataset)} valid samples. Running RAGAs evaluation...\n"
    )

    # Build the judge LLM and embeddings for RAGAs evaluation
    judge_llm, judge_embeddings = build_judge(args)

    metric_builders = {
        "faithfulness": lambda: Faithfulness(llm=judge_llm),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=judge_llm,
            embeddings=judge_embeddings,
        ),
        "context_recall": lambda: ContextRecall(llm=judge_llm),
        "llm_context_precision_with_reference": lambda: (
            LLMContextPrecisionWithReference(llm=judge_llm)
        ),
    }
    metrics = [metric_builders[metric_name]() for metric_name in args.metrics]

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=args.judge_timeout,
            max_retries=1,
            max_workers=args.ragas_max_workers,
        ),
        raise_exceptions=args.raise_eval_exceptions,
    )

    logger.info("\n=== Aggregate RAGAs scores ===")
    logger.info(result)

    df = result.to_pandas()
    metadata_df = pd.DataFrame(sample_metadata)
    df = pd.concat([metadata_df, df.reset_index(drop=True)], axis=1)

    df = add_derived_correctness(
        df,
        metric_cols=args.correctness_metrics,
        threshold=args.correctness_threshold,
    )

    models = get_active_ollama_models(args.model_info_url)
    df["response_model"] = models["chat_model"]
    df["judge_model"] = args.judge_model or models["chat_model"]
    df["embedding_model"] = models["embedding_model"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nPer-sample results saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()
