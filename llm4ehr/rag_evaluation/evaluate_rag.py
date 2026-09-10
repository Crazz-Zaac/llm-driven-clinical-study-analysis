import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
import os

# Suppress the deprecation warnings so the logs stay clean
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from loguru import logger
import requests

CSV_METADATA_COLUMNS = [
    "article_id",
    "article_title",
    "field_id",
    "category",
    "status",
    "section",
]

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

MISSING_PAPER_RESPONSE = "MISSING_RESPONSE_FOR_THIS_PAPER"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.6-luna"
APP_ENV_PATH = Path(__file__).resolve().parents[1] / "app" / ".env"


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
    from openai import OpenAI
    from ragas.llms import llm_factory
    from dotenv import load_dotenv

    # Use LangChain's native embeddings component directly to satisfy the legacy metric loop
    from langchain_ollama import OllamaEmbeddings

    load_dotenv(APP_ENV_PATH, override=False)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set. Add it to {APP_ENV_PATH} or export it "
            "before running the evaluator."
        )

    models = get_active_ollama_models(args.model_info_url)
    active_chat_model = models["chat_model"]
    judge_model = args.judge_model
    embedding_model = models["embedding_model"]

    logger.info(f"Active chat model: {active_chat_model}")
    logger.info(f"Judge LLM: {judge_model}")
    logger.info(f"Judge Embeddings: {embedding_model}")

    openai_client = OpenAI(
        api_key=openai_api_key,
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
            max_completion_tokens=16,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"  Judge preflight completed in {elapsed:.2f}s")

    llm = llm_factory(
        model=judge_model,
        client=openai_client,
    )

    # Newer OpenAI models reject the legacy Chat Completions `max_tokens`
    # field. RAGAS defaults include it, so replace it explicitly.
    llm.model_args.pop("max_tokens", None)
    llm.model_args.pop("temperature", None)
    llm.model_args.pop("top_p", None)
    llm.model_args["max_completion_tokens"] = 8192

    # Passing raw LangChain objects matches the legacy namespace metric signature requirements perfectly
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=args.ollama_base_url,
    )

    return llm, embeddings


def _clean_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def csv_to_test_cases(path: str | Path) -> list[dict]:
    """Convert the ML4EHR CSV into all-papers evaluation cases.

    The source CSV is one row per paper/question answer. For all-papers RAG,
    the system receives each question once and should answer across papers, so
    each JSON case groups all paper-specific reference answers for a category.
    """
    rows = []
    carry_forward = {
        "article_id": "",
        "article_title": "",
        "field_id": "",
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [
            fieldname
            for fieldname in (reader.fieldnames or [])
            if fieldname and not fieldname.startswith("Unnamed")
        ]

        for source_index, row in enumerate(reader, start=2):
            cleaned_row = {
                fieldname: _clean_value(row.get(fieldname))
                for fieldname in fieldnames
            }
            if not any(cleaned_row.values()):
                continue

            for col in carry_forward:
                if cleaned_row.get(col):
                    carry_forward[col] = cleaned_row[col]
                else:
                    cleaned_row[col] = carry_forward[col]

            cleaned_row["source_row"] = source_index
            rows.append(cleaned_row)

    required_cols = {"question", "reference_answer"}
    available_cols = set(rows[0].keys()) if rows else set()
    missing_cols = sorted(required_cols - available_cols)
    if missing_cols:
        raise ValueError(
            f"CSV dataset is missing required column(s): {', '.join(missing_cols)}"
        )

    test_cases = []
    grouped_rows: dict[str, list[dict]] = {}
    for row in rows:
        if not _clean_value(row.get("question")) or not _clean_value(
            row.get("reference_answer")
        ):
            continue

        category = _clean_value(row.get("category"))
        question_key = category or _clean_value(row.get("question"))
        grouped_rows.setdefault(question_key, []).append(row)

    for query_id, (question_key, rows) in enumerate(grouped_rows.items(), start=1):
        first_row = rows[0]
        question = _clean_value(first_row.get("question"))
        category = _clean_value(first_row.get("category"))
        field_id = _clean_value(first_row.get("field_id"))
        references = []

        for row in rows:
            reference = {
                "article_id": _clean_value(row.get("article_id")),
                "article_title": _clean_value(row.get("article_title")),
                "section": _clean_value(row.get("section")),
                "status": _clean_value(row.get("status")),
                "answer": _clean_value(row.get("reference_answer")),
                "source_row": int(row.get("source_row")),
            }
            references.append(reference)

        ground_truth = format_references_for_ragas(references)
        test_cases.append(
            {
                "question_id": category or question_key,
                "question": question,
                "ground_truth": ground_truth,
                "contexts": [ground_truth],
                "references": references,
                "metadata": {
                    "grouped_query_id": query_id,
                    "category": category,
                    "field_id": field_id,
                    "reference_count": len(references),
                    "reference_article_ids": [
                        reference["article_id"] for reference in references
                    ],
                },
            }
        )

    return test_cases


def format_references_for_ragas(references: list[dict]) -> str:
    """Render structured paper references as one RAGAS reference answer."""
    blocks = []
    for reference in references:
        blocks.append(
            "\n".join(
                [
                    f"Paper ID: {reference.get('article_id', '')}",
                    f"Title: {reference.get('article_title', '')}",
                    f"Section: {reference.get('section', '')}",
                    f"Status: {reference.get('status', '')}",
                    f"Answer: {reference.get('answer', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def get_case_reference(case: dict) -> str:
    if case.get("ground_truth"):
        return case["ground_truth"]
    if case.get("references"):
        return format_references_for_ragas(case["references"])
    return ""


def normalize_match_key(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", _clean_value(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\([|*_`[\]()])", r"\1", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())
    return " ".join(value.split())


def markdown_link_text(value: str) -> str:
    match = re.search(r"\[([^\]]+)\]\([^)]+\)", value)
    if match:
        return match.group(1).strip()
    return _clean_value(value)


def extract_article_id(value: str) -> str:
    match = re.search(r"\bs\d{4,5}-\d{3}-\d{5}-[a-z0-9]+\b", value, re.IGNORECASE)
    return match.group(0) if match else ""


def split_markdown_table_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]

    cells = []
    current = []
    escaped = False
    for char in row:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells
    )


def parse_response_table_by_title(response: str) -> tuple[dict[str, dict], list[dict]]:
    """Return response table rows keyed by normalized paper title."""
    rows_by_title = {}
    parsed_rows = []
    for line in _clean_value(response).splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = split_markdown_table_row(stripped)
        if len(cells) < 3 or is_markdown_separator(cells):
            continue
        if [cell.lower() for cell in cells[:3]] == ["paper", "answer", "evidence"]:
            continue

        paper_title = markdown_link_text(cells[0])
        row = {
            "row_index": len(parsed_rows),
            "paper": cells[0],
            "article_id": extract_article_id(cells[0]),
            "paper_title": paper_title,
            "answer": cells[1],
            "evidence": cells[2],
        }
        parsed_rows.append(row)
        title_key = normalize_match_key(paper_title)
        if title_key:
            rows_by_title[title_key] = row

    return rows_by_title, parsed_rows


def find_response_row_for_reference(
    reference: dict,
    response_rows_by_title: dict[str, dict],
    response_rows: list[dict],
) -> dict | None:
    reference_title_key = normalize_match_key(reference.get("article_title", ""))
    response_row = response_rows_by_title.get(reference_title_key)
    if response_row:
        return response_row

    for row in response_rows:
        response_title_key = normalize_match_key(row.get("paper_title", ""))
        if (
            reference_title_key
            and response_title_key
            and (
                reference_title_key in response_title_key
                or response_title_key in reference_title_key
            )
        ):
            return row

    reference_article_id = normalize_match_key(reference.get("article_id", ""))
    for row in response_rows:
        response_article_id = normalize_match_key(row.get("article_id", ""))
        if reference_article_id and reference_article_id == response_article_id:
            return row

    return None


def context_matches_reference(context: str, reference: dict) -> bool:
    context_key = normalize_match_key(context)
    article_id = normalize_match_key(reference.get("article_id", ""))
    title = normalize_match_key(reference.get("article_title", ""))
    return bool(
        (article_id and article_id in context_key)
        or (title and title in context_key)
    )


def contexts_for_reference(contexts: list[str], reference: dict) -> list[str]:
    matching_contexts = [
        context for context in contexts if context_matches_reference(context, reference)
    ]
    return matching_contexts or contexts


def write_test_cases_json(test_cases: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"test_cases": test_cases}, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_test_cases(path: str | Path) -> list[dict]:
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".csv":
        return csv_to_test_cases(dataset_path)

    with dataset_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data["test_cases"]


def query_rag_api(
    api_url: str,
    question: str,
    retrieval_scope: str,
    timeout: int | None = 120,
) -> dict:
    """Call the live RAG endpoint and return {response, contexts}."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "include_contexts": True,
        "retrieval_scope": retrieval_scope,
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
) -> tuple[object, list[dict]]:
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    sample_metadata = []
    selected_cases = test_cases[args.offset :]
    if args.limit is not None:
        selected_cases = selected_cases[: args.limit]

    for i, case in enumerate(selected_cases, start=1):
        question = case["question"]
        ground_truth = get_case_reference(case)
        references = case.get("references") or []
        metadata = case.get("metadata", {})
        query_id = args.offset + i

        logger.info(f"[{i}/{len(selected_cases)}] Querying: {question[:80]}...")
        try:
            start = time.perf_counter()
            api_result = query_rag_api(
                args.api_url,
                question,
                retrieval_scope=args.retrieval_scope,
                timeout=args.api_timeout,
            )
            elapsed = time.perf_counter() - start
            logger.info(f"  API response received in {elapsed:.2f}s")
        except requests.RequestException as e:
            logger.info(f"  !! API call failed for this question: {e}", file=sys.stderr)
            continue

        answer = api_result.get("response", "")
        retrieved_contexts = api_result.get("contexts") or []
        expected_contexts = case.get("contexts") or []

        if args.retrieval_scope == "all_papers" and references:
            response_rows_by_title, response_rows = parse_response_table_by_title(answer)
            matched_response_indexes = set()

            for paper_index, reference in enumerate(references, start=1):
                response_row = find_response_row_for_reference(
                    reference,
                    response_rows_by_title,
                    response_rows,
                )
                if response_row:
                    matched_response_indexes.add(response_row["row_index"])

                paper_response = (
                    response_row["answer"] if response_row else MISSING_PAPER_RESPONSE
                )
                paper_contexts = contexts_for_reference(retrieved_contexts, reference)
                paper_contexts = limit_contexts(
                    paper_contexts,
                    args.max_eval_contexts,
                    args.max_eval_context_chars,
                )
                paper_reference = reference.get("answer", "")
                paper_title = reference.get("article_title", "")
                paper_question = (
                    f"{question}\n\nEvaluate only this paper: {paper_title}"
                ).strip()

                samples.append(
                    SingleTurnSample(
                        user_input=paper_question,
                        response=paper_response,
                        retrieved_contexts=paper_contexts,
                        reference=paper_reference,
                    )
                )
                sample_metadata.append(
                    {
                        "query_id": query_id,
                        "eval_unit": "paper",
                        "paper_index": paper_index,
                        "article_id": reference.get("article_id", ""),
                        "article_title": paper_title,
                        "reference_section": reference.get("section", ""),
                        "reference_status": reference.get("status", ""),
                        "source_row": reference.get("source_row", ""),
                        "api_latency_seconds": elapsed,
                        "retrieved_context_count": len(paper_contexts),
                        "expected_context_count": 1,
                        "max_eval_contexts": args.max_eval_contexts,
                        "max_eval_context_chars": args.max_eval_context_chars,
                        "retrieval_scope": args.retrieval_scope,
                        "response_paper_count": len(response_rows),
                        "response_paper_titles": [
                            row["paper_title"] for row in response_rows
                        ],
                        "matched_response_paper": response_row is not None,
                        **metadata,
                    }
                )

            extra_response_titles = [
                row["paper_title"]
                for row in response_rows
                if row["row_index"] not in matched_response_indexes
            ]
            missing_reference_titles = [
                reference.get("article_title", "")
                for reference in references
                if not find_response_row_for_reference(
                    reference,
                    response_rows_by_title,
                    response_rows,
                )
            ]
            for metadata_row in sample_metadata[-len(references) :]:
                metadata_row["missing_paper_count"] = len(missing_reference_titles)
                metadata_row["missing_paper_titles"] = missing_reference_titles
                metadata_row["extra_paper_count"] = len(extra_response_titles)
                metadata_row["extra_paper_titles"] = extra_response_titles

            time.sleep(0.2)
            continue

        retrieved_contexts = limit_contexts(
            retrieved_contexts,
            args.max_eval_contexts,
            args.max_eval_context_chars,
        )

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
                "query_id": query_id,
                "eval_unit": "query",
                "api_latency_seconds": elapsed,
                "retrieved_context_count": len(retrieved_contexts),
                "expected_context_count": len(expected_contexts),
                "max_eval_contexts": args.max_eval_contexts,
                "max_eval_context_chars": args.max_eval_context_chars,
                "retrieval_scope": args.retrieval_scope,
                **metadata,
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


def add_query_level_correctness(df):
    if "eval_unit" not in df.columns or "query_id" not in df.columns:
        return df

    paper_df = df[df["eval_unit"] == "paper"]
    if paper_df.empty:
        return df

    df = df.copy()
    query_summaries = {}
    for query_id, group in paper_df.groupby("query_id", dropna=False):
        missing_paper_count = int(group["missing_paper_count"].max())
        extra_paper_count = int(group["extra_paper_count"].max())
        paper_total_count = int(len(group))
        paper_pass_count = int(group["is_correct"].sum())
        paper_accuracy = paper_pass_count / paper_total_count if paper_total_count else 0
        min_paper_correctness = group["correctness_score"].min(skipna=False)
        all_papers_correct = (
            missing_paper_count == 0
            and extra_paper_count == 0
            and paper_total_count > 0
            and bool(group["is_correct"].all())
        )
        query_summaries[query_id] = {
            "paper_total_count": paper_total_count,
            "paper_pass_count": paper_pass_count,
            "paper_accuracy": paper_accuracy,
            "min_paper_correctness": min_paper_correctness,
            "all_papers_correct": all_papers_correct,
            "query_outcome": "Correct" if all_papers_correct else "Needs review",
        }

    for column in [
        "paper_total_count",
        "paper_pass_count",
        "paper_accuracy",
        "min_paper_correctness",
        "all_papers_correct",
        "query_outcome",
    ]:
        df[column] = df["query_id"].map(
            {
                query_id: summary[column]
                for query_id, summary in query_summaries.items()
            }
        )

    return df


def main():
    dataset_path = Path(__file__).parent / "datasets" / "ML4EHR.csv"
    results_path = Path(__file__).parent / "results_v2" / "ragas_results_llama_3_1_8b.csv"

    parser = argparse.ArgumentParser(
        description="Run RAGAs evaluation against the live RAG API"
    )

    parser.add_argument(
        "--judge-model",
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help=(
            "OpenAI model used as the RAGAS judge. "
            f"Defaults to {DEFAULT_OPENAI_JUDGE_MODEL}."
        ),
    )

    parser.add_argument(
        "--dataset",
        default=dataset_path,
        help="Path to a test_cases JSON file or ML4EHR-style CSV file",
    )
    parser.add_argument(
        "--write-json-dataset",
        default=None,
        help="Optional path where the converted test_cases JSON should be written",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Convert --dataset to JSON and exit. Requires --write-json-dataset.",
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
        "--api-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each RAG API request. Use 0 to wait indefinitely.",
    )
    parser.add_argument(
        "--retrieval-scope",
        choices=["all_papers", "relevant_papers", "specific_paper"],
        default="all_papers",
        help="Retrieval scope to send to the RAG chat endpoint",
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
        default=None,
        help=(
            "Maximum retrieved contexts to pass into RAGAS per sample. "
            "Defaults to no count limit. Use 0 to disable the count limit."
        ),
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
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset rows to evaluate",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of dataset rows to skip before evaluation",
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

    if args.convert_only and not args.write_json_dataset:
        parser.error("--convert-only requires --write-json-dataset")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than 0")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.api_timeout < 0:
        parser.error("--api-timeout cannot be negative")
    if args.api_timeout == 0:
        args.api_timeout = None
    if args.max_eval_contexts is not None and args.max_eval_contexts < 0:
        parser.error("--max-eval-contexts cannot be negative")
    if args.max_eval_contexts == 0:
        args.max_eval_contexts = None
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

    if args.write_json_dataset:
        write_test_cases_json(test_cases, args.write_json_dataset)
        logger.info(f"Converted dataset saved to {Path(args.write_json_dataset).absolute()}")
    if args.convert_only:
        return

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

    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        Faithfulness,  # measures whether LLM's response is faithful to the retrieved contexts
        AnswerRelevancy,  # measures whether LLM's response is relevant to the user's question
        ContextRecall,  # measures whether the retrieved contexts contain the answer to the user's question
        LLMContextPrecisionWithReference,  # measures whether the retrieved contexts are relevant to the user's question and the LLM's response
    )

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
    import pandas as pd

    metadata_df = pd.DataFrame(sample_metadata)
    df = pd.concat([metadata_df, df.reset_index(drop=True)], axis=1)

    df = add_derived_correctness(
        df,
        metric_cols=args.correctness_metrics,
        threshold=args.correctness_threshold,
    )
    df = add_query_level_correctness(df)

    models = get_active_ollama_models(args.model_info_url)
    df["response_model"] = models["chat_model"]
    df["judge_model"] = args.judge_model
    df["embedding_model"] = models["embedding_model"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nPer-sample results saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()
