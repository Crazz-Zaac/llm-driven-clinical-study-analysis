"""Evaluate the live LLM4EHR RAG API with RAGAS.

For every benchmark question the script queries the running backend, splits the
answer into one evaluation sample per paper (in the ``all_papers`` scope), labels
whether the answer says the information is reported, and scores the samples with
RAGAS using an LLM judge (OpenAI, Mistral or a local Ollama model).

Example (run from ``llm4ehr/rag_evaluation`` with the Docker stack running)::

    python evaluate_rag.py \\
        --dataset datasets/ML4EHR.csv \\
        --response-model llama3.2:3b \\
        --judge-model gpt-5.6-luna \\
        --api-timeout 0 --judge-timeout 1200 \\
        --output results/llama_3_2b.csv

Outputs (``<output>`` is the ``--output`` path):

* ``<output>``                        per-sample RAGAS scores and labels
* ``<output>__status_mask.csv``       reported / not-reported labels for all cells
* ``<output>__status_mask__status_summary.csv`` and ``__status_errors.csv``
* ``<output>__by_category.csv``, ``__by_field_id.csv``, ``__by_reference_status.csv``
* ``<output>__needs_review.csv``      samples below the correctness threshold
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import requests
from loguru import logger

# Keep the evaluation log readable; RAGAS and LangChain emit many deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANSWER_CORRECTNESS_METRIC = "answer_correctness"
METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "llm_context_precision_with_reference",
    ANSWER_CORRECTNESS_METRIC,
]

DEFAULT_EXPECTED_QUESTION_COUNT = 38
DEFAULT_CORRECTNESS_THRESHOLD = 0.75
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5.6-luna"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
JUDGE_MAX_TOKENS = 8192

APP_ENV_PATH = Path(__file__).resolve().parents[1] / "app" / ".env"
JUDGE_PROVIDERS = ["auto", "openai", "mistral", "ollama"]
MISTRAL_MODEL_PREFIXES = (
    "codestral-",
    "devstral-",
    "magistral-",
    "ministral-",
    "mistral-",
    "pixtral-",
    "voxtral-",
)

# Placeholder answer for a paper that is missing from the generated answer table.
MISSING_PAPER_RESPONSE = "MISSING_RESPONSE_FOR_THIS_PAPER"

REPORTED_STATUS = "reported"
NOT_REPORTED_STATUS = "no_reported"
REFERENCE_STATUS_FILTERS = ["all", REPORTED_STATUS, NOT_REPORTED_STATUS]

# Column order of the main results CSV (shared by all runs used in the thesis).
ALL_PAPER_RESULT_COLUMNS = [
    "query_id",
    "eval_unit",
    "paper_index",
    "article_id",
    "article_title",
    "reference_section",
    "reference_status",
    "source_row",
    "api_latency_seconds",
    "retrieved_context_count",
    "expected_context_count",
    "max_eval_contexts",
    "max_eval_context_chars",
    "generator_seed",
    "judge_seed",
    "retrieval_scope",
    "response_paper_count",
    "response_paper_titles",
    "matched_response_paper",
    "grouped_query_id",
    "category",
    "field_id",
    "reference_count",
    "reference_article_ids",
    "missing_paper_count",
    "missing_paper_titles",
    "extra_paper_count",
    "extra_paper_titles",
    "user_input",
    "retrieved_contexts",
    "response",
    "response_status",
    "status_match",
    "status_outcome",
    "reference",
    *METRIC_COLS,
    "correctness_score",
    "is_correct",
    "correctness_threshold",
    "correctness_metrics",
    "outcome",
    "paper_total_count",
    "paper_pass_count",
    "paper_accuracy",
    "min_paper_correctness",
    "all_papers_correct",
    "query_outcome",
    "response_model",
    "judge_model",
    "embedding_model",
]


# ---------------------------------------------------------------------------
# Backend and Ollama model helpers
# ---------------------------------------------------------------------------


def get_active_ollama_models(model_info_url: str) -> dict:
    """Return the backend's active chat and embedding models."""
    resp = requests.get(model_info_url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_model_list_url(model_info_url: str) -> str:
    return model_info_url.rsplit("/", maxsplit=1)[0] + "/list"


def get_app_model_names(model_info_url: str) -> list[str]:
    """Return the exact model tags reported by the backend's model API."""
    resp = requests.get(get_model_list_url(model_info_url), timeout=10)
    resp.raise_for_status()
    return sorted(model["name"] for model in resp.json().get("models", []))


def get_local_ollama_model_names(ollama_base_url: str) -> list[str]:
    """Return the exact model tags available on the Ollama server."""
    resp = requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=10)
    resp.raise_for_status()
    return sorted(model["name"] for model in resp.json().get("models", []))


def _model_suggestions(model_name: str, available_models: list[str]) -> str:
    family = model_name.split(":", maxsplit=1)[0]
    similar_models = [name for name in available_models if name.startswith(family)]
    suggestions = similar_models or available_models
    return ", ".join(suggestions[:12]) if suggestions else "none"


def validate_ollama_model_available(model_name: str, ollama_base_url: str) -> None:
    available_models = get_local_ollama_model_names(ollama_base_url)
    if model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' is not available in local Ollama. "
            "Use one of the exact tags shown by `ollama list`. "
            f"Available/similar models: {_model_suggestions(model_name, available_models)}"
        )


def validate_response_model_available(
    model_name: str,
    model_info_url: str,
    ollama_base_url: str,
) -> None:
    try:
        available_models = get_app_model_names(model_info_url)
    except requests.RequestException:
        validate_ollama_model_available(model_name, ollama_base_url)
        return

    if model_name not in available_models:
        raise ValueError(
            f"Model '{model_name}' is not reported by the app model API. "
            f"Use one of the exact tags from {get_model_list_url(model_info_url)}. "
            f"Available/similar models: {_model_suggestions(model_name, available_models)}"
        )


def activate_chat_model(model_name: str, model_info_url: str) -> None:
    """Set the backend's active chat model before the evaluation starts."""
    activate_url = get_model_list_url(model_info_url).rsplit("/", maxsplit=1)[0] + "/activate"
    resp = requests.post(
        activate_url,
        json={"model_name": model_name, "model_type": "chat"},
        timeout=30,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Judge (RAGAS LLM and embeddings)
# ---------------------------------------------------------------------------


def resolve_judge_provider(judge_model: str, judge_provider: str) -> str:
    """Choose the judge backend from common model-name conventions."""
    if judge_provider != "auto":
        return judge_provider
    if ":" in judge_model:
        return "ollama"
    if judge_model.lower().startswith(MISTRAL_MODEL_PREFIXES):
        return "mistral"
    return "openai"


def _run_preflight(provider_label: str, request_fn) -> None:
    """Send one tiny request so a misconfigured judge fails before the long run."""
    logger.info(f"Checking {provider_label} judge model with a tiny preflight request...")
    start = time.perf_counter()
    request_fn()
    logger.info(f"  Judge preflight completed in {time.perf_counter() - start:.2f}s")


def _require_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        f"{' or '.join(names)} is not set. Add it to {APP_ENV_PATH} or export it "
        "before running the evaluator."
    )


def build_judge(args):
    """Build the RAGAS judge LLM and the embeddings used by the RAGAS metrics."""
    from dotenv import load_dotenv
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from openai import OpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import llm_factory
    from ragas.llms.base import LangchainLLMWrapper

    load_dotenv(APP_ENV_PATH, override=False)

    # rescore_saved_reference.py passes args without a judge seed.
    judge_seed = getattr(args, "judge_seed", None)
    judge_model = args.judge_model
    judge_provider = resolve_judge_provider(judge_model, args.judge_provider)
    models = get_active_ollama_models(args.model_info_url)
    embedding_model = models["embedding_model"]

    logger.info(f"Active chat model: {models['chat_model']}")
    logger.info(f"Judge LLM: {judge_model} ({judge_provider})")
    logger.info(f"Judge Embeddings: {embedding_model}")

    if judge_provider == "ollama":
        validate_ollama_model_available(judge_model, args.ollama_base_url)
        chat_ollama = ChatOllama(
            model=judge_model,
            base_url=args.ollama_base_url,
            temperature=0,
            num_predict=JUDGE_MAX_TOKENS,
            seed=judge_seed,
        )
        if not args.skip_judge_preflight:
            _run_preflight("Ollama", lambda: chat_ollama.invoke("Reply with exactly: OK"))
        llm = LangchainLLMWrapper(chat_ollama)

    elif judge_provider == "mistral":
        mistral_client = OpenAI(
            api_key=_require_env("MISTRALAI_API_KEY", "MISTRAL_API_KEY"),
            base_url=(
                os.environ.get("MISTRALAI_BASE_URL")
                or os.environ.get("MISTRAL_BASE_URL")
                or args.mistral_base_url
            ),
            timeout=args.judge_timeout,
        )
        if not args.skip_judge_preflight:
            _run_preflight(
                "Mistral",
                lambda: mistral_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=16,
                ),
            )
        llm = llm_factory(model=judge_model, client=mistral_client)
        # Mistral's OpenAI-compatible endpoint expects `max_tokens`.
        llm.model_args.pop("max_completion_tokens", None)
        llm.model_args.pop("temperature", None)
        llm.model_args.pop("top_p", None)
        llm.model_args["max_tokens"] = JUDGE_MAX_TOKENS
        if judge_seed is not None:
            # Mistral's seed parameter; reproducibility is best-effort on the provider side.
            llm.model_args["extra_body"] = {"random_seed": judge_seed}

    else:
        openai_client = OpenAI(
            api_key=_require_env("OPENAI_API_KEY"),
            timeout=args.judge_timeout,
        )
        if not args.skip_judge_preflight:
            _run_preflight(
                "OpenAI",
                lambda: openai_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    max_completion_tokens=16,
                ),
            )
        llm = llm_factory(model=judge_model, client=openai_client)
        # Newer OpenAI models reject the legacy `max_tokens` field that RAGAS sets by default.
        llm.model_args.pop("max_tokens", None)
        llm.model_args.pop("temperature", None)
        llm.model_args.pop("top_p", None)
        llm.model_args["max_completion_tokens"] = JUDGE_MAX_TOKENS
        if judge_seed is not None:
            # OpenAI seed parameter; reproducibility is best-effort on the provider side.
            llm.model_args["seed"] = judge_seed

    # RAGAS metrics need embed_text in addition to LangChain's embed_query API.
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embedding_model, base_url=args.ollama_base_url)
    )
    return llm, embeddings


# ---------------------------------------------------------------------------
# Reported / not-reported labels
# ---------------------------------------------------------------------------


def _clean_value(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_reference_status(value) -> str:
    """Map the spellings of the status label to ``reported`` / ``no_reported``."""
    text = re.sub(r"[\s_-]+", " ", _clean_value(value).lower()).strip()
    if text in {"reported", "report"}:
        return REPORTED_STATUS
    if text in {"not reported", "non reported", "unreported", "no reported"}:
        return NOT_REPORTED_STATUS
    return text


_ABSENCE_STARTS = (
    "not reported",
    "not specified",
    "not explicitly reported",
    "not provided",
    "not mentioned",
    "not found in the retrieved documents",
    "not found in retrieved documents",
    "not found in this paper",
    "no relevant information",
)
_ABSENCE_SENTENCE_PATTERNS = (
    r"^(the\s+)?(paper|study|article|authors?)\s+"
    r"(does|do|did)\s+not\s+"
    r"(report|state|specify|provide|mention|describe|evaluate|include)\b",
    r"^no,\s+(the\s+)?(paper|study|article|authors?)\s+"
    r"(does|do|did)\s+not\s+"
    r"(report|state|specify|provide|mention|describe|evaluate|include)\b",
    r"^(the\s+)?requested\s+information\s+"
    r"(is|was)\s+not\s+(reported|specified|provided|mentioned)\b",
)


def infer_response_status(response) -> str:
    """Rule-based label: does the generated answer say the information is reported?"""
    text = _clean_value(response)
    status_label = normalize_reference_status(text)
    if status_label in {REPORTED_STATUS, NOT_REPORTED_STATUS}:
        return status_label

    normalized = re.sub(r"\s+", " ", text.lower()).strip(" .;:")
    if not normalized or normalized == MISSING_PAPER_RESPONSE.lower():
        return NOT_REPORTED_STATUS
    if normalized.startswith(_ABSENCE_STARTS):
        return NOT_REPORTED_STATUS
    if any(re.search(pattern, normalized) for pattern in _ABSENCE_SENTENCE_PATTERNS):
        return NOT_REPORTED_STATUS
    return REPORTED_STATUS


def status_outcome(reference_status: str, response_status: str) -> str:
    """Name the confusion-matrix cell of a (reference, response) status pair."""
    reference_status = normalize_reference_status(reference_status)
    response_status = normalize_reference_status(response_status)
    known = {REPORTED_STATUS, NOT_REPORTED_STATUS}
    if reference_status not in known or response_status not in known:
        return "unknown_status"
    return f"{reference_status}_gt_marked_{response_status}"


def add_status_mask_columns(df):
    df = df.copy()
    df["reference_status"] = df["reference_status"].map(normalize_reference_status)
    df["response_status"] = df["response"].map(infer_response_status)
    df["status_match"] = df["reference_status"] == df["response_status"]
    df["status_outcome"] = [
        status_outcome(reference_status, response_status)
        for reference_status, response_status in zip(
            df["reference_status"], df["response_status"]
        )
    ]
    return df


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def format_references_for_ragas(references: list[dict]) -> str:
    """Render structured paper references as one RAGAS reference answer."""
    return "\n\n".join(
        "\n".join(
            [
                f"Paper ID: {reference.get('article_id', '')}",
                f"Title: {reference.get('article_title', '')}",
                f"Section: {reference.get('section', '')}",
                f"Status: {reference.get('status', '')}",
                f"Answer: {reference.get('answer', '')}",
            ]
        )
        for reference in references
    )


def _read_benchmark_rows(path: str | Path) -> list[dict]:
    """Read the benchmark CSV, filling blank article/field cells from the row above."""
    rows = []
    carry_forward = {"article_id": "", "article_title": "", "field_id": ""}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [
            name for name in (reader.fieldnames or []) if name and not name.startswith("Unnamed")
        ]
        for source_index, row in enumerate(reader, start=2):
            cleaned_row = {name: _clean_value(row.get(name)) for name in fieldnames}
            if not any(cleaned_row.values()):
                continue
            for col in carry_forward:
                if cleaned_row.get(col):
                    carry_forward[col] = cleaned_row[col]
                else:
                    cleaned_row[col] = carry_forward[col]
            cleaned_row["source_row"] = source_index
            rows.append(cleaned_row)

    missing_cols = sorted({"question", "reference_answer"} - set(rows[0] if rows else {}))
    if missing_cols:
        raise ValueError(f"CSV dataset is missing required column(s): {', '.join(missing_cols)}")
    return rows


def csv_to_test_cases(path: str | Path) -> list[dict]:
    """Convert the ML4EHR CSV into all-papers evaluation cases.

    The CSV has one row per paper and question. In the all-papers scope the RAG
    system receives each question once and answers for every paper, so each case
    groups the paper-specific reference answers of one question category.
    """
    grouped_rows: dict[str, list[dict]] = {}
    for row in _read_benchmark_rows(path):
        if not _clean_value(row.get("question")) or not _clean_value(row.get("reference_answer")):
            continue
        question_key = _clean_value(row.get("category")) or _clean_value(row.get("question"))
        grouped_rows.setdefault(question_key, []).append(row)

    test_cases = []
    for query_id, (question_key, rows) in enumerate(grouped_rows.items(), start=1):
        first_row = rows[0]
        category = _clean_value(first_row.get("category"))
        references = [
            {
                "article_id": _clean_value(row.get("article_id")),
                "article_title": _clean_value(row.get("article_title")),
                "section": _clean_value(row.get("section")),
                "status": normalize_reference_status(row.get("status")),
                "answer": _clean_value(row.get("reference_answer")),
                "source_row": int(row.get("source_row")),
            }
            for row in rows
        ]
        ground_truth = format_references_for_ragas(references)
        test_cases.append(
            {
                "question_id": category or question_key,
                "question": _clean_value(first_row.get("question")),
                "ground_truth": ground_truth,
                "contexts": [ground_truth],
                "references": references,
                "metadata": {
                    "grouped_query_id": query_id,
                    "category": category,
                    "field_id": _clean_value(first_row.get("field_id")),
                    "reference_count": len(references),
                    "reference_article_ids": [ref["article_id"] for ref in references],
                },
            }
        )
    return test_cases


def load_test_cases(path: str | Path) -> list[dict]:
    """Load test cases from the benchmark CSV or from a converted JSON file."""
    dataset_path = Path(path)
    if dataset_path.suffix.lower() == ".csv":
        return csv_to_test_cases(dataset_path)
    with dataset_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)["test_cases"]


def write_test_cases_json(test_cases: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"test_cases": test_cases}, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_case_reference(case: dict) -> str:
    if case.get("ground_truth"):
        return case["ground_truth"]
    if case.get("references"):
        return format_references_for_ragas(case["references"])
    return ""


# ---------------------------------------------------------------------------
# Paper selection (specific_paper scope)
# ---------------------------------------------------------------------------


def normalize_match_key(value: str) -> str:
    """Lower-case alphanumeric key used to match titles and IDs across formats."""
    value = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", _clean_value(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\\([|*_`[\]()])", r"\1", value)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value.lower())
    return " ".join(value.split())


def _titles_overlap(first_key: str, second_key: str) -> bool:
    return bool(first_key and second_key and (first_key in second_key or second_key in first_key))


def _paper_matches(reference: dict, paper_id: str | None, paper_title: str | None) -> bool:
    if paper_id and _clean_value(reference.get("article_id")) == paper_id:
        return True
    if paper_title:
        return _titles_overlap(
            normalize_match_key(paper_title),
            normalize_match_key(reference.get("article_title", "")),
        )
    return False


def get_ordered_papers(test_cases: list[dict]) -> list[dict]:
    """Return the unique papers in the order they appear in the dataset."""
    papers = []
    seen_article_ids = set()
    seen_titles = set()
    for case in test_cases:
        for reference in case.get("references") or []:
            article_id = _clean_value(reference.get("article_id"))
            article_title = _clean_value(reference.get("article_title"))
            dedupe_key = article_id or normalize_match_key(article_title)
            if not dedupe_key or dedupe_key in seen_article_ids or dedupe_key in seen_titles:
                continue
            if article_id:
                seen_article_ids.add(article_id)
            seen_titles.add(normalize_match_key(article_title))
            papers.append({"article_id": article_id, "article_title": article_title})
    return papers


def select_specific_paper_cases(
    test_cases: list[dict],
    paper_id: str | None = None,
    paper_title: str | None = None,
    paper_index: int | None = None,
) -> list[dict]:
    """Keep one paper's reference per question for single-paper evaluation."""
    if paper_index is not None:
        papers = get_ordered_papers(test_cases)
        if paper_index < 1 or paper_index > len(papers):
            raise ValueError(f"--paper-index must be between 1 and {len(papers)} for this dataset")
        paper_id = papers[paper_index - 1]["article_id"]
        paper_title = papers[paper_index - 1]["article_title"]

    if not paper_id and not paper_title:
        raise ValueError(
            "Single-paper evaluation requires --paper-id, --paper-title, or --paper-index."
        )

    selected_cases = []
    for case in test_cases:
        matching_references = [
            reference
            for reference in case.get("references") or []
            if _paper_matches(reference, paper_id, paper_title)
        ]
        if not matching_references:
            continue

        reference = matching_references[0]
        metadata = dict(case.get("metadata", {}))
        metadata.update(
            {
                "reference_count": 1,
                "reference_article_ids": [reference.get("article_id", "")],
                "selected_paper_id": reference.get("article_id", ""),
                "selected_paper_title": reference.get("article_title", ""),
            }
        )
        case_copy = dict(case)
        case_copy["references"] = [reference]
        case_copy["ground_truth"] = reference.get("answer", "")
        case_copy["contexts"] = [case_copy["ground_truth"]]
        case_copy["metadata"] = metadata
        selected_cases.append(case_copy)

    if not selected_cases:
        selector = paper_id or paper_title or f"paper index {paper_index}"
        raise ValueError(f"No CSV references matched selected paper: {selector}")
    return selected_cases


# ---------------------------------------------------------------------------
# Parsing the all-papers answer table
# ---------------------------------------------------------------------------


def markdown_link_text(value: str) -> str:
    match = re.search(r"\[([^\]]+)\]\([^)]+\)", value)
    return match.group(1).strip() if match else _clean_value(value)


def extract_article_id(value: str) -> str:
    match = re.search(r"\bs\d{4,5}-\d{3}-\d{5}-[a-z0-9]+\b", value, re.IGNORECASE)
    return match.group(0) if match else ""


def split_markdown_table_row(row: str) -> list[str]:
    """Split a Markdown table row on unescaped pipes."""
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
        elif char == "\\":
            current.append(char)
            escaped = True
        elif char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    cells.append("".join(current).strip())
    return cells


def is_markdown_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def parse_response_table_by_title(response: str) -> tuple[dict[str, dict], list[dict]]:
    """Parse the ``| Paper | Answer | Evidence |`` table into rows keyed by title."""
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
    """Find the answer-table row for a reference paper (exact title, partial title, then ID)."""
    reference_title_key = normalize_match_key(reference.get("article_title", ""))
    if reference_title_key in response_rows_by_title:
        return response_rows_by_title[reference_title_key]

    for row in response_rows:
        if _titles_overlap(reference_title_key, normalize_match_key(row.get("paper_title", ""))):
            return row

    reference_article_id = normalize_match_key(reference.get("article_id", ""))
    for row in response_rows:
        if reference_article_id and reference_article_id == normalize_match_key(
            row.get("article_id", "")
        ):
            return row
    return None


def context_matches_reference(context: str, reference: dict) -> bool:
    context_key = normalize_match_key(context)
    article_id = normalize_match_key(reference.get("article_id", ""))
    title = normalize_match_key(reference.get("article_title", ""))
    return bool((article_id and article_id in context_key) or (title and title in context_key))


def contexts_for_reference(contexts: list[str], reference: dict) -> list[str]:
    """Keep the retrieved contexts of one paper; fall back to all contexts."""
    matching_contexts = [c for c in contexts if context_matches_reference(c, reference)]
    return matching_contexts or contexts


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


# ---------------------------------------------------------------------------
# Question-count validation
# ---------------------------------------------------------------------------


def select_test_cases_for_run(test_cases: list[dict], offset: int, limit: int | None) -> list[dict]:
    selected_cases = test_cases[offset:]
    return selected_cases[:limit] if limit is not None else selected_cases


def duplicate_values(values: list[str]) -> list[str]:
    seen = set()
    duplicated = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    return sorted(duplicated)


def case_question_id(case: dict) -> str:
    metadata = case.get("metadata", {})
    return (
        _clean_value(case.get("question_id"))
        or _clean_value(metadata.get("category"))
        or _clean_value(case.get("question"))
    )


def validate_expected_question_count(
    test_cases: list[dict],
    expected_count: int | None,
    label: str,
) -> None:
    if expected_count is None:
        return

    if len(test_cases) != expected_count:
        raise ValueError(
            f"{label} contains {len(test_cases)} questions; expected "
            f"{expected_count}. Use --expected-question-count 0 only for "
            "intentional partial/smoke runs."
        )
    duplicate_ids = duplicate_values([case_question_id(case) for case in test_cases])
    if duplicate_ids:
        raise ValueError(
            f"{label} contains duplicate question ids/categories: " + ", ".join(duplicate_ids)
        )
    if duplicate_values([_clean_value(case.get("question")) for case in test_cases]):
        raise ValueError(f"{label} contains duplicate question text.")

    logger.info(f"{label}: validated {expected_count} questions.")


def validate_sample_metadata_question_count(
    sample_metadata: list[dict],
    expected_count: int | None,
    label: str,
) -> None:
    if expected_count is None:
        return

    query_ids = sorted(
        {int(row["query_id"]) for row in sample_metadata if row.get("query_id") is not None}
    )
    if len(query_ids) != expected_count:
        raise RuntimeError(
            f"{label} contains {len(query_ids)} unique questions; expected "
            f"{expected_count}. Present query_ids: {query_ids}"
        )
    if query_ids != list(range(query_ids[0], query_ids[0] + expected_count)):
        raise RuntimeError(f"{label} query_ids are not contiguous. Present query_ids: {query_ids}")

    logger.info(f"{label}: validated {expected_count} unique questions.")


# ---------------------------------------------------------------------------
# Collecting answers from the RAG API
# ---------------------------------------------------------------------------


def query_rag_api(
    api_url: str,
    question: str,
    retrieval_scope: str,
    article_id: str | None = None,
    timeout: int | None = 120,
    seed: int | None = None,
) -> dict:
    """Call the live RAG endpoint and return its JSON (``response``, ``contexts``, ...)."""
    payload = {
        "messages": [{"role": "user", "content": question}],
        "include_contexts": True,
        "retrieval_scope": retrieval_scope,
    }
    if article_id:
        payload["article_id"] = article_id
    if seed is not None:
        # A fixed generator seed gives identical answers for identical contexts.
        payload["seed"] = seed
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _run_settings(args) -> dict:
    """Run settings recorded with every sample."""
    return {
        "max_eval_contexts": args.max_eval_contexts,
        "max_eval_context_chars": args.max_eval_context_chars,
        "generator_seed": args.generator_seed,
        "judge_seed": args.judge_seed,
        "retrieval_scope": args.retrieval_scope,
    }


def _collect_all_papers_samples(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    references: list[dict],
    metadata: dict,
    query_id: int,
    elapsed: float,
    args,
) -> tuple[list, list[dict]]:
    """Split one all-papers answer table into one RAGAS sample per reference paper."""
    from ragas import SingleTurnSample

    response_rows_by_title, response_rows = parse_response_table_by_title(answer)
    matched_response_indexes = set()
    missing_reference_titles = []
    samples = []
    sample_metadata = []

    for paper_index, reference in enumerate(references, start=1):
        response_row = find_response_row_for_reference(
            reference, response_rows_by_title, response_rows
        )
        if response_row:
            matched_response_indexes.add(response_row["row_index"])
        else:
            missing_reference_titles.append(reference.get("article_title", ""))

        paper_contexts = limit_contexts(
            contexts_for_reference(retrieved_contexts, reference),
            args.max_eval_contexts,
            args.max_eval_context_chars,
        )
        paper_title = reference.get("article_title", "")
        samples.append(
            SingleTurnSample(
                user_input=f"{question}\n\nEvaluate only this paper: {paper_title}".strip(),
                response=response_row["answer"] if response_row else MISSING_PAPER_RESPONSE,
                retrieved_contexts=paper_contexts,
                reference=reference.get("answer", ""),
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
                "reference_status": normalize_reference_status(reference.get("status", "")),
                "source_row": reference.get("source_row", ""),
                "api_latency_seconds": elapsed,
                "retrieved_context_count": len(paper_contexts),
                "expected_context_count": 1,
                **_run_settings(args),
                "response_paper_count": len(response_rows),
                "response_paper_titles": [row["paper_title"] for row in response_rows],
                "matched_response_paper": response_row is not None,
                **metadata,
            }
        )

    extra_response_titles = [
        row["paper_title"]
        for row in response_rows
        if row["row_index"] not in matched_response_indexes
    ]
    for row in sample_metadata:
        row["missing_paper_count"] = len(missing_reference_titles)
        row["missing_paper_titles"] = missing_reference_titles
        row["extra_paper_count"] = len(extra_response_titles)
        row["extra_paper_titles"] = extra_response_titles
    return samples, sample_metadata


def _collect_single_sample(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    case: dict,
    query_id: int,
    elapsed: float,
    args,
) -> tuple[object, dict]:
    """Build one RAGAS sample for the relevant_papers or specific_paper scope."""
    from ragas import SingleTurnSample

    references = case.get("references") or []
    metadata = case.get("metadata", {})
    retrieved_contexts = limit_contexts(
        retrieved_contexts, args.max_eval_contexts, args.max_eval_context_chars
    )
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=get_case_reference(case),
    )

    selected_reference = references[0] if references else {}
    is_specific_paper = bool(metadata.get("selected_paper_id"))
    output_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"selected_paper_id", "selected_paper_title"}
    }
    sample_metadata = {
        "query_id": query_id,
        "eval_unit": "paper" if is_specific_paper else "query",
        "paper_index": 1 if is_specific_paper else None,
        "article_id": selected_reference.get("article_id", ""),
        "article_title": selected_reference.get("article_title", ""),
        "reference_section": selected_reference.get("section", ""),
        "reference_status": normalize_reference_status(selected_reference.get("status", "")),
        "source_row": selected_reference.get("source_row", ""),
        "api_latency_seconds": elapsed,
        "retrieved_context_count": len(retrieved_contexts),
        "expected_context_count": 1 if is_specific_paper else len(case.get("contexts") or []),
        **_run_settings(args),
        "response_paper_count": 1 if is_specific_paper else None,
        "response_paper_titles": (
            [selected_reference.get("article_title", "")] if is_specific_paper else None
        ),
        "matched_response_paper": True if is_specific_paper else None,
        "missing_paper_count": 0 if is_specific_paper else None,
        "missing_paper_titles": [] if is_specific_paper else None,
        "extra_paper_count": 0 if is_specific_paper else None,
        "extra_paper_titles": [] if is_specific_paper else None,
        **output_metadata,
    }
    return sample, sample_metadata


def build_eval_dataset(
    test_cases: list[dict],
    args,
    query_id_offset: int = 0,
) -> tuple[object, list[dict]]:
    """Query the RAG API for every test case and build the RAGAS dataset."""
    from ragas import EvaluationDataset

    samples = []
    sample_metadata = []

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        references = case.get("references") or []
        metadata = case.get("metadata", {})
        query_id = query_id_offset + i

        logger.info(f"[{i}/{len(test_cases)}] Querying: {question[:80]}...")
        try:
            start = time.perf_counter()
            api_result = query_rag_api(
                args.api_url,
                question,
                retrieval_scope=args.retrieval_scope,
                article_id=metadata.get("selected_paper_id"),
                timeout=args.api_timeout,
                seed=args.generator_seed,
            )
            elapsed = time.perf_counter() - start
            logger.info(f"  API response received in {elapsed:.2f}s")
        except requests.RequestException as e:
            logger.error(f"  API call failed for this question: {e}")
            continue

        answer = api_result.get("response", "")
        retrieved_contexts = api_result.get("contexts") or []

        if args.retrieval_scope == "all_papers" and references:
            case_samples, case_metadata = _collect_all_papers_samples(
                question, answer, retrieved_contexts, references, metadata, query_id, elapsed, args
            )
            samples.extend(case_samples)
            sample_metadata.extend(case_metadata)
        else:
            sample, row_metadata = _collect_single_sample(
                question, answer, retrieved_contexts, case, query_id, elapsed, args
            )
            samples.append(sample)
            sample_metadata.append(row_metadata)

        time.sleep(0.2)

    return EvaluationDataset(samples=samples), sample_metadata


def build_status_mask_dataframe(eval_dataset, sample_metadata: list[dict]):
    """Table of all collected cells with their reported / not-reported labels."""
    import pandas as pd

    records = []
    for sample, metadata in zip(eval_dataset.samples, sample_metadata):
        record = dict(metadata)
        record.update(
            {
                "user_input": getattr(sample, "user_input", ""),
                "retrieved_contexts": getattr(sample, "retrieved_contexts", []),
                "response": getattr(sample, "response", ""),
                "reference": getattr(sample, "reference", ""),
            }
        )
        records.append(record)

    if not records:
        return pd.DataFrame()
    return add_status_mask_columns(pd.DataFrame(records))


def refresh_filtered_query_metadata(sample_metadata: list[dict]) -> None:
    """Keep query-level counts scoped to the filtered paper rows."""
    grouped_indexes: dict[object, list[int]] = {}
    for index, metadata in enumerate(sample_metadata):
        grouped_indexes.setdefault(metadata.get("query_id"), []).append(index)

    for indexes in grouped_indexes.values():
        missing_titles = [
            sample_metadata[index].get("article_title", "")
            for index in indexes
            if sample_metadata[index].get("matched_response_paper") is False
        ]
        for index in indexes:
            sample_metadata[index]["missing_paper_count"] = len(missing_titles)
            sample_metadata[index]["missing_paper_titles"] = missing_titles
            sample_metadata[index]["extra_paper_count"] = 0
            sample_metadata[index]["extra_paper_titles"] = []


def filter_eval_dataset_by_reference_status(
    eval_dataset,
    sample_metadata: list[dict],
    reference_status: str,
) -> tuple[object, list[dict]]:
    """Keep only the samples whose reference status is sent to RAGAS."""
    if reference_status == "all":
        return eval_dataset, sample_metadata

    from ragas import EvaluationDataset

    normalized_status = normalize_reference_status(reference_status)
    selected_indexes = [
        index
        for index, metadata in enumerate(sample_metadata)
        if normalize_reference_status(metadata.get("reference_status", "")) == normalized_status
    ]
    samples = [eval_dataset.samples[index] for index in selected_indexes]
    metadata = [sample_metadata[index] for index in selected_indexes]
    refresh_filtered_query_metadata(metadata)
    return EvaluationDataset(samples=samples), metadata


# ---------------------------------------------------------------------------
# Scoring and correctness labels
# ---------------------------------------------------------------------------


def run_ragas(eval_dataset, args):
    """Score the dataset with the selected RAGAS metrics and return a DataFrame."""
    from ragas import evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextRecall,
        Faithfulness,
        LLMContextPrecisionWithReference,
    )
    from ragas.run_config import RunConfig

    judge_llm, judge_embeddings = build_judge(args)
    metric_builders = {
        # Is the answer supported by the retrieved contexts?
        "faithfulness": lambda: Faithfulness(llm=judge_llm),
        # Does the answer address the question?
        "answer_relevancy": lambda: AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        # Do the retrieved contexts contain the reference answer?
        "context_recall": lambda: ContextRecall(llm=judge_llm),
        # Are the relevant contexts ranked above the irrelevant ones?
        "llm_context_precision_with_reference": lambda: LLMContextPrecisionWithReference(
            llm=judge_llm
        ),
        # Does the answer agree with the reference answer?
        ANSWER_CORRECTNESS_METRIC: lambda: AnswerCorrectness(
            llm=judge_llm, embeddings=judge_embeddings
        ),
    }

    result = evaluate(
        dataset=eval_dataset,
        metrics=[metric_builders[name]() for name in args.metrics],
        run_config=RunConfig(
            timeout=args.judge_timeout,
            max_retries=1,
            max_workers=args.ragas_max_workers,
        ),
        raise_exceptions=args.raise_eval_exceptions,
    )
    logger.info("=== Aggregate RAGAS scores ===")
    logger.info(result)
    return result.to_pandas()


def add_answer_correctness_labels(df, threshold: float):
    """Automated label: a sample is correct when answer correctness >= threshold."""
    if ANSWER_CORRECTNESS_METRIC not in df.columns:
        raise ValueError(
            f"Cannot label correctness because {ANSWER_CORRECTNESS_METRIC} is "
            "missing. Include it in --metrics."
        )

    df = df.copy()
    df["correctness_score"] = df[ANSWER_CORRECTNESS_METRIC]
    df["is_correct"] = df[ANSWER_CORRECTNESS_METRIC].ge(threshold)
    df["correctness_threshold"] = threshold
    df["correctness_metrics"] = ANSWER_CORRECTNESS_METRIC
    df["outcome"] = df["is_correct"].map({True: "Correct", False: "Needs review"})
    return df


QUERY_LEVEL_COLUMNS = [
    "paper_total_count",
    "paper_pass_count",
    "paper_accuracy",
    "min_paper_correctness",
    "all_papers_correct",
    "query_outcome",
]


def add_query_level_correctness(df):
    """Summarise the per-paper labels of each question (all-papers scope)."""
    if "eval_unit" not in df.columns or "query_id" not in df.columns:
        return df

    paper_df = df[df["eval_unit"] == "paper"]
    if paper_df.empty:
        return df

    df = df.copy()
    query_summaries = {}
    for query_id, group in paper_df.groupby("query_id", dropna=False):
        paper_total_count = int(len(group))
        paper_pass_count = int(group["is_correct"].sum())
        all_papers_correct = (
            int(group["missing_paper_count"].max()) == 0
            and int(group["extra_paper_count"].max()) == 0
            and paper_total_count > 0
            and bool(group["is_correct"].all())
        )
        query_summaries[query_id] = {
            "paper_total_count": paper_total_count,
            "paper_pass_count": paper_pass_count,
            "paper_accuracy": paper_pass_count / paper_total_count if paper_total_count else 0,
            "min_paper_correctness": group["correctness_score"].min(skipna=False),
            "all_papers_correct": all_papers_correct,
            "query_outcome": "Correct" if all_papers_correct else "Needs review",
        }

    for column in QUERY_LEVEL_COLUMNS:
        df[column] = df["query_id"].map(
            {query_id: summary[column] for query_id, summary in query_summaries.items()}
        )
    return df


def align_to_all_paper_result_schema(df):
    """Order (and complete) the columns so every results CSV has the same schema."""
    df = df.copy()
    for column in ALL_PAPER_RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[ALL_PAPER_RESULT_COLUMNS]


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------


def _sibling_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}__{suffix}{output_path.suffix}")


def write_diagnostic_summaries(df, output_path: Path) -> None:
    """Write per-category / per-field summaries and the samples below the threshold."""
    summary_columns = [
        "sample_count",
        "correct_rate",
        "mean_correctness_score",
        *[f"mean_{metric}" for metric in METRIC_COLS],
    ]
    aggregations = {
        "is_correct": ["count", "mean"],
        "correctness_score": "mean",
        **{metric: "mean" for metric in METRIC_COLS if metric in df.columns},
    }
    rename_columns = {
        "is_correct_count": "sample_count",
        "is_correct_mean": "correct_rate",
        "correctness_score_mean": "mean_correctness_score",
        **{f"{metric}_mean": f"mean_{metric}" for metric in METRIC_COLS},
    }

    for group_col in ["category", "field_id", "reference_status"]:
        if group_col not in df.columns:
            continue
        summary = df.groupby(group_col, dropna=False).agg(aggregations)
        summary.columns = [
            "_".join(column).strip("_") if isinstance(column, tuple) else column
            for column in summary.columns
        ]
        summary = summary.rename(columns=rename_columns).reset_index()
        ordered_cols = [group_col, *[col for col in summary_columns if col in summary.columns]]
        summary = summary[ordered_cols].sort_values(
            ["correct_rate", "mean_correctness_score"], ascending=[True, True]
        )
        summary_path = _sibling_path(output_path, f"by_{group_col}")
        summary.to_csv(summary_path, index=False)
        logger.info(f"Diagnostic summary saved to {summary_path.absolute()}")

    failure_cols = [
        col
        for col in [
            "category",
            "field_id",
            "question",
            "user_input",
            "article_id",
            "article_title",
            "reference_status",
            "correctness_score",
            "outcome",
            "response",
            "reference",
            "retrieved_contexts",
            *METRIC_COLS,
        ]
        if col in df.columns
    ]
    failures = df[~df["is_correct"].astype(bool)][failure_cols].sort_values(
        "correctness_score", ascending=True
    )
    failures_path = _sibling_path(output_path, "needs_review")
    failures.to_csv(failures_path, index=False)
    logger.info(f"Needs-review samples saved to {failures_path.absolute()}")


def write_status_mask_summaries(df, output_path: Path) -> None:
    """Write the status confusion counts and the mislabelled cells."""
    if df.empty or "status_outcome" not in df.columns:
        return

    summary = (
        df.groupby(["reference_status", "response_status", "status_outcome"], dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )
    summary_path = _sibling_path(output_path, "status_summary")
    summary.to_csv(summary_path, index=False)
    logger.info(f"Status-mask summary saved to {summary_path.absolute()}")

    review_cols = [
        col
        for col in [
            "query_id",
            "eval_unit",
            "paper_index",
            "article_id",
            "article_title",
            "category",
            "field_id",
            "reference_status",
            "response_status",
            "status_outcome",
            "question",
            "user_input",
            "response",
            "reference",
            "retrieved_contexts",
        ]
        if col in df.columns
    ]
    errors = df[~df["status_match"].astype(bool)][review_cols].sort_values("status_outcome")
    errors_path = _sibling_path(output_path, "status_errors")
    errors.to_csv(errors_path, index=False)
    logger.info(f"Status-mask errors saved to {errors_path.absolute()}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Run a RAGAS evaluation against the live RAG API")

    data = parser.add_argument_group("dataset and output")
    data.add_argument(
        "--dataset",
        default=here / "datasets" / "ML4EHR.csv",
        help="Benchmark CSV (ML4EHR format) or a converted test_cases JSON file",
    )
    data.add_argument(
        "--output",
        default=here / "results" / "ragas_results.csv",
        help="Per-sample results CSV; summary CSVs are written next to it",
    )
    data.add_argument(
        "--write-json-dataset",
        default=None,
        help="Also write the converted test_cases JSON to this path",
    )
    data.add_argument(
        "--convert-only",
        action="store_true",
        help="Convert --dataset to JSON and exit (requires --write-json-dataset)",
    )
    data.add_argument("--limit", type=int, default=None, help="Evaluate at most this many questions")
    data.add_argument("--offset", type=int, default=0, help="Skip this many questions first")
    data.add_argument(
        "--expected-question-count",
        type=int,
        default=DEFAULT_EXPECTED_QUESTION_COUNT,
        help=(
            "Fail unless the run covers this many unique questions "
            f"(default {DEFAULT_EXPECTED_QUESTION_COUNT}; 0 disables the check for smoke runs)"
        ),
    )

    rag = parser.add_argument_group("RAG system")
    rag.add_argument(
        "--api-url",
        default="http://localhost:8001/api/v1/rag/local",
        help="Full URL of the RAG chat endpoint",
    )
    rag.add_argument(
        "--api-timeout",
        type=int,
        default=600,
        help="Timeout in seconds per RAG request (0 waits indefinitely)",
    )
    rag.add_argument(
        "--retrieval-scope",
        choices=["all_papers", "relevant_papers", "specific_paper"],
        default="all_papers",
        help="Retrieval scope sent to the RAG endpoint",
    )
    rag.add_argument("--paper-id", default=None, help="Article ID for --retrieval-scope specific_paper")
    rag.add_argument(
        "--paper-title", default=None, help="Title (or fragment) for --retrieval-scope specific_paper"
    )
    rag.add_argument(
        "--paper-index",
        type=int,
        default=None,
        help="1-based paper position in the CSV for --retrieval-scope specific_paper",
    )
    rag.add_argument(
        "--response-model",
        default=None,
        help="Ollama chat model tag to activate before querying (default: the active model)",
    )
    rag.add_argument(
        "--generator-seed",
        type=int,
        default=None,
        help="Fixed Ollama seed sent with every RAG request (default: unseeded sampling)",
    )
    rag.add_argument(
        "--model-info-url",
        default="http://localhost:8001/api/v1/model/active-models",
        help="Backend endpoint that reports the active models",
    )
    rag.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Base URL of the Ollama server (embeddings and Ollama judges)",
    )

    judge = parser.add_argument_group("RAGAS judge")
    judge.add_argument(
        "--judge-model",
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help=f"Judge model (default {DEFAULT_OPENAI_JUDGE_MODEL})",
    )
    judge.add_argument(
        "--judge-provider",
        choices=JUDGE_PROVIDERS,
        default="auto",
        help=(
            "Judge backend. 'auto' uses Ollama for tags containing ':', Mistral for "
            "known Mistral prefixes (e.g. 'ministral-') and OpenAI otherwise"
        ),
    )
    judge.add_argument(
        "--mistral-base-url",
        default=DEFAULT_MISTRAL_BASE_URL,
        help="Mistral OpenAI-compatible base URL (or set MISTRALAI_BASE_URL)",
    )
    judge.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        help="Judge seed (Ollama seed, OpenAI seed, Mistral random_seed; best-effort for APIs)",
    )
    judge.add_argument(
        "--judge-timeout",
        type=int,
        default=600,
        help="Timeout in seconds per judge call",
    )
    judge.add_argument(
        "--skip-judge-preflight",
        action="store_true",
        help="Skip the test request to the judge before scoring",
    )

    ragas = parser.add_argument_group("RAGAS scoring")
    ragas.add_argument(
        "--metrics",
        nargs="+",
        default=METRIC_COLS,
        choices=METRIC_COLS,
        help="RAGAS metrics to compute (must include answer_correctness)",
    )
    ragas.add_argument(
        "--ragas-reference-status",
        type=normalize_reference_status,
        choices=REFERENCE_STATUS_FILTERS,
        default=REPORTED_STATUS,
        help="Reference status of the cells sent to RAGAS (default: reported only)",
    )
    ragas.add_argument(
        "--correctness-threshold",
        type=float,
        default=DEFAULT_CORRECTNESS_THRESHOLD,
        help="Answer correctness at or above which a sample is labelled correct",
    )
    ragas.add_argument(
        "--max-eval-contexts",
        type=int,
        default=None,
        help="Maximum retrieved contexts per sample passed to RAGAS (default/0: no limit)",
    )
    ragas.add_argument(
        "--max-eval-context-chars",
        type=int,
        default=2500,
        help="Maximum characters per retrieved context passed to RAGAS",
    )
    ragas.add_argument(
        "--ragas-max-workers", type=int, default=1, help="Concurrent RAGAS workers"
    )
    ragas.add_argument(
        "--raise-eval-exceptions",
        action="store_true",
        help="Raise RAGAS errors instead of recording failed metric calls as NaN",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    """Check the arguments and turn the '0 = no limit' values into None."""
    if args.convert_only and not args.write_json_dataset:
        parser.error("--convert-only requires --write-json-dataset")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than 0")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.expected_question_count < 0:
        parser.error("--expected-question-count cannot be negative")
    if args.paper_index is not None and args.paper_index < 1:
        parser.error("--paper-index must be greater than 0")
    if args.api_timeout < 0:
        parser.error("--api-timeout cannot be negative")
    if args.max_eval_contexts is not None and args.max_eval_contexts < 0:
        parser.error("--max-eval-contexts cannot be negative")
    if not 0 <= args.correctness_threshold <= 1:
        parser.error("--correctness-threshold must be between 0 and 1")
    if ANSWER_CORRECTNESS_METRIC not in args.metrics:
        parser.error(
            f"--metrics must include {ANSWER_CORRECTNESS_METRIC}; the correctness label uses it"
        )

    paper_selector_requested = bool(args.paper_id or args.paper_title or args.paper_index)
    if args.retrieval_scope == "specific_paper" and not paper_selector_requested:
        args.paper_index = 1
    if args.retrieval_scope != "specific_paper" and paper_selector_requested:
        parser.error(
            "--paper-id, --paper-title, and --paper-index require --retrieval-scope specific_paper"
        )

    args.expected_question_count = args.expected_question_count or None
    args.api_timeout = args.api_timeout or None
    args.max_eval_contexts = args.max_eval_contexts or None


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    # 1. Load the benchmark.
    logger.info(f"Loading dataset from {args.dataset}...")
    test_cases = load_test_cases(args.dataset)
    logger.info(f"Loaded {len(test_cases)} test cases.")

    if args.retrieval_scope == "specific_paper":
        test_cases = select_specific_paper_cases(
            test_cases,
            paper_id=args.paper_id,
            paper_title=args.paper_title,
            paper_index=args.paper_index,
        )
        selected_paper = test_cases[0]["metadata"]
        logger.info(
            "Specific-paper evaluation selected article_id={} title={} with {} question categories.",
            selected_paper.get("selected_paper_id"),
            selected_paper.get("selected_paper_title"),
            len(test_cases),
        )

    if args.write_json_dataset:
        write_test_cases_json(test_cases, args.write_json_dataset)
        logger.info(f"Converted dataset saved to {Path(args.write_json_dataset).absolute()}")
    if args.convert_only:
        return

    selected_test_cases = select_test_cases_for_run(test_cases, args.offset, args.limit)
    validate_expected_question_count(
        selected_test_cases, args.expected_question_count, "Selected evaluation set"
    )

    # 2. Query the RAG API.
    if args.response_model:
        validate_response_model_available(
            args.response_model, args.model_info_url, args.ollama_base_url
        )
        activate_chat_model(args.response_model, args.model_info_url)
        logger.info(f"Activated chat model for RAG responses: {args.response_model}")

    logger.info("Querying live API for each question...")
    eval_dataset, sample_metadata = build_eval_dataset(
        selected_test_cases, args, query_id_offset=args.offset
    )
    if len(eval_dataset) == 0:
        logger.error("No samples were successfully collected. Aborting.")
        sys.exit(1)
    validate_sample_metadata_question_count(
        sample_metadata, args.expected_question_count, "Collected API response set"
    )

    # 3. Label reported / not reported for every cell (no judge involved).
    import pandas as pd

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_mask_df = build_status_mask_dataframe(eval_dataset, sample_metadata)
    status_mask_path = _sibling_path(output_path, "status_mask")
    status_mask_df.to_csv(status_mask_path, index=False)
    logger.info(f"Status-mask results saved to {status_mask_path.absolute()}")
    write_status_mask_summaries(status_mask_df, status_mask_path)

    # 4. Score the selected cells with RAGAS.
    eval_dataset, sample_metadata = filter_eval_dataset_by_reference_status(
        eval_dataset, sample_metadata, args.ragas_reference_status
    )
    if len(eval_dataset) == 0:
        logger.error(
            f"No samples matched --ragas-reference-status={args.ragas_reference_status}. Aborting."
        )
        sys.exit(1)
    # Status filtering can remove whole question categories; coverage was checked above.
    filtered_question_count = len({row["query_id"] for row in sample_metadata})
    logger.info(
        f"Collected {len(eval_dataset)} samples across {filtered_question_count} questions "
        f"for RAGAS (reference_status={args.ragas_reference_status}). Running RAGAS..."
    )
    scores_df = run_ragas(eval_dataset, args)

    # 5. Add labels and write the results.
    df = pd.concat([pd.DataFrame(sample_metadata), scores_df.reset_index(drop=True)], axis=1)
    df = add_status_mask_columns(df)
    df = add_answer_correctness_labels(df, threshold=args.correctness_threshold)
    df = add_query_level_correctness(df)

    models = get_active_ollama_models(args.model_info_url)
    df["response_model"] = models["chat_model"]
    df["judge_model"] = args.judge_model
    df["embedding_model"] = models["embedding_model"]
    df = align_to_all_paper_result_schema(df)

    df.to_csv(output_path, index=False)
    logger.info(f"Per-sample results saved to {output_path.absolute()}")
    write_diagnostic_summaries(df, output_path)


if __name__ == "__main__":
    main()
