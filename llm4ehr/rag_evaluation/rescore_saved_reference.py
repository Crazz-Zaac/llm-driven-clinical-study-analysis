"""Correct one saved reference and rescore it without calling the RAG endpoint."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

try:
    from . import evaluate_rag as evaluator
except ImportError:
    import evaluate_rag as evaluator


METRICS = ["context_recall", "llm_context_precision_with_reference", "answer_correctness"]
QUERY_COLUMNS = [
    "paper_total_count", "paper_pass_count", "paper_accuracy", "min_paper_correctness",
    "all_papers_correct", "query_outcome",
]


def prepare(results, dataset, query_id, article_id):
    result_bytes, dataset_bytes = results.read_bytes(), dataset.read_bytes()
    reader = csv.DictReader(io.StringIO(result_bytes.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    matches = [i for i, row in enumerate(rows)
               if row["query_id"] == str(query_id) and row["article_id"] == article_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved observation; found {len(matches)}")
    index = matches[0]
    row = rows[index]
    if row["eval_unit"] != "paper":
        raise ValueError("Only per-paper results can be rescored with this utility.")

    gold = pd.read_csv(io.BytesIO(dataset_bytes), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    gold["article_id"] = gold["article_id"].str.strip().replace("", pd.NA).ffill()
    match = gold.loc[gold["article_id"].eq(article_id) & gold["category"].str.strip().eq(row["category"])]
    if len(match) != 1:
        raise ValueError(f"Expected one canonical reference; found {len(match)}")
    canonical = match.iloc[0]
    if evaluator.normalize_reference_status(canonical["status"]) != evaluator.normalize_reference_status(row["reference_status"]):
        raise ValueError("Reference status also changed; this utility only corrects reference text.")
    reference = canonical["reference_answer"]
    if not reference.strip():
        raise ValueError("The canonical reference is empty.")
    contexts = ast.literal_eval(row["retrieved_contexts"])
    if not isinstance(contexts, list) or not all(isinstance(item, str) for item in contexts):
        raise ValueError("Saved contexts must be a serialized list of strings.")
    if not row["judge_model"] or not row["embedding_model"]:
        raise ValueError("Recorded judge and embedding models are required.")
    sample = {"user_input": row["user_input"], "response": row["response"],
              "retrieved_contexts": contexts, "reference": reference}
    return rows, reader.fieldnames, index, sample, result_bytes, dataset_bytes


def update_scores(rows, index, reference, scores):
    if set(scores) != set(METRICS):
        raise ValueError("All three reference-dependent scores are required.")
    for metric, value in scores.items():
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"Invalid score for {metric}: {value}")
    updated = [dict(row) for row in rows]
    row = updated[index]
    threshold = float(row["correctness_threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved correctness threshold.")
    row["reference"] = reference
    for metric, value in scores.items():
        row[metric] = str(float(value))
    labeled = evaluator.add_answer_correctness_labels(
        pd.DataFrame({"answer_correctness": [float(scores["answer_correctness"])]}), threshold,
    ).iloc[0]
    for column in ["correctness_score", "is_correct", "correctness_threshold", "correctness_metrics", "outcome"]:
        row[column] = str(labeled[column])

    # Refresh only this query's derived summaries; preserve all manual-review fields.
    indices = [i for i, item in enumerate(updated) if item["query_id"] == row["query_id"]]
    group = pd.DataFrame([updated[i] for i in indices])
    flags = group["is_correct"].str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("Unrecognized saved correctness labels in this query.")
    group["is_correct"] = flags
    for column in ["correctness_score", "missing_paper_count", "extra_paper_count"]:
        group[column] = pd.to_numeric(group[column], errors="raise")
    refreshed = evaluator.add_query_level_correctness(group)
    for position, original_index in enumerate(indices):
        for column in QUERY_COLUMNS:
            updated[original_index][column] = str(refreshed.iloc[position][column])
    return updated


def score_sample(args, saved_row, sample):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import AnswerCorrectness, ContextRecall, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    active = evaluator.get_active_ollama_models(args.model_info_url)
    if active["embedding_model"] != saved_row["embedding_model"]:
        raise ValueError(f"Set the active embedding model to {saved_row['embedding_model']} before rescoring.")
    args.judge_model = saved_row["judge_model"]
    args.judge_provider = "auto"
    args.skip_judge_preflight = True
    judge, embeddings = evaluator.build_judge(args)
    if embeddings.embeddings.model != saved_row["embedding_model"]:
        raise ValueError("The active embedding model changed during setup.")
    metrics = [ContextRecall(llm=judge), LLMContextPrecisionWithReference(llm=judge),
               AnswerCorrectness(llm=judge, embeddings=embeddings)]
    result = evaluate(
        dataset=EvaluationDataset(samples=[SingleTurnSample(**sample)]), metrics=metrics,
        run_config=RunConfig(timeout=args.judge_timeout, max_retries=1, max_workers=1),
        raise_exceptions=True,
    ).to_pandas()
    if len(result) != 1:
        raise ValueError("Expected exactly one rescored observation.")
    return {metric: float(result.iloc[0][metric]) for metric in METRICS}


def save_correction(results, dataset, original_bytes, dataset_bytes, rows, columns, audit):
    if results.read_bytes() != original_bytes or dataset.read_bytes() != dataset_bytes:
        raise RuntimeError("Input files changed during rescoring; no correction was applied.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = results.with_name(f"{results.stem}.before_rescore_{stamp}.csv")
    audit_path = results.with_name(f"{results.stem}.rescore_{stamp}.json")
    audit.update({"timestamp_utc": stamp, "backup": str(backup),
                  "results_sha256_before": hashlib.sha256(original_bytes).hexdigest(),
                  "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest()})
    # Keep an exact backup and an audit record before atomically replacing the CSV.
    with backup.open("xb") as stream:
        stream.write(original_bytes)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=results.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.chmod(results.stat().st_mode & 0o777)
        if results.read_bytes() != original_bytes or dataset.read_bytes() != dataset_bytes:
            raise RuntimeError("Input files changed before replacement; original results retained.")
        os.replace(temporary, results)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return backup, audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets/ML4EHR.csv")
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--article-id", required=True)
    parser.add_argument("--apply", action="store_true", help="Call the recorded judge and update the CSV with a backup; otherwise preview only.")
    parser.add_argument("--model-info-url", default="http://localhost:8001/api/v1/model/active-models")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--mistral-base-url", default=evaluator.DEFAULT_MISTRAL_BASE_URL)
    parser.add_argument("--judge-timeout", type=int, default=600)
    args = parser.parse_args()
    rows, columns, index, sample, original, gold_bytes = prepare(
        args.results, args.dataset, args.query_id, args.article_id,
    )
    saved = rows[index]
    print(f"Spreadsheet row {index + 2}: query {saved['query_id']}, {saved['category']}, {saved['article_id']}")
    print(f"Judge: {saved['judge_model']}; embedding model: {saved['embedding_model']}")
    if sample["reference"] == saved["reference"]:
        print("The saved reference already matches the dataset. No changes or API calls made.")
        return
    print("\n".join(difflib.unified_diff(
        saved["reference"].splitlines(), sample["reference"].splitlines(),
        fromfile="saved reference", tofile="canonical reference", lineterm="",
    )))
    # Validate derived-column handling before any remote calls.
    update_scores(rows, index, sample["reference"], {metric: float(saved[metric]) for metric in METRICS})
    if not args.apply:
        print("Dry run only. Add --apply to rescore three metrics and update this CSV. No API calls made.")
        return
    scores = score_sample(args, saved, sample)
    corrected = update_scores(rows, index, sample["reference"], scores)
    audit = {
        "results": str(args.results.resolve()), "dataset": str(args.dataset.resolve()),
        "query_id": args.query_id, "article_id": args.article_id, "category": saved["category"],
        "judge_model": saved["judge_model"], "embedding_model": saved["embedding_model"],
        "old_reference": saved["reference"], "new_reference": sample["reference"],
        "old_scores": {metric: saved[metric] for metric in METRICS}, "new_scores": scores,
        "response_and_contexts_unchanged": True,
    }
    backup, audit_path = save_correction(args.results, args.dataset, original, gold_bytes, corrected, columns, audit)
    print(f"Updated {args.results}\nBackup: {backup}\nAudit: {audit_path}")
    print("Rerun rag_evaluation_analysis.ipynb to refresh tables, figures, and input-comparability checks.")


if __name__ == "__main__":
    main()
