"""Offline checks for correcting one reference without regenerating responses."""

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from rag_evaluation import rescore_saved_reference as rescore


def saved_row(article="paper-a", query="9"):
    return {
        "query_id": query, "article_id": article, "category": "cohort.demographics",
        "eval_unit": "paper", "reference_status": "reported", "reference": "Old reference",
        "user_input": "Demographics?", "response": "Saved answer",
        "retrieved_contexts": "['Saved context']", "judge_model": "ministral-8b-2512",
        "embedding_model": "nomic-embed-text:latest", "correctness_threshold": "0.75",
        "context_recall": "0.2", "llm_context_precision_with_reference": "1.0",
        "answer_correctness": "0.3", "correctness_score": "0.3", "is_correct": "False",
        "correctness_metrics": "answer_correctness", "outcome": "Needs review",
        "missing_paper_count": "0", "extra_paper_count": "0",
        "paper_total_count": "2", "paper_pass_count": "0", "paper_accuracy": "0.0",
        "min_paper_correctness": "0.3", "all_papers_correct": "False", "query_outcome": "Needs review",
        "faithfulness": "0.123456789", "answer_relevancy": "0.987654321",
        "manual_label": "generation_miss", "manual_notes": "Keep this, including punctuation.",
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestSavedReferenceRescoring(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.results = Path(self.temp.name) / "results.csv"
        self.dataset = Path(self.temp.name) / "gold.csv"
        self.rows = [saved_row(), saved_row("paper-b"), saved_row("paper-c", "10")]
        write_csv(self.results, self.rows)
        write_csv(self.dataset, [{"article_id": "paper-a", "category": "cohort.demographics",
                                  "reference_answer": "Corrected reference", "status": "reported"}])
        self.scores = dict(zip(rescore.METRICS, [0.8, 1.0, 0.9]))

    def prepare(self):
        return rescore.prepare(self.results, self.dataset, 9, "paper-a")

    def test_preparation_keeps_saved_response_and_context(self):
        rows, columns, index, sample, original, gold = self.prepare()
        self.assertEqual(index, 0)
        self.assertEqual(sample, {"user_input": "Demographics?", "response": "Saved answer",
                                  "retrieved_contexts": ["Saved context"], "reference": "Corrected reference"})
        self.assertEqual(original, self.results.read_bytes())
        self.assertEqual(gold, self.dataset.read_bytes())
        self.assertEqual(rows, self.rows)
        self.assertEqual(columns, list(self.rows[0]))

    def test_duplicate_saved_match_is_rejected(self):
        write_csv(self.results, self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Expected one saved"):
            self.prepare()

    def test_reference_status_change_is_rejected(self):
        write_csv(self.dataset, [{"article_id": "paper-a", "category": "cohort.demographics",
                                  "reference_answer": "Corrected", "status": "no_reported"}])
        with self.assertRaisesRegex(ValueError, "Reference status also changed"):
            self.prepare()

    def test_updates_only_target_and_its_query_summaries(self):
        corrected = rescore.update_scores(self.rows, 0, "Corrected", self.scores)
        self.assertEqual(corrected[0]["is_correct"], "True")
        self.assertEqual(corrected[0]["outcome"], "Correct")
        for row in corrected[:2]:
            self.assertEqual(row["paper_pass_count"], "1")
            self.assertEqual(row["paper_accuracy"], "0.5")
        for column in self.rows[1]:
            if column not in rescore.QUERY_COLUMNS:
                self.assertEqual(corrected[1][column], self.rows[1][column])
        self.assertEqual(corrected[2], self.rows[2])
        for column in ["response", "retrieved_contexts", "faithfulness", "answer_relevancy", "manual_label", "manual_notes"]:
            self.assertEqual(corrected[0][column], self.rows[0][column])
        self.assertEqual(self.rows[0]["reference"], "Old reference")

    def test_failed_scores_do_not_modify_rows(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid score"):
                rescore.update_scores(self.rows, 0, "Corrected", {**self.scores, "context_recall": value})
        self.assertEqual(self.rows[0]["reference"], "Old reference")

    def test_backup_audit_and_corrected_csv(self):
        rows, columns, index, sample, original, gold = self.prepare()
        corrected = rescore.update_scores(rows, index, sample["reference"], self.scores)
        backup, audit = rescore.save_correction(self.results, self.dataset, original, gold, corrected, columns, {})
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(list(csv.DictReader(io.StringIO(self.results.read_text()))), corrected)
        self.assertEqual(json.loads(audit.read_text())["backup"], str(backup))

    def test_concurrent_edit_is_not_overwritten(self):
        rows, columns, index, sample, original, gold = self.prepare()
        self.results.write_text("A concurrent change")
        with self.assertRaisesRegex(RuntimeError, "Input files changed"):
            rescore.save_correction(self.results, self.dataset, original, gold, rows, columns, {})
        self.assertEqual(self.results.read_text(), "A concurrent change")
        self.assertEqual(len(list(Path(self.temp.name).glob('*.before_rescore*'))), 0)

    def test_preview_makes_no_network_calls_or_writes(self):
        original = self.results.read_bytes()
        args = ["rescore", "--results", str(self.results), "--dataset", str(self.dataset),
                "--query-id", "9", "--article-id", "paper-a"]
        with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO), patch.object(rescore, "score_sample") as score:
            rescore.main()
        score.assert_not_called()
        self.assertEqual(self.results.read_bytes(), original)

    def test_scoring_uses_one_saved_sample_and_three_metrics(self):
        args = SimpleNamespace(model_info_url="unused", judge_timeout=10)
        sample = self.prepare()[3]
        embeddings = SimpleNamespace(embeddings=SimpleNamespace(model="nomic-embed-text:latest"))
        with patch.object(rescore.evaluator, "get_active_ollama_models", return_value={"embedding_model": "nomic-embed-text:latest"}), \
                patch.object(rescore.evaluator, "build_judge", return_value=(object(), embeddings)), \
                patch("ragas.metrics.ContextRecall"), patch("ragas.metrics.LLMContextPrecisionWithReference"), \
                patch("ragas.metrics.AnswerCorrectness"), patch("ragas.evaluate") as evaluate:
            evaluate.return_value.to_pandas.return_value = pd.DataFrame([self.scores])
            self.assertEqual(rescore.score_sample(args, self.rows[0], sample), self.scores)
            call = evaluate.call_args.kwargs
            self.assertEqual(len(call["dataset"].samples), 1)
            self.assertEqual(call["dataset"].samples[0].response, "Saved answer")
            self.assertEqual(call["dataset"].samples[0].retrieved_contexts, ["Saved context"])
            self.assertEqual(len(call["metrics"]), 3)
            self.assertTrue(call["raise_exceptions"])
            self.assertEqual(args.judge_model, "ministral-8b-2512")


if __name__ == "__main__":
    unittest.main()
