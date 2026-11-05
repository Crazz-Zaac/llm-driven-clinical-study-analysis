"""Status-label compatibility for dataset conversion and reporting detection."""

import unittest

import pandas as pd

from rag_evaluation import evaluate_rag as evaluator


class TestReportingStatus(unittest.TestCase):
    def test_absence_status_aliases_use_one_canonical_label(self):
        for label in ["not reported", "not_reported", "no_reported", "no reported",
                      "non-reported", "unreported", " NOT REPORTED "]:
            with self.subTest(label=label):
                self.assertEqual(evaluator.normalize_reference_status(label), "no_reported")

    def test_absence_responses_are_detected(self):
        for response in ["not reported", "not_reported", "no_reported", "Not reported in the paper.",
                         "The study did not report this information.", "", evaluator.MISSING_PAPER_RESPONSE]:
            with self.subTest(response=response):
                self.assertEqual(evaluator.infer_response_status(response), "no_reported")

    def test_confusion_outcomes_accept_legacy_and_canonical_statuses(self):
        cases = [
            ("reported", "reported", "reported_gt_marked_reported"),
            ("reported", "not reported", "reported_gt_marked_no_reported"),
            ("not_reported", "no_reported", "no_reported_gt_marked_no_reported"),
            ("no_reported", "reported", "no_reported_gt_marked_reported"),
        ]
        for reference, response, outcome in cases:
            with self.subTest(reference=reference, response=response):
                self.assertEqual(evaluator.status_outcome(reference, response), outcome)

    def test_status_masks_use_canonical_labels_and_keep_response_text(self):
        original = pd.DataFrame({
            "reference_status": ["reported", "not_reported", "no_reported", "reported"],
            "response": ["Not reported.", "The sample contained 100 patients.",
                         "no_reported", "The sample contained 100 patients."],
        })
        result = evaluator.add_status_mask_columns(original)
        self.assertEqual(result["reference_status"].tolist(), ["reported", "no_reported", "no_reported", "reported"])
        self.assertEqual(result["response_status"].tolist(), ["no_reported", "reported", "no_reported", "reported"])
        self.assertEqual(result["status_match"].tolist(), [False, False, True, True])
        self.assertTrue(result["response"].equals(original["response"]))
        self.assertEqual(original.loc[1, "reference_status"], "not_reported")
