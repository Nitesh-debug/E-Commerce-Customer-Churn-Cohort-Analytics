"""
test_pipeline.py
Automated Quality Assurance & Pipeline Validation Tests

Validates:
1. SQLite Database integrity ('analytics.db' table structure, row counts, non-negativity, nulls).
2. SQL Analytics outputs ('rfm_customer_segments.csv', 'cohort_retention_matrix.csv').
3. Machine Learning artifacts ('churn_model.joblib', prediction schema, metrics thresholds).
"""

import json
import os
import sqlite3
import joblib
import pandas as pd
import unittest

from feature_engineering import FEATURE_COLUMNS


class TestAnalyticsPipeline(unittest.TestCase):

    def setUp(self):
        self.db_path = "analytics.db"
        self.output_dir = "output"
        self.model_path = "churn_model.joblib"

    def test_database_exists_and_populated(self):
        """Verify SQLite database exists and contains expected record count."""
        self.assertTrue(os.path.exists(self.db_path), "analytics.db not found!")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions';")
        self.assertIsNotNone(cursor.fetchone(), "Table 'transactions' does not exist in analytics.db.")

        # Check count
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT CustomerID) FROM transactions;")
        row_count, customer_count = cursor.fetchone()
        self.assertEqual(row_count, 397884, "Record count does not match expected 397,884.")
        self.assertEqual(customer_count, 4338, "Customer count does not match expected 4,338.")

        # Check no negative quantities or prices
        cursor.execute("SELECT COUNT(*) FROM transactions WHERE Quantity <= 0 OR UnitPrice <= 0 OR CustomerID IS NULL;")
        invalid_count = cursor.fetchone()[0]
        self.assertEqual(invalid_count, 0, "Found invalid records (negative values or NULL CustomerID) in transactions.")

        conn.close()

    def test_rfm_segments_export(self):
        """Verify RFM customer segments CSV exists, has correct shape and required segments."""
        rfm_path = os.path.join(self.output_dir, "rfm_customer_segments.csv")
        self.assertTrue(os.path.exists(rfm_path), "rfm_customer_segments.csv not found in output/.")
        df = pd.read_csv(rfm_path)

        self.assertEqual(len(df), 4338, "RFM segment records must equal total unique customers (4,338).")
        expected_cols = {"CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Customer_Segment"}
        self.assertTrue(expected_cols.issubset(df.columns), "Missing required columns in RFM CSV.")

        # Check required segments exist
        segments = set(df["Customer_Segment"].unique())
        self.assertIn("Champions", segments)
        self.assertIn("At-Risk", segments)
        self.assertIn("Churned", segments)

        # Check RFM scores are between 1 and 5
        self.assertTrue(df["R_Score"].between(1, 5).all(), "R_Score values must be between 1 and 5.")
        self.assertTrue(df["F_Score"].between(1, 5).all(), "F_Score values must be between 1 and 5.")
        self.assertTrue(df["M_Score"].between(1, 5).all(), "M_Score values must be between 1 and 5.")

    def test_cohort_retention_matrix(self):
        """Verify Cohort Retention matrices exist and initial retention is 100%."""
        matrix_path = os.path.join(self.output_dir, "cohort_retention_rates.csv")
        self.assertTrue(os.path.exists(matrix_path), "cohort_retention_rates.csv not found in output/.")
        df = pd.read_csv(matrix_path)

        self.assertTrue("CohortMonth" in df.columns, "CohortMonth column missing.")
        self.assertTrue("Month_0" in df.columns, "Month_0 column missing.")

        # Month_0 retention must be 100.0%
        self.assertTrue((df["Month_0"] == 100.0).all(), "All Month_0 retention rates must be 100.0%.")

    def test_churn_model_artifact_and_metrics(self):
        """Verify trained ML model can be loaded, executes inference, and achieves expected metrics."""
        self.assertTrue(os.path.exists(self.model_path), "churn_model.joblib not found!")
        model = joblib.load(self.model_path)

        # Test inference with dummy sample
        dummy_sample = pd.DataFrame(
            [[3, 500.0, 166.67, 120, 40.0, 15, 200, 0.015, 1]],
            columns=FEATURE_COLUMNS,
        )
        prediction = model.predict(dummy_sample)
        probabilities = model.predict_proba(dummy_sample)

        self.assertIn(prediction[0], [0, 1], "Prediction output must be 0 or 1.")
        self.assertEqual(probabilities.shape, (1, 2), "Prediction probabilities shape mismatch.")

        # Check evaluation metrics JSON
        metrics_path = os.path.join(self.output_dir, "model_evaluation_metrics.json")
        self.assertTrue(os.path.exists(metrics_path), "model_evaluation_metrics.json not found in output/.")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertGreater(metrics["roc_auc"], 0.90, "Model ROC-AUC is below expected threshold (0.90).")
        self.assertGreater(metrics["recall"], 0.80, "Model Recall on churners is below expected threshold (0.80).")
        self.assertGreater(metrics["accuracy"], 0.80, "Model Accuracy is below expected threshold (0.80).")


if __name__ == "__main__":
    unittest.main(verbosity=2)
