"""
test_api.py
Automated Unit and Integration Tests for FastAPI Churn Prediction API.

Executes automated verification using FastAPI's TestClient:
1. GET /health: Model loading, status, and feature verification.
2. POST /predict: High-risk customer inference validation.
3. POST /predict: Low-risk customer inference validation.
4. POST /predict: Fallback/auto-computed defaults inference validation.
5. POST /predict: Input validation error handling (422 Unprocessable Entity).
6. GET /: Root navigation check.
"""

import unittest
from fastapi.testclient import TestClient

from app import app


class TestChurnPredictionAPI(unittest.TestCase):
    """Test suite covering FastAPI churn prediction service."""

    @classmethod
    def setUpClass(cls):
        """Initializes TestClient context manager to trigger lifespan events."""
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Verify root endpoint responds with API metadata and documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("docs_url", data)
        self.assertIn("health_url", data)
        self.assertIn("predict_url", data)
        self.assertEqual(data["docs_url"], "/docs")
        self.assertEqual(data["configuration"]["high_risk_threshold"], 0.70)
        self.assertEqual(data["configuration"]["medium_risk_threshold"], 0.30)
        self.assertEqual(data["model_version"], "churn_model.joblib")

    def test_health_endpoint(self):
        """Verify health check endpoint returns 200, healthy status, and verified model load."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_artifact"], "churn_model.joblib")
        self.assertEqual(data["feature_count"], 9)
        self.assertIn("PurchaseVelocity", data["features"])
        self.assertIn("CustomerTenure", data["features"])
        self.assertEqual(data["model_version"], "churn_model.joblib")
        self.assertEqual(len(data["model_sha256"]), 64)

    def test_model_status_endpoint(self):
        """Model status exposes the serving artifact and feature contract."""
        response = self.client.get("/model/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["model_version"], "churn_model.joblib")
        self.assertEqual(len(data["model_sha256"]), 64)
        self.assertEqual(len(data["feature_columns"]), 9)

    def test_predict_high_risk_customer(self):
        """
        Verify that an inactive, single-purchase, long-tenure customer
        is predicted with high churn probability and categorized as High Risk.
        """
        payload = {
            "frequency": 1,
            "monetary": 120.0,
            "recency_days": 280,
            "tenure": 280,
            "purchase_velocity": 0.0035,
            "avg_order_value": 120.0,
            "total_units": 15,
            "avg_basket_size": 15.0,
            "distinct_products": 5,
            "is_uk": 1,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate structure
        self.assertIn("churn_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("recommended_action", data)
        self.assertIn("predicted_churn", data)

        # Validate types and ranges
        self.assertIsInstance(data["churn_probability"], float)
        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertEqual(data["risk_level"], "High Risk")
        self.assertGreaterEqual(data["churn_probability"], 0.70)
        self.assertEqual(data["predicted_churn"], 1)
        self.assertIn("Win-Back", data["recommended_action"])

    def test_predict_low_risk_customer(self):
        """
        Verify that a highly active repeat customer with fast purchase velocity
        is classified as Low Risk and suggested loyalty expansion.
        """
        payload = {
            "frequency": 12,
            "monetary": 5500.0,
            "recency_days": 8,
            "tenure": 120,
            "purchase_velocity": 0.099,
            "avg_order_value": 458.33,
            "total_units": 1800,
            "avg_basket_size": 150.0,
            "distinct_products": 60,
            "is_uk": 1,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIsInstance(data["churn_probability"], float)
        self.assertTrue(0.0 <= data["churn_probability"] <= 1.0)
        self.assertEqual(data["risk_level"], "Low Risk")
        self.assertLess(data["churn_probability"], 0.30)
        self.assertEqual(data["predicted_churn"], 0)
        self.assertIn("Loyalty", data["recommended_action"])

    def test_predict_auto_computed_defaults(self):
        """
        Verify inference succeeds when only the required core metrics are provided,
        confirming that secondary and derived features compute automatically.
        """
        payload = {
            "frequency": 5,
            "monetary": 1250.0,
            "tenure": 150,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("churn_probability", data)
        self.assertIn(data["risk_level"], ["Low Risk", "Medium Risk", "High Risk"])
        self.assertTrue(len(data["recommended_action"]) > 10)

    def test_predict_invalid_frequency(self):
        """Verify API returns 422 Unprocessable Entity when frequency <= 0."""
        payload = {
            "frequency": 0,
            "monetary": 500.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_predict_negative_monetary(self):
        """Verify API returns 422 Unprocessable Entity when monetary is negative."""
        payload = {
            "frequency": 2,
            "monetary": -100.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_predict_invalid_type(self):
        """Verify API returns 422 when non-numeric data is provided."""
        payload = {
            "frequency": "invalid_number",
            "monetary": 100.0,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_upload_valid_transaction_csv(self):
        """Validate a business-friendly raw transaction CSV is accepted via upload."""
        csv_content = (
            "customer_id,order_id,order_date,amount,quantity,product_id,country\n"
            "C1001,INV-001,2024-01-10,120.00,2,P-1,United Kingdom\n"
            "C1001,INV-002,2024-01-30,90.00,1,P-2,United Kingdom\n"
            "C1002,INV-003,2024-02-05,300.00,5,P-3,France\n"
        )
        response = self.client.post(
            "/upload",
            files={"file": ("transactions.csv", csv_content, "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("customers_processed", data)
        self.assertIn("feature_rows", data)
        self.assertGreaterEqual(data["customers_processed"], 1)

    def test_upload_invalid_csv_missing_required_columns(self):
        """Reject a CSV missing required fields with a 422 response."""
        csv_content = "customer_id,order_date,amount\nC1001,2024-01-10,100.00\n"
        response = self.client.post(
            "/upload",
            files={"file": ("bad.csv", csv_content, "text/csv")},
        )
        self.assertEqual(response.status_code, 422)

    def test_upload_minimal_transaction_csv_uses_optional_defaults(self):
        """Accept the four required transaction fields and derive optional defaults."""
        csv_content = (
            "customer_id,order_date,amount,quantity\n"
            "C1501,2024-01-10,120.00,2\n"
            "C1501,2024-01-30,90.00,1\n"
        )
        response = self.client.post(
            "/upload",
            files={"file": ("minimal.csv", csv_content, "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customers_processed"], 1)
        self.assertIn("feature_rows", data)

    def test_customer_lookup_returns_prediction(self):
        """Customer lookup should return a prediction and analysis payload."""
        csv_content = (
            "customer_id,order_id,order_date,amount,quantity,product_id,country\n"
            "C2001,INV-100,2024-01-09,100.00,1,P-01,United Kingdom\n"
            "C2001,INV-101,2024-02-09,120.00,2,P-02,United Kingdom\n"
            "C2001,INV-102,2024-03-09,80.00,1,P-03,United Kingdom\n"
            "C2002,INV-200,2024-09-01,250.00,5,P-10,France\n"
        )
        upload_response = self.client.post(
            "/upload",
            files={"file": ("customers.csv", csv_content, "text/csv")},
        )
        self.assertEqual(upload_response.status_code, 200)

        response = self.client.get("/customers/C2001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer_id"], "C2001")
        self.assertIn("churn_probability", data)
        self.assertIn("risk_level", data)
        self.assertIn("rfm_segment", data)

    def test_customer_lookup_handles_quoted_identifier(self):
        """Customer identifiers are treated as values rather than SQL fragments."""
        response = self.client.get("/customers/C2001' OR '1'='1")
        self.assertEqual(response.status_code, 404)

    def test_bulk_prediction_returns_rows(self):
        """Bulk prediction endpoint should process multiple customer rows."""
        csv_content = (
            "customer_id,order_id,order_date,amount,quantity,product_id,country\n"
            "C3001,INV-A,2024-01-02,200.00,3,P-1,United Kingdom\n"
            "C3002,INV-B,2024-04-02,500.00,8,P-2,Germany\n"
            "C3003,INV-C,2024-08-02,150.00,2,P-3,United Kingdom\n"
        )
        response = self.client.post(
            "/predict/bulk",
            files={"file": ("bulk.csv", csv_content, "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rows", data)
        self.assertGreaterEqual(len(data["rows"]), 1)

    def test_priority_endpoint_returns_ranked_candidates(self):
        """Retention priorities should be derived from uploaded transactions and model outputs."""
        csv_content = (
            "customer_id,order_id,order_date,amount,quantity,product_id,country\n"
            "C4001,INV-A,2024-01-02,200.00,3,P-1,United Kingdom\n"
            "C4001,INV-B,2024-02-02,500.00,8,P-2,United Kingdom\n"
            "C4002,INV-C,2024-08-02,150.00,2,P-3,France\n"
        )
        upload_response = self.client.post(
            "/upload",
            files={"file": ("priority.csv", csv_content, "text/csv")},
        )
        self.assertEqual(upload_response.status_code, 200)

        response = self.client.get("/priority")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], len(data["candidates"]))
        self.assertGreaterEqual(data["count"], 1)
        self.assertIn("priority_score", data["candidates"][0])
        self.assertIn("recommended_action", data["candidates"][0])

    def test_campaign_generation_is_simulated(self):
        """The campaign endpoint should store and return a simulated campaign record."""
        payload = {
            "customer_id": "C9999",
            "risk_level": "High Risk",
            "rfm_segment": "Champions",
            "monetary_value": 2500.0,
            "churn_probability": 0.82,
            "campaign_type": "VIP Win-Back",
            "offer": "20% reactivation credit",
            "message": "We value your business."
        }
        response = self.client.post("/campaign/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SIMULATED")
        self.assertIn("campaign_type", data)
        self.assertEqual(data["model_version"], "churn_model.joblib")


if __name__ == "__main__":
    unittest.main()
