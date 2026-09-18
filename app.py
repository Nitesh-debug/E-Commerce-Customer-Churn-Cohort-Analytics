"""
app.py
Production-Ready FastAPI Service for E-Commerce Customer Churn Prediction.

This microservice exposes endpoints to score customer churn probability in real time
using a pre-trained Random Forest model ('churn_model.joblib').
"""
import io
import hashlib
import json
import logging
import os
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from campaign_db import init_campaigns_db, list_campaigns, save_campaign
from feature_engineering import (
    FEATURE_COLUMNS,
    CHURN_RECENCY_THRESHOLD_DAYS,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    VIP_MONETARY_THRESHOLD,
    classify_risk,
    compute_features_from_transactions,
    get_model_input_df,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_model.joblib")
UPLOAD_DB_PATH = os.path.join(os.path.dirname(__file__), "uploads.db")
MODEL_VERSION = os.getenv("CHURN_MODEL_VERSION", "churn_model.joblib")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
API_KEY = os.getenv("API_KEY", "").strip()
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
MODEL_VALIDATION_NOTE = (
    "Prediction quality is evaluated on a held-out test set. "
    "A single customer prediction does not prove model accuracy."
)
# Global model container
model_store: Dict[str, Any] = {"model": None}
uploaded_transaction_store: Dict[str, pd.DataFrame] = {"transactions": None}
rate_limit_store: Dict[str, deque] = defaultdict(deque)
logger = logging.getLogger("churn_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")

# Ensure campaign storage exists even before startup lifecycle completes.
init_campaigns_db()


def init_uploads_db() -> None:
    """Create persistent storage for the latest normalized transaction upload."""
    conn = sqlite3.connect(UPLOAD_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploaded_transactions (
            customer_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            order_date TEXT NOT NULL,
            amount REAL NOT NULL,
            quantity REAL NOT NULL,
            product_id TEXT NOT NULL,
            country TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_uploaded_transactions(frame: pd.DataFrame) -> None:
    """Replace the persisted latest upload with normalized transaction rows."""
    init_uploads_db()
    conn = sqlite3.connect(UPLOAD_DB_PATH)
    frame.to_sql("uploaded_transactions", conn, if_exists="replace", index=False)
    conn.close()


def load_uploaded_transactions() -> pd.DataFrame:
    """Load the latest upload, returning an empty frame when none exists."""
    if not os.path.exists(UPLOAD_DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(UPLOAD_DB_PATH)
    try:
        return pd.read_sql_query("SELECT * FROM uploaded_transactions", conn)
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        conn.close()


init_uploads_db()


class RuntimeProtectionMiddleware(BaseHTTPMiddleware):
    """Apply opt-in API-key protection and a lightweight per-client rate limit."""

    public_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/dashboard"}

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        request_times = rate_limit_store[client_key]
        while request_times and now - request_times[0] >= RATE_LIMIT_WINDOW_SECONDS:
            request_times.popleft()
        if request.url.path not in self.public_paths and len(request_times) >= RATE_LIMIT_REQUESTS:
            logger.warning(json.dumps({"event": "rate_limited", "path": request.url.path, "client": client_key}))
            return JSONResponse({"detail": "Rate limit exceeded."}, status_code=429)
        request_times.append(now)

        if API_KEY and request.url.path not in self.public_paths:
            supplied_key = request.headers.get("x-api-key", "")
            if supplied_key != API_KEY:
                logger.warning(json.dumps({"event": "unauthorized", "path": request.url.path, "client": client_key}))
                return JSONResponse({"detail": "API key required."}, status_code=401)

        response = await call_next(request)
        logger.info(json.dumps({
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }))
        return response


def load_model(path: str = MODEL_PATH):
    """Loads the serialized joblib machine learning model."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trained model artifact not found at '{path}'.")
    return joblib.load(path)


def model_artifact_sha256(path: str = MODEL_PATH) -> str:
    """Return a stable checksum for the loaded model artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML model and campaign state on startup."""
    try:
        model_store["model"] = load_model(MODEL_PATH)
        print(f"[OK] Churn prediction model loaded successfully from '{MODEL_PATH}'.")
    except Exception as exc:
        print(f"[ERROR] Failed to load model artifact '{MODEL_PATH}': {exc}")
        model_store["model"] = None

    try:
        init_campaigns_db()
        print("[OK] Campaign simulation database initialized.")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize campaign database: {exc}")

    persisted_upload = load_uploaded_transactions()
    if not persisted_upload.empty:
        uploaded_transaction_store["transactions"] = persisted_upload
        print("[OK] Persisted transaction upload restored.")

    yield
    model_store.clear()


# Initialize FastAPI application
app = FastAPI(
    title="E-Commerce Churn Prediction API",
    description=(
        "Production-grade REST API providing real-time customer churn probability scoring, "
        "risk tier categorization, and dynamic marketing recommendations based on transactional behavior."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS. Restrict origins in deployed environments through CORS_ORIGINS.
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RuntimeProtectionMiddleware)


# -------------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Input schema defining customer transactional and behavioral metrics.
    Accepts primary RFM features and automatically computes derived features
    if they are not provided explicitly.
    """
    frequency: int = Field(
        ...,
        gt=0,
        description="Total count of completed distinct orders.",
        json_schema_extra={"example": 4},
    )
    monetary: float = Field(
        ...,
        ge=0.0,
        description="Total lifetime spend/revenue ($) across all transactions.",
        json_schema_extra={"example": 1797.24},
    )
    recency_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days elapsed since the customer's most recent order.",
        json_schema_extra={"example": 76},
    )
    tenure: Optional[int] = Field(
        default=None,
        ge=0,
        description="Customer relationship age in days (from first purchase date).",
        json_schema_extra={"example": 359},
    )
    customer_tenure: Optional[int] = Field(
        default=None,
        ge=0,
        description="Alias for tenure in days.",
        json_schema_extra={"example": 359},
    )
    purchase_velocity: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Purchase velocity: order frequency per day of tenure (Frequency / (Tenure + 1)).",
        json_schema_extra={"example": 0.0111},
    )
    avg_order_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average order monetary value (Monetary / Frequency).",
        json_schema_extra={"example": 449.31},
    )
    total_units: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total physical quantity of items purchased across all orders.",
        json_schema_extra={"example": 2341},
    )
    avg_basket_size: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average units per order (TotalUnits / Frequency).",
        json_schema_extra={"example": 585.25},
    )
    distinct_products: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of distinct product SKUs purchased.",
        json_schema_extra={"example": 22},
    )
    is_uk: Optional[int] = Field(
        default=1,
        ge=0,
        le=1,
        description="Geographic indicator: 1 if customer resides in the UK, 0 otherwise.",
        json_schema_extra={"example": 0},
    )

    @field_validator("frequency")
    @classmethod
    def validate_frequency(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Frequency must be greater than 0.")
        return v

    @field_validator("monetary")
    @classmethod
    def validate_monetary(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Monetary spend cannot be negative.")
        return v

    def to_model_input(self) -> pd.DataFrame:
        """
        Derives all necessary model features and returns a single-row DataFrame
        with exact column order and naming expected by the trained Random Forest classifier.
        """
        # Resolve tenure from direct field, alias, or recency fallback
        effective_tenure = (
            self.tenure
            if self.tenure is not None
            else (
                self.customer_tenure
                if self.customer_tenure is not None
                else (self.recency_days if self.recency_days is not None else 30)
            )
        )

        # Derived AvgOrderValue
        effective_aov = (
            self.avg_order_value
            if self.avg_order_value is not None
            else round(self.monetary / max(self.frequency, 1), 2)
        )

        # Derived PurchaseVelocity
        effective_velocity = (
            self.purchase_velocity
            if self.purchase_velocity is not None
            else round(self.frequency / (effective_tenure + 1), 4)
        )

        # Derived TotalUnits and AvgBasketSize
        if self.avg_basket_size is not None and self.total_units is not None:
            effective_basket_size = self.avg_basket_size
            effective_total_units = self.total_units
        elif self.avg_basket_size is not None:
            effective_basket_size = self.avg_basket_size
            effective_total_units = int(self.avg_basket_size * self.frequency)
        elif self.total_units is not None:
            effective_total_units = self.total_units
            effective_basket_size = round(self.total_units / max(self.frequency, 1), 2)
        else:
            # Domain heuristic estimate: ~$20 per unit or 10 units per order
            effective_basket_size = max(1.0, round(effective_aov / 20.0, 2))
            effective_total_units = int(effective_basket_size * self.frequency)

        # Derived DistinctProducts
        if self.distinct_products is not None:
            effective_distinct = self.distinct_products
        else:
            effective_distinct = max(1, min(effective_total_units, int(self.frequency * 6)))

        # Geographic flag
        effective_is_uk = self.is_uk if self.is_uk is not None else 1

        data = {
            "Frequency": [self.frequency],
            "Monetary": [self.monetary],
            "AvgOrderValue": [effective_aov],
            "TotalUnits": [effective_total_units],
            "AvgBasketSize": [effective_basket_size],
            "DistinctProducts": [effective_distinct],
            "CustomerTenure": [effective_tenure],
            "PurchaseVelocity": [effective_velocity],
            "Is_UK": [effective_is_uk],
        }
        return pd.DataFrame(data, columns=FEATURE_COLUMNS)


class PredictionResponse(BaseModel):
    """Structured response schema returned by the /predict endpoint."""
    churn_probability: float = Field(
        ...,
        description="Predicted probability of customer churning (0.0000 to 1.0000).",
        json_schema_extra={"example": 0.4532},
    )
    risk_level: str = Field(
        ...,
        description="Customer churn risk tier: 'High Risk', 'Medium Risk', or 'Low Risk'.",
        json_schema_extra={"example": "Medium Risk"},
    )
    recommended_action: str = Field(
        ...,
        description="Dynamic, rule-based marketing and retention strategy tailored to the customer's risk profile.",
        json_schema_extra={
            "example": "Proactive Nurture: Deploy targeted engagement campaign with personalized catalog highlights."
        },
    )
    predicted_churn: int = Field(
        ...,
        description="Binary classification prediction: 1 (Likely to Churn) or 0 (Active Customer).",
        json_schema_extra={"example": 0},
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_artifact: str = Field(..., json_schema_extra={"example": "churn_model.joblib"})
    feature_count: int = Field(..., json_schema_extra={"example": 9})
    features: List[str] = Field(..., json_schema_extra={"example": FEATURE_COLUMNS})
    model_version: str = Field(..., json_schema_extra={"example": "churn_model.joblib"})
    model_sha256: str = Field(..., json_schema_extra={"example": " sha256 checksum"})


# -------------------------------------------------------------------------
# Dynamic Marketing Recommendation Logic
# -------------------------------------------------------------------------

def determine_risk_and_action(churn_prob: float, monetary: float) -> tuple[str, str]:
    """
    Categorizes churn risk into defined business tiers and produces actionable
    marketing/retention interventions based on risk level and customer monetary profile.

        Tiers align with pipeline standards:
            - High Risk:   prob >= HIGH_RISK_THRESHOLD
            - Medium Risk: MEDIUM_RISK_THRESHOLD <= prob < HIGH_RISK_THRESHOLD
            - Low Risk:    prob < MEDIUM_RISK_THRESHOLD
    """
    if churn_prob >= HIGH_RISK_THRESHOLD:
        risk_level = "High Risk"
        if monetary >= VIP_MONETARY_THRESHOLD:
            recommended_action = (
                "VIP Win-Back Campaign: High lifetime value customer at immediate risk of loss. "
                "Trigger immediate outreach by senior account manager with an exclusive 20% loyalty reactivation "
                "credit and personalized consultation."
            )
        else:
            recommended_action = (
                "Urgent Win-Back: Customer shows severe disengagement. Deploy automated 3-step re-activation "
                "email workflow offering a 15% discount voucher and free express shipping."
            )
    elif churn_prob >= MEDIUM_RISK_THRESHOLD:
        risk_level = "Medium Risk"
        recommended_action = (
            "Proactive Retention & Nurture: Customer activity is decelerating. Trigger personalized email "
            "campaign with replenishment reminders, curated category recommendations, and double reward points "
            "on the next purchase."
        )
    else:
        risk_level = "Low Risk"
        recommended_action = (
            "Loyalty Expansion & Upsell: Customer exhibits high engagement and purchase velocity. "
            "Enroll in the VIP Champions Tier, offer early access to new catalog arrivals, and provide "
            "referral incentives."
        )
    return risk_level, recommended_action


def _ensure_model_loaded():
    model = model_store.get("model")
    if model is None:
        try:
            model = load_model(MODEL_PATH)
            model_store["model"] = model
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifact could not be loaded: {exc}",
            )
    return model


def _normalize_uploaded_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["customer_id", "order_date", "amount", "quantity"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Missing required columns: {missing_columns}",
        )

    normalized = df.copy()
    normalized["customer_id"] = normalized["customer_id"].astype(str)
    if "order_id" not in normalized.columns:
        normalized["order_id"] = normalized.index.astype(str)
    default_order_ids = pd.Series(normalized.index.astype(str), index=normalized.index)
    normalized["order_id"] = normalized["order_id"].fillna(default_order_ids).astype(str)
    if "product_id" not in normalized.columns:
        normalized["product_id"] = normalized["order_id"]
    normalized["product_id"] = normalized["product_id"].fillna("UNKNOWN").astype(str)
    if "country" not in normalized.columns:
        normalized["country"] = "Unknown"
    normalized["country"] = normalized["country"].fillna("Unknown").astype(str)

    normalized["order_date"] = pd.to_datetime(normalized["order_date"], errors="coerce")
    if normalized["order_date"].isna().any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="One or more order_date values are invalid or missing.",
        )

    normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce")
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")

    if normalized["amount"].isna().any() or normalized["quantity"].isna().any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Amount and quantity columns must contain numeric values.",
        )

    if (normalized["amount"] <= 0).any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Transaction amount must be greater than zero for all rows.",
        )

    if (normalized["quantity"] <= 0).any():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Transaction quantity must be greater than zero for all rows.",
        )

    normalized = normalized[["customer_id", "order_id", "order_date", "amount", "quantity", "product_id", "country"]].copy()
    normalized["order_date"] = normalized["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    normalized["amount"] = normalized["amount"].round(2)
    return normalized


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only CSV uploads are supported.",
        )

    try:
        raw_content = file.file.read()
        if not raw_content:
            raise ValueError("Uploaded CSV file is empty.")
        if len(raw_content) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Uploaded CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB size limit."
            )
        df = pd.read_csv(io.StringIO(raw_content.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unable to read uploaded CSV file: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded CSV file contains no rows.",
        )

    return _normalize_uploaded_transactions(df)


def _coerce_customer_id(value: Any) -> str:
    return str(value).strip()


def _get_customer_records(customer_id: Optional[str] = None) -> pd.DataFrame:
    current_upload = uploaded_transaction_store.get("transactions")
    if current_upload is None:
        current_upload = load_uploaded_transactions()
        if not current_upload.empty:
            uploaded_transaction_store["transactions"] = current_upload
    if current_upload is not None and not current_upload.empty:
        df = current_upload.copy()
        if customer_id is not None:
            return df[df["customer_id"].astype(str) == str(customer_id)].copy()
        return df

    analytics_db_path = os.path.join(os.path.dirname(__file__), "analytics.db")
    if not os.path.exists(analytics_db_path):
        return pd.DataFrame(columns=["customer_id", "order_id", "order_date", "amount", "quantity", "product_id", "country"])

    conn = sqlite3.connect(analytics_db_path)
    query = "SELECT CAST(CustomerID AS TEXT) AS customer_id, CAST(InvoiceNo AS TEXT) AS order_id, InvoiceDate AS order_date, TotalAmount AS amount, Quantity AS quantity, CAST(StockCode AS TEXT) AS product_id, Country AS country FROM transactions"
    query_params = []
    if customer_id is not None:
        query += " WHERE CAST(CustomerID AS TEXT) = ?"
        query_params.append(str(customer_id))
    df = pd.read_sql_query(query, conn, params=query_params)
    conn.close()
    return df


def _rfm_segment_for_customer(churn_probability: float, monetary: float, frequency: int, recency_days: int) -> str:
    if churn_probability >= HIGH_RISK_THRESHOLD and monetary >= VIP_MONETARY_THRESHOLD:
        return "Champions"
    if churn_probability >= HIGH_RISK_THRESHOLD:
        return "At-Risk"
    if churn_probability >= MEDIUM_RISK_THRESHOLD:
        return "Loyal Customers"
    if monetary >= 5000 or frequency >= 10:
        return "Champions"
    if recency_days > 120:
        return "At-Risk"
    if frequency <= 2 and monetary < 500:
        return "Promising"
    return "Loyal Customers"


def _build_customer_analysis(customer_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' was not found.",
        )

    customer_df = df.copy()
    customer_df["amount"] = pd.to_numeric(customer_df["amount"], errors="coerce")
    customer_df["quantity"] = pd.to_numeric(customer_df["quantity"], errors="coerce")
    customer_df["order_date"] = pd.to_datetime(customer_df["order_date"], errors="coerce")
    customer_df = customer_df.dropna(subset=["amount", "quantity", "order_date"]).copy()

    if customer_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer '{customer_id}' has no valid transactions.",
        )

    customer_features = compute_features_from_transactions(customer_df)
    customer_row = customer_features.iloc[0]
    model_input = get_model_input_df(customer_row)
    model = _ensure_model_loaded()
    probabilities = model.predict_proba(model_input)
    churn_probability = round(float(probabilities[0, 1]), 4)
    predicted_churn = int(model.predict(model_input)[0])

    risk_level, _campaign_type, _offer, recommended_action = classify_risk(
        churn_probability, float(customer_row["Monetary"])
    )

    rfm_segment = _rfm_segment_for_customer(
        churn_probability,
        float(customer_row["Monetary"]),
        int(customer_row["Frequency"]),
        int(customer_row["Recency"]),
    )

    return {
        "customer_id": customer_id,
        "total_orders": int(customer_row["Frequency"]),
        "total_spend": round(float(customer_row["Monetary"]), 2),
        "tenure": int(customer_row["CustomerTenure"]),
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "rfm_segment": rfm_segment,
        "recommended_action": recommended_action,
        "predicted_churn": predicted_churn,
        "status": "OK",
    }


def _predict_from_transaction_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame.empty:
        return []

    model = _ensure_model_loaded()
    feature_rows = compute_features_from_transactions(frame)
    predictions = []

    for _, row in feature_rows.iterrows():
        customer_id = str(row["CustomerID"])
        model_input = get_model_input_df(row)
        probability = round(float(model.predict_proba(model_input)[0, 1]), 4)
        predicted_class = int(model.predict(model_input)[0])
        risk_level, _campaign_type, _offer, recommended_action = classify_risk(probability, float(row["Monetary"]))
        rfm_segment = _rfm_segment_for_customer(
            probability,
            float(row["Monetary"]),
            int(row["Frequency"]),
            int(row["Recency"]),
        )
        predictions.append({
            "customer_id": customer_id,
            "churn_probability": probability,
            "predicted_churn": predicted_class,
            "risk_level": risk_level,
            "rfm_segment": rfm_segment,
            "monetary_value": round(float(row["Monetary"]), 2),
            "recommended_action": recommended_action,
        })

    return predictions


def _build_priority_candidates(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame.empty:
        return []

    predictions = _predict_from_transaction_frame(frame)
    for item in predictions:
        item["priority_score"] = round(
            item["churn_probability"] * 0.6
            + (1.0 if item["risk_level"] == "High Risk" else 0.4 if item["risk_level"] == "Medium Risk" else 0.1) * 0.3
            + min(float(item["monetary_value"]) / 5000.0, 1.0) * 0.1,
            4,
        )

    return sorted(predictions, key=lambda x: (-x["priority_score"], -float(x["monetary_value"]), x["customer_id"]))


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/", summary="Root Welcome Endpoint", tags=["General"])
def root():
    """Welcome endpoint providing metadata and API navigation links."""
    return {
        "title": "E-Commerce Customer Churn Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "dashboard_url": "/dashboard",
        "model_validation_note": MODEL_VALIDATION_NOTE,
        "model_version": MODEL_VERSION,
        "configuration": {
            "churn_recency_threshold_days": CHURN_RECENCY_THRESHOLD_DAYS,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
            "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
            "vip_monetary_threshold": VIP_MONETARY_THRESHOLD,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        },
    }


@app.get("/dashboard", summary="Executive Analytics Dashboard", tags=["General"])
def get_dashboard():
    """Serves the interactive Executive Analytics Dashboard HTML."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard file not found. Run generate_dashboard.py first.",
        )
    return FileResponse(dashboard_path, media_type="text/html")


@app.get("/customers", summary="List available customer IDs", tags=["Customer Analysis"])
def list_customers():
    """Return distinct customer IDs currently available in the uploaded transactions or SQLite database."""
    df = _get_customer_records()
    if df.empty:
        return {"customers": []}
    customer_ids = sorted(df["customer_id"].astype(str).unique().tolist())
    return {"customers": customer_ids, "count": len(customer_ids)}


@app.get("/customers/{customer_id}", summary="Analyze a specific customer", tags=["Customer Analysis"])
def get_customer_analysis(customer_id: str):
    """Analyze one customer using the real model and their stored transactions."""
    customer_records = _get_customer_records(customer_id)
    return _build_customer_analysis(_coerce_customer_id(customer_id), customer_records)


@app.post("/predict/customer/{customer_id}", summary="Predict churn for a single customer", tags=["Customer Analysis"])
def predict_customer(customer_id: str):
    """Predict churn probability for a specific customer ID using the real trained model."""
    customer_records = _get_customer_records(customer_id)
    analysis = _build_customer_analysis(_coerce_customer_id(customer_id), customer_records)
    return {
        "customer_id": analysis["customer_id"],
        "churn_probability": analysis["churn_probability"],
        "risk_level": analysis["risk_level"],
        "predicted_churn": analysis["predicted_churn"],
        "rfm_segment": analysis["rfm_segment"],
        "recommended_action": analysis["recommended_action"],
    }


@app.post("/predict/bulk", summary="Bulk customer churn prediction from CSV upload", tags=["Bulk Prediction"])
async def bulk_predict(file: UploadFile = File(...)):
    """Generate churn predictions for all customers in a raw transaction CSV and return model outputs."""
    df = _read_csv_upload(file)
    uploaded_transaction_store["transactions"] = df
    save_uploaded_transactions(df)
    rows = _predict_from_transaction_frame(df)
    return {"rows": rows, "count": len(rows)}


@app.post("/upload", summary="Upload raw transaction data", tags=["Data Upload"])
async def upload_transactions(file: UploadFile = File(...)):
    """Accept a transaction CSV with a business-friendly schema and return feature rows for all customers."""
    df = _read_csv_upload(file)
    uploaded_transaction_store["transactions"] = df
    save_uploaded_transactions(df)
    feature_rows = compute_features_from_transactions(df)
    customers_processed = int(feature_rows["CustomerID"].nunique()) if not feature_rows.empty else 0
    return {
        "message": "Transaction data validated and customer features generated successfully.",
        "customers_processed": customers_processed,
        "feature_rows": len(feature_rows),
        "columns": list(feature_rows.columns),
    }


@app.post("/campaign/generate", summary="Generate simulated retention campaign", tags=["Campaigns"])
def generate_campaign(payload: Dict[str, Any]):
    """Create a simulated retention action record without pretending a real external provider is used."""
    customer_id = str(payload.get("customer_id", "")).strip()
    risk_level = str(payload.get("risk_level", "Low Risk")).strip()
    campaign_type = str(payload.get("campaign_type", "Re-Engagement")).strip()
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="customer_id is required.",
        )

    offer = payload.get("offer") or "Standard retention offer"
    message = payload.get("message") or "We value your business and want to keep you engaged."
    rfm_segment = payload.get("rfm_segment") or "Not available"
    monetary_value = payload.get("monetary_value")
    churn_probability = payload.get("churn_probability")
    created_at = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "customer_id": customer_id,
        "risk_level": risk_level,
        "rfm_segment": rfm_segment,
        "monetary_value": float(monetary_value) if monetary_value is not None else None,
        "churn_probability": float(churn_probability) if churn_probability is not None else None,
        "campaign_type": campaign_type,
        "offer": offer,
        "message": message,
        "status": "SIMULATED",
        "model_version": MODEL_VERSION,
        "created_at": created_at,
    }
    try:
        save_campaign(record)
    except sqlite3.OperationalError:
        init_campaigns_db()
        save_campaign(record)
    return {
        "customer_id": customer_id,
        "risk_level": risk_level,
        "rfm_segment": rfm_segment,
        "campaign_type": campaign_type,
        "offer": offer,
        "message": message,
        "status": "SIMULATED",
        "model_version": MODEL_VERSION,
        "timestamp": created_at,
    }


@app.get("/campaigns", summary="List simulated campaigns", tags=["Campaigns"])
def get_campaigns():
    """Return the stored simulated campaign history."""
    return {"campaigns": list_campaigns(limit=200)}


@app.get("/priority", summary="Retention priority view", tags=["Business Prioritization"])
def get_priority_candidates():
    """Return prioritized retention candidates using churn probability, customer value, and RFM segment."""
    tx_df = uploaded_transaction_store.get("transactions")
    if tx_df is None:
        tx_df = _get_customer_records()
    if tx_df.empty:
        return {"candidates": []}
    candidates = _build_priority_candidates(tx_df)
    return {
        "candidates": candidates,
        "count": len(candidates),
        "note": "Model/business-rule based recommendations only; actual CRM or messaging execution requires external integration.",
    }


@app.get("/model/status", summary="Model governance status", tags=["System"])
def model_status():
    """Return model identity, checksum, feature contract, and active thresholds."""
    return {
        "model_loaded": model_store.get("model") is not None or os.path.exists(MODEL_PATH),
        "model_version": MODEL_VERSION,
        "model_sha256": model_artifact_sha256(),
        "feature_columns": FEATURE_COLUMNS,
        "churn_recency_threshold_days": CHURN_RECENCY_THRESHOLD_DAYS,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "vip_monetary_threshold": VIP_MONETARY_THRESHOLD,
    }


@app.get("/health", response_model=HealthResponse, summary="API Health Check", tags=["System"])
def health_check():
    """
    Verifies that the API service is active and the pre-trained ML model
    artifact is successfully loaded in memory.
    """
    model = model_store.get("model")
    if model is None:
        # Fallback attempt to load model if not yet loaded
        try:
            model = load_model(MODEL_PATH)
            model_store["model"] = model
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Churn prediction model is not loaded: {exc}",
            )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_artifact="churn_model.joblib",
        feature_count=len(FEATURE_COLUMNS),
        features=FEATURE_COLUMNS,
        model_version=MODEL_VERSION,
        model_sha256=model_artifact_sha256(),
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict Customer Churn", tags=["Inference"])
def predict_churn(features: CustomerFeatures):
    """
    Executes real-time inference on customer transactional behavior to compute:
    1. **churn_probability**: Calibrated probability (0.0 to 1.0) rounded to 4 decimals.
    2. **risk_level**: Categorization into 'High Risk', 'Medium Risk', or 'Low Risk'.
    3. **recommended_action**: Actionable CRM and marketing intervention strategy.
    4. **predicted_churn**: Binary label (1=Churn, 0=Active).
    """
    model = model_store.get("model")
    if model is None:
        # Fallback load attempt
        try:
            model = load_model(MODEL_PATH)
            model_store["model"] = model
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifact could not be loaded: {exc}",
            )

    try:
        # Convert input features to DataFrame with expected columns
        input_df = features.to_model_input()

        # Run inference through Random Forest model
        probabilities = model.predict_proba(input_df)
        churn_probability = round(float(probabilities[0, 1]), 4)
        predicted_class = int(model.predict(input_df)[0])

        # Derive business tier and actionable CRM intervention
        risk_level, recommended_action = determine_risk_and_action(
            churn_prob=churn_probability,
            monetary=features.monetary,
        )

        return PredictionResponse(
            churn_probability=churn_probability,
            risk_level=risk_level,
            recommended_action=recommended_action,
            predicted_churn=predicted_class,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing model inference: {str(exc)}",
        )
