"""
feature_engineering.py
Shared Feature Engineering Module

Single source of truth for computing customer-level ML features from raw
transaction data. This module is used by:
  - routers/customers.py  (customer lookup & prediction)
  - routers/upload.py     (CSV upload analysis)
  - routers/bulk.py       (bulk prediction)

All features are computed identically to train_churn_model.py::build_feature_dataset(),
ensuring zero drift between training and inference.

CHURN DEFINITION: Is_Churned = 1 if Recency > 90 days (relative to snapshot_date).
TARGET LEAKAGE NOTE: Recency is computed here only to determine Is_Churned.
Recency is NEVER included in the model feature vector.

MODEL FEATURE ORDER (must match churn_model.joblib training order):
  ["Frequency", "Monetary", "AvgOrderValue", "TotalUnits", "AvgBasketSize",
   "DistinctProducts", "CustomerTenure", "PurchaseVelocity", "Is_UK"]
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Model feature column order — must never change without retraining the model
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "TotalUnits",
    "AvgBasketSize",
    "DistinctProducts",
    "CustomerTenure",
    "PurchaseVelocity",
    "Is_UK",
]

# Churn definition threshold in days
CHURN_RECENCY_THRESHOLD_DAYS = 90

# Risk tier thresholds (configurable — change here to affect all endpoints)
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

# Campaign value threshold for VIP treatment
VIP_MONETARY_THRESHOLD = 1000.0


# ---------------------------------------------------------------------------
# Accepted upload column names (user-facing data contract)
# ---------------------------------------------------------------------------
#
# Business users may upload CSV files with these column names.
# They do NOT need to match Online Retail column names exactly.
#
UPLOAD_SCHEMA = {
    "required": {
        "customer_id": "Unique identifier for the customer (integer or string).",
        "order_date": "Date of the transaction (ISO 8601: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).",
        "amount": "Monetary value of the transaction line (positive float, e.g., 29.99).",
        "quantity": "Number of units purchased in this line item (positive integer).",
    },
    "optional": {
        "order_id": "Order/invoice identifier. Used to count distinct orders. If absent, each row is counted as a separate order.",
        "product_id": "Product/SKU identifier. Used to count distinct products purchased. If absent, distinct products defaults to order count.",
        "country": "Customer's country. Used to compute Is_UK flag. Defaults to non-UK (0) if absent.",
    },
}

# Mapping from Online Retail column names to upload schema names (for documentation)
ONLINE_RETAIL_COLUMN_MAP = {
    "InvoiceNo":    "order_id",
    "StockCode":    "product_id",
    "Quantity":     "quantity",
    "InvoiceDate":  "order_date",
    "UnitPrice":    "(amount = Quantity × UnitPrice)",
    "CustomerID":   "customer_id",
    "Country":      "country",
}


# ---------------------------------------------------------------------------
# Feature derivation: raw transaction DataFrame → model-ready feature row(s)
# ---------------------------------------------------------------------------

def compute_features_from_transactions(
    df: pd.DataFrame,
    snapshot_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Compute customer-level ML features from a raw transaction DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: customer_id, order_date, amount, quantity.
        Optional columns: order_id, product_id, country.
    snapshot_date : datetime, optional
        Reference date for Recency/Tenure calculations.
        Defaults to MAX(order_date) + 1 day (matches training behaviour).

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        CustomerID, Country, Frequency, Monetary, TotalUnits, DistinctProducts,
        CustomerTenure, Recency, AvgOrderValue, AvgBasketSize, PurchaseVelocity,
        Is_UK, Is_Churned
        plus all FEATURE_COLUMNS in the correct model order.
    """
    df = df.copy()

    # Normalise order_date
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    if snapshot_date is None:
        snapshot_date = df["order_date"].max() + timedelta(days=1)

    # Optional columns with defaults
    has_order_id = "order_id" in df.columns
    has_product_id = "product_id" in df.columns
    has_country = "country" in df.columns

    # Build aggregation
    agg = (
        df.groupby("customer_id")
        .agg(
            FirstPurchaseDate=("order_date", "min"),
            LastPurchaseDate=("order_date", "max"),
            Monetary=("amount", "sum"),
            TotalUnits=("quantity", "sum"),
            **({
                "Frequency": ("order_id", "nunique"),
                "DistinctProducts": ("product_id", "nunique"),
            } if (has_order_id and has_product_id) else
            {
                "Frequency": ("order_id", "nunique"),
                "DistinctProducts": ("order_date", "count"),  # fallback
            } if has_order_id else
            {
                "Frequency": ("order_date", "count"),  # each row = 1 order
                "DistinctProducts": (
                    "product_id", "nunique"
                ) if has_product_id else ("order_date", "count"),
            }),
        )
        .reset_index()
        .rename(columns={"customer_id": "CustomerID"})
    )

    agg["Monetary"] = agg["Monetary"].round(2)

    # Country
    if has_country:
        country_map = (
            df.groupby("customer_id")["country"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown")
            .reset_index()
            .rename(columns={"customer_id": "CustomerID", "country": "Country"})
        )
        agg = agg.merge(country_map, on="CustomerID", how="left")
    else:
        agg["Country"] = "Unknown"

    # Temporal features
    agg["CustomerTenure"] = (
        (snapshot_date - agg["FirstPurchaseDate"]).dt.days
    ).clip(lower=0).astype(int)
    agg["Recency"] = (
        (snapshot_date - agg["LastPurchaseDate"]).dt.days
    ).clip(lower=0).astype(int)

    # Derived features
    agg["AvgOrderValue"] = (agg["Monetary"] / agg["Frequency"].clip(lower=1)).round(2)
    agg["AvgBasketSize"] = (agg["TotalUnits"] / agg["Frequency"].clip(lower=1)).round(2)
    agg["PurchaseVelocity"] = (
        agg["Frequency"] / (agg["CustomerTenure"] + 1)
    ).round(4)
    agg["Is_UK"] = (
        agg["Country"].str.strip().str.lower() == "united kingdom"
    ).astype(int)

    # Target (for reference; never used as model input feature)
    agg["Is_Churned"] = (agg["Recency"] > CHURN_RECENCY_THRESHOLD_DAYS).astype(int)

    return agg


def get_model_input_df(customer_row: pd.Series) -> pd.DataFrame:
    """
    Convert a single customer aggregated row into the exact feature DataFrame
    expected by churn_model.joblib.

    The column order matches FEATURE_COLUMNS exactly.
    """
    row = {col: [customer_row[col]] for col in FEATURE_COLUMNS}
    return pd.DataFrame(row, columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Feature engineering from SQLite (primary analytics.db path)
# ---------------------------------------------------------------------------

def compute_features_from_db(
    conn: sqlite3.Connection,
    customer_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute customer features directly from analytics.db using the exact same
    SQL logic as train_churn_model.py to ensure zero drift.

    Parameters
    ----------
    conn : sqlite3.Connection
    customer_id : int, optional
        If provided, returns features for a single customer.
        If None, returns features for all customers.

    Returns
    -------
    pd.DataFrame  with all FEATURE_COLUMNS plus CustomerID, Country, Recency, Is_Churned.
    """
    where_clause = f"AND t.CustomerID = {customer_id}" if customer_id else ""

    query = f"""
    WITH snapshot AS (
        SELECT datetime(MAX(InvoiceDate), '+1 day') AS snapshot_date
        FROM transactions
    )
    SELECT
        t.CustomerID,
        t.Country,
        COUNT(DISTINCT t.InvoiceNo)                                              AS Frequency,
        ROUND(SUM(t.TotalAmount), 2)                                             AS Monetary,
        SUM(t.Quantity)                                                          AS TotalUnits,
        COUNT(DISTINCT t.StockCode)                                              AS DistinctProducts,
        CAST(ROUND(julianday(s.snapshot_date) - julianday(MAX(t.InvoiceDate))) AS INTEGER) AS Recency,
        CAST(ROUND(julianday(s.snapshot_date) - julianday(MIN(t.InvoiceDate))) AS INTEGER) AS CustomerTenure
    FROM transactions t
    CROSS JOIN snapshot s
    WHERE 1=1 {where_clause}
    GROUP BY t.CustomerID;
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        return df

    df["AvgOrderValue"]   = (df["Monetary"] / df["Frequency"].clip(lower=1)).round(2)
    df["AvgBasketSize"]   = (df["TotalUnits"] / df["Frequency"].clip(lower=1)).round(2)
    df["PurchaseVelocity"] = (df["Frequency"] / (df["CustomerTenure"] + 1)).round(4)
    df["Is_UK"]           = (df["Country"] == "United Kingdom").astype(int)
    df["Is_Churned"]      = (df["Recency"] > CHURN_RECENCY_THRESHOLD_DAYS).astype(int)

    return df


# ---------------------------------------------------------------------------
# Risk classification (configurable thresholds)
# ---------------------------------------------------------------------------

def classify_risk(churn_probability: float, monetary: float) -> tuple[str, str, str, str]:
    """
    Classify churn risk and generate campaign recommendation.

    Returns
    -------
    (risk_level, campaign_type, offer, recommended_action)
    """
    if churn_probability >= HIGH_RISK_THRESHOLD:
        risk_level = "High Risk"
        if monetary >= VIP_MONETARY_THRESHOLD:
            campaign_type = "VIP Win-Back"
            offer = "Exclusive 20% loyalty reactivation credit + VIP account manager outreach"
            action = (
                "VIP Win-Back Campaign [Model-Based Recommendation]: High lifetime value "
                "customer at immediate risk of loss. Trigger senior account manager outreach "
                "with exclusive 20% loyalty reactivation credit and personalised consultation."
            )
        else:
            campaign_type = "Urgent Win-Back"
            offer = "15% discount voucher + free express shipping"
            action = (
                "Urgent Win-Back [Model-Based Recommendation]: Customer shows severe "
                "disengagement. Deploy 3-step re-activation email with 15% discount "
                "voucher and free express shipping."
            )
    elif churn_probability >= MEDIUM_RISK_THRESHOLD:
        risk_level = "Medium Risk"
        campaign_type = "Re-Engagement"
        offer = "Personalised replenishment reminders + double reward points"
        action = (
            "Re-Engagement Campaign [Model-Based Recommendation]: Customer activity is "
            "decelerating. Trigger personalised email with replenishment reminders, "
            "curated category recommendations, and double reward points on next purchase."
        )
    else:
        risk_level = "Low Risk"
        campaign_type = "Loyalty & Upsell"
        offer = "Early access to new arrivals + referral incentives"
        action = (
            "Loyalty Expansion & Upsell [Model-Based Recommendation]: Customer exhibits "
            "high engagement. Enroll in VIP Champions Tier, offer early access to new "
            "catalogue arrivals, and provide referral incentives."
        )

    return risk_level, campaign_type, offer, action
