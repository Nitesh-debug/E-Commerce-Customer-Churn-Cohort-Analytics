"""
train_churn_model.py
Stage 3: Machine Learning Churn Prediction Pipeline

This script:
1. Extracts customer-level behavioral and transactional features from 'analytics.db'.
2. Defines churn target 'Is_Churned' = 1 if inactive >90 days, else 0.
3. Splits data into stratified 80/20 train/test sets.
4. Trains a tuned RandomForestClassifier to predict customer churn probability.
5. Evaluates model performance: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
6. Computes feature importance rankings.
7. Saves:
   - 'churn_model.joblib'
   - 'output/model_evaluation_metrics.json'
   - 'output/feature_importance.csv'
   - 'output/customer_churn_predictions.csv'
"""

import json
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DB_PATH = "analytics.db"
MODEL_OUTPUT_PATH = "churn_model.joblib"
OUTPUT_DIR = "output"


def build_feature_dataset(conn):
    """
    Extracts customer behavioral features without target leakage.
    Snapshot date is dynamic: 1 day past the latest transaction.
    """
    query = """
    WITH snapshot AS (
        SELECT datetime(MAX(InvoiceDate), '+1 day') AS snapshot_date
        FROM transactions
    )
    SELECT
        t.CustomerID,
        t.Country,
        COUNT(DISTINCT t.InvoiceNo) AS Frequency,
        ROUND(SUM(t.TotalAmount), 2) AS Monetary,
        SUM(t.Quantity) AS TotalUnits,
        COUNT(DISTINCT t.StockCode) AS DistinctProducts,
        -- Recency is used strictly to define the target variable
        CAST(ROUND(julianday(s.snapshot_date) - julianday(MAX(t.InvoiceDate))) AS INTEGER) AS Recency,
        -- CustomerTenure: Days since customer's first purchase
        CAST(ROUND(julianday(s.snapshot_date) - julianday(MIN(t.InvoiceDate))) AS INTEGER) AS CustomerTenure
    FROM transactions t
    CROSS JOIN snapshot s
    GROUP BY t.CustomerID;
    """
    df = pd.read_sql_query(query, conn)

    # Derived engineered features
    df["AvgOrderValue"] = (df["Monetary"] / df["Frequency"]).round(2)
    df["AvgBasketSize"] = (df["TotalUnits"] / df["Frequency"]).round(2)
    df["PurchaseVelocity"] = (df["Frequency"] / (df["CustomerTenure"] + 1)).round(4)
    df["Is_UK"] = (df["Country"] == "United Kingdom").astype(int)

    # Target variable definition: 1 if inactive > 90 days, else 0
    df["Is_Churned"] = (df["Recency"] > 90).astype(int)

    return df


def train_and_evaluate():
    print("=" * 60)
    print("STAGE 3: MACHINE LEARNING CHURN PREDICTION PIPELINE")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database '{DB_PATH}' not found. Run clean_data.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[*] Extracting customer features from '{DB_PATH}'...")
    conn = sqlite3.connect(DB_PATH)
    data = build_feature_dataset(conn)
    conn.close()

    print(f"    [OK] Extracted features for {len(data):,} customers.")
    churn_count = data["Is_Churned"].sum()
    churn_rate = churn_count / len(data)
    print(f"    - Active Customers (Recency <= 90 days): {len(data) - churn_count:,} ({1 - churn_rate:.1%})")
    print(f"    - Churned Customers (Recency > 90 days):  {churn_count:,} ({churn_rate:.1%})")

    # Feature selection
    feature_cols = [
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

    X = data[feature_cols]
    y = data["Is_Churned"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[*] Train/Test Split: {len(X_train):,} train rows, {len(X_test):,} test rows (Stratified).")

    # Model definition & training
    print("[*] Training RandomForestClassifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    print("    [OK] Model training complete.")

    # Predictions & Probabilities
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "-" * 40)
    print("MODEL PERFORMANCE EVALUATION METRICS")
    print("-" * 40)
    print(f"  Accuracy:       {accuracy:.4f} ({accuracy:.1%})")
    print(f"  Precision:      {precision:.4f} ({precision:.1%})")
    print(f"  Recall:         {recall:.4f} ({recall:.1%})")
    print(f"  F1-Score:       {f1:.4f}")
    print(f"  ROC-AUC Score:  {roc_auc:.4f}")
    print("-" * 40)
    print("Confusion Matrix (Test Set):")
    print(f"  [TN: {cm[0,0]:3d}  |  FP: {cm[0,1]:3d}]")
    print(f"  [FN: {cm[1,0]:3d}  |  TP: {cm[1,1]:3d}]")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Active (0)", "Churned (1)"]))

    # Feature Importance
    feat_importance_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)
    feat_importance_df["Importance_Rank"] = range(1, len(feat_importance_df) + 1)

    print("[+] Feature Importances:")
    for _, row in feat_importance_df.iterrows():
        bar = "#" * int(row["Importance"] * 40)
        print(f"  {row['Importance_Rank']:2d}. {row['Feature']:18s} : {row['Importance']:.4f} | {bar}")

    # Save artifacts
    print(f"\n[*] Saving trained model to '{MODEL_OUTPUT_PATH}'...")
    joblib.dump(rf_model, MODEL_OUTPUT_PATH)
    print(f"    [OK] Model successfully serialized to '{MODEL_OUTPUT_PATH}'.")

    # Save metrics json
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_cols),
    }
    metrics_path = os.path.join(OUTPUT_DIR, "model_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"    [OK] Metrics saved to '{metrics_path}'.")

    # Save feature importances CSV
    feat_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feat_importance_df.to_csv(feat_path, index=False)
    print(f"    [OK] Feature importances saved to '{feat_path}'.")

    # Generate predictions & churn probabilities for the entire customer base
    data["Churn_Probability"] = rf_model.predict_proba(data[feature_cols])[:, 1].round(4)
    data["Predicted_Churn"] = rf_model.predict(data[feature_cols])
    data["Risk_Level"] = pd.cut(
        data["Churn_Probability"],
        bins=[0.0, 0.3, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )
    predictions_path = os.path.join(OUTPUT_DIR, "customer_churn_predictions.csv")
    data.to_csv(predictions_path, index=False)
    print(f"    [OK] Full customer risk scoring saved to '{predictions_path}'.")

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
