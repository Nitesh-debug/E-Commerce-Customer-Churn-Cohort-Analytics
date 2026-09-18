"""
run_sql_analytics.py
Stage 2: Execute SQL RFM Segmentation & Cohort Analysis

This script:
1. Executes 'rfm_cohort_analysis.sql' against 'analytics.db'.
2. Exports RFM segmentation results to 'output/rfm_customer_segments.csv'.
3. Generates and pivots the Monthly Retention Cohort Matrix into:
   - 'output/cohort_retention_matrix.csv' (Customer counts)
   - 'output/cohort_retention_rates.csv' (Retention percentages)
4. Displays formatted summary tables of segments and cohort retention.
"""

import os
import sqlite3
import pandas as pd

DB_PATH = "analytics.db"
SQL_FILE_PATH = "rfm_cohort_analysis.sql"
OUTPUT_DIR = "output"


def run_analytics():
    print("=" * 60)
    print("STAGE 2: SQL RFM SEGMENTATION & COHORT ANALYSIS")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database '{DB_PATH}' does not exist. Run clean_data.py first.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[*] Created output directory: '{OUTPUT_DIR}/'")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Execute SQL script
    print(f"[*] Executing SQL analytics script from '{SQL_FILE_PATH}'...")
    with open(SQL_FILE_PATH, "r", encoding="utf-8") as f:
        sql_commands = f.read()

    cursor.executescript(sql_commands)
    conn.commit()
    print("    [OK] SQL Views 'v_rfm_customer_segments' and 'v_cohort_retention' created.")

    # 2. Extract RFM Segments
    print("\n[*] Exporting RFM Segmentation results...")
    rfm_df = pd.read_sql_query("SELECT * FROM v_rfm_customer_segments ORDER BY Monetary DESC;", conn)
    rfm_output_path = os.path.join(OUTPUT_DIR, "rfm_customer_segments.csv")
    rfm_df.to_csv(rfm_output_path, index=False)
    print(f"    [OK] Saved {len(rfm_df):,} segmented customer records to '{rfm_output_path}'.")

    # Display RFM summary
    rfm_summary = (
        rfm_df.groupby("Customer_Segment")
        .agg(
            Customer_Count=("CustomerID", "count"),
            Total_Revenue=("Monetary", "sum"),
            Avg_Recency_Days=("Recency", "mean"),
            Avg_Frequency_Orders=("Frequency", "mean"),
            Avg_Monetary_Spend=("Monetary", "mean"),
        )
        .reset_index()
    )
    rfm_summary["Revenue_Share_%"] = (
        (rfm_summary["Total_Revenue"] / rfm_summary["Total_Revenue"].sum()) * 100
    ).round(2)
    rfm_summary = rfm_summary.sort_values(by="Total_Revenue", ascending=False)

    print("\n[+] RFM Customer Segment Breakdown:")
    print(
        rfm_summary.to_string(
            index=False,
            formatters={
                "Customer_Count": "{:,}".format,
                "Total_Revenue": "${:,.2f}".format,
                "Avg_Recency_Days": "{:.1f}".format,
                "Avg_Frequency_Orders": "{:.1f}".format,
                "Avg_Monetary_Spend": "${:,.2f}".format,
                "Revenue_Share_%": "{:.1f}%".format,
            },
        )
    )

    # 3. Extract Cohort Retention Analysis
    print("\n[*] Processing Monthly Cohort Retention Matrix...")
    cohort_df = pd.read_sql_query("SELECT * FROM v_cohort_retention;", conn)
    
    # Save raw cohort details
    cohort_details_path = os.path.join(OUTPUT_DIR, "cohort_retention_details.csv")
    cohort_df.to_csv(cohort_details_path, index=False)

    # Pivot customer counts
    matrix_counts = cohort_df.pivot(
        index=["CohortMonth", "CohortSize"],
        columns="CohortIndex",
        values="ActiveCustomers"
    ).fillna(0).astype(int)

    # Pivot retention percentages
    matrix_rates = cohort_df.pivot(
        index=["CohortMonth", "CohortSize"],
        columns="CohortIndex",
        values="RetentionRatePercent"
    ).fillna(0.0)

    # Rename column headers to Month 0, Month 1, etc.
    matrix_counts.columns = [f"Month_{col}" for col in matrix_counts.columns]
    matrix_rates.columns = [f"Month_{col}" for col in matrix_rates.columns]

    matrix_counts_path = os.path.join(OUTPUT_DIR, "cohort_retention_matrix.csv")
    matrix_rates_path = os.path.join(OUTPUT_DIR, "cohort_retention_rates.csv")

    matrix_counts.to_csv(matrix_counts_path)
    matrix_rates.to_csv(matrix_rates_path)

    print(f"    [OK] Saved cohort customer count matrix to '{matrix_counts_path}'.")
    print(f"    [OK] Saved cohort retention rate matrix to '{matrix_rates_path}'.")

    # Display Cohort Retention Matrix (Rates)
    print("\n[+] Monthly Cohort Retention Rates (%):")
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)
    print(matrix_rates.map(lambda x: f"{x:.1f}%" if x > 0 else "-"))

    conn.close()
    print("\n" + "=" * 60)
    print("STAGE 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_analytics()
