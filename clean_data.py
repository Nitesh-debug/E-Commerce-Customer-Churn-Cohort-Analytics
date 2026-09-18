"""
clean_data.py
Stage 1: Data Cleaning and SQLite Ingestion

This script:
1. Loads raw transactional data from 'online_retail.csv'.
2. Removes records with missing CustomerID.
3. Filters out negative/zero quantities and unit prices (returns/cancellations).
4. Formats InvoiceDate to standard ISO format and casts CustomerID to int.
5. Calculates 'TotalAmount' = Quantity * UnitPrice.
6. Ingests cleaned data into 'analytics.db' in the 'transactions' table.
7. Creates database indexes for high-performance SQL analytics.
"""

import os
import sqlite3
import pandas as pd

RAW_DATA_PATH = "online_retail.csv"
DB_PATH = "analytics.db"
TABLE_NAME = "transactions"


def clean_and_ingest():
    print("=" * 60)
    print("STAGE 1: DATA CLEANING & INGESTION")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file '{RAW_DATA_PATH}' not found.")

    print(f"[*] Reading raw data from '{RAW_DATA_PATH}'...")
    raw_df = pd.read_csv(RAW_DATA_PATH, dtype={"InvoiceNo": str, "StockCode": str})
    initial_rows = len(raw_df)
    print(f"    - Initial records: {initial_rows:,}")

    # 1. Handle missing CustomerIDs
    missing_customers = raw_df["CustomerID"].isnull().sum()
    df = raw_df.dropna(subset=["CustomerID"]).copy()
    print(f"    - Dropped {missing_customers:,} records with missing CustomerID.")

    # 2. Filter out non-positive quantities and unit prices (cancellations, adjustments)
    non_positive_qty = (df["Quantity"] <= 0).sum()
    non_positive_price = (df["UnitPrice"] <= 0).sum()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    print(f"    - Filtered out non-positive quantities and prices (removed {initial_rows - missing_customers - len(df):,} records).")

    # 3. Data type formatting & new columns
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()
    df["InvoiceDate"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 4. Create TotalAmount column
    df["TotalAmount"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    # Reorder columns cleanly
    columns_order = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "TotalAmount",
    ]
    df = df[columns_order]
    clean_rows = len(df)

    print("\n[+] Data Cleaning Summary:")
    print(f"    - Initial rows:       {initial_rows:,}")
    print(f"    - Cleaned rows:       {clean_rows:,} ({clean_rows / initial_rows:.1%} retained)")
    print(f"    - Unique Customers:   {df['CustomerID'].nunique():,}")
    print(f"    - Unique Invoices:    {df['InvoiceNo'].nunique():,}")
    print(f"    - Total Revenue:      ${df['TotalAmount'].sum():,.2f}")
    print(f"    - Date Range:         {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # 5. Save to SQLite database
    print(f"\n[*] Saving cleaned data into SQLite database: '{DB_PATH}'...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"    - Removed existing '{DB_PATH}' for fresh build.")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print("    - Creating indexes on CustomerID, InvoiceDate, and InvoiceNo...")
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX idx_transactions_customer ON {TABLE_NAME}(CustomerID);")
    cursor.execute(f"CREATE INDEX idx_transactions_invoicedate ON {TABLE_NAME}(InvoiceDate);")
    cursor.execute(f"CREATE INDEX idx_transactions_invoiceno ON {TABLE_NAME}(InvoiceNo);")
    conn.commit()

    # Verification query
    cursor.execute(f"SELECT COUNT(*), COUNT(DISTINCT CustomerID), ROUND(SUM(TotalAmount), 2) FROM {TABLE_NAME};")
    db_count, db_custs, db_rev = cursor.fetchone()
    conn.close()

    print(f"[OK] Database verification passed:")
    print(f"    - Table '{TABLE_NAME}' record count: {db_count:,}")
    print(f"    - Table '{TABLE_NAME}' customer count: {db_custs:,}")
    print(f"    - Table '{TABLE_NAME}' total revenue: ${db_rev:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    clean_and_ingest()
