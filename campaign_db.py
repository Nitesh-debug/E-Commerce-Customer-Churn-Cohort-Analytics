"""
campaign_db.py
Campaign History SQLite Storage

Manages a lightweight SQLite database (campaigns.db) for storing simulated
campaign records. Completely separate from analytics.db to preserve pipeline integrity.

Architecture note:
  This module is intentionally simple. The campaign table stores all generated
  campaigns with a SIMULATED status. Future integration with real email/SMS
  providers would change the status field and add provider-specific fields.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

CAMPAIGNS_DB_PATH = os.path.join(os.path.dirname(__file__), "campaigns.db")
CAMPAIGNS_TABLE = "campaigns"


def get_connection() -> sqlite3.Connection:
    """Return a connection to campaigns.db."""
    conn = sqlite3.connect(CAMPAIGNS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_campaigns_db() -> None:
    """
    Create campaigns table if it does not exist.
    Called once at application startup.
    """
    conn = get_connection()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {CAMPAIGNS_TABLE} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id     TEXT    NOT NULL,
            risk_level      TEXT    NOT NULL,
            rfm_segment     TEXT,
            monetary_value  REAL,
            churn_probability REAL,
            campaign_type   TEXT    NOT NULL,
            offer           TEXT,
            message         TEXT,
            status          TEXT    NOT NULL DEFAULT 'SIMULATED',
            model_version   TEXT,
            created_at      TEXT    NOT NULL
        );
    """)
    columns = {
        row["name"] for row in conn.execute(f"PRAGMA table_info({CAMPAIGNS_TABLE})").fetchall()
    }
    if "model_version" not in columns:
        conn.execute(f"ALTER TABLE {CAMPAIGNS_TABLE} ADD COLUMN model_version TEXT")
    conn.commit()
    conn.close()


def save_campaign(record: Dict[str, Any]) -> int:
    """
    Insert a campaign record into campaigns.db.

    Parameters
    ----------
    record : dict with keys matching campaigns table columns
        Required: customer_id, risk_level, campaign_type
        Optional: rfm_segment, monetary_value, churn_probability, offer, message

    Returns
    -------
    int : row id of the inserted campaign
    """
    conn = get_connection()
    cursor = conn.execute(
        f"""
        INSERT INTO {CAMPAIGNS_TABLE}
            (customer_id, risk_level, rfm_segment, monetary_value, churn_probability,
            campaign_type, offer, message, status, model_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(record.get("customer_id", "")),
            record.get("risk_level", ""),
            record.get("rfm_segment"),
            record.get("monetary_value"),
            record.get("churn_probability"),
            record.get("campaign_type", ""),
            record.get("offer"),
            record.get("message"),
            record.get("status", "SIMULATED"),
            record.get("model_version"),
            record.get("created_at", datetime.utcnow().isoformat()),
        ),
    )
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def list_campaigns(
    limit: int = 200,
    customer_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List campaign history, optionally filtered by customer_id.

    Returns
    -------
    List[dict] ordered by created_at DESC.
    """
    conn = get_connection()
    if customer_id:
        rows = conn.execute(
            f"""SELECT * FROM {CAMPAIGNS_TABLE}
                WHERE customer_id = ?
                ORDER BY created_at DESC LIMIT ?""",
            (str(customer_id), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT * FROM {CAMPAIGNS_TABLE}
                ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_campaign_count() -> int:
    """Return total number of campaigns stored."""
    conn = get_connection()
    count = conn.execute(f"SELECT COUNT(*) FROM {CAMPAIGNS_TABLE}").fetchone()[0]
    conn.close()
    return count
