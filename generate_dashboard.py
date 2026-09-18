"""
generate_dashboard.py
Stage 4: Generate Interactive Executive Analytics Dashboard (HTML)

This script reads data from 'analytics.db' and the 'output/' folder,
and produces a standalone, interactive, dark-mode 'dashboard.html'
with KPI cards, interactive RFM breakdown, Cohort Retention Heatmap,
ML Model metrics, Feature Importance bars, and Power BI DAX reference.
"""

import json
import os
import sqlite3
import pandas as pd

DB_PATH = "analytics.db"
OUTPUT_DIR = "output"
DASHBOARD_PATH = "dashboard.html"


def build_dashboard():
    print("=" * 60)
    print("STAGE 4: GENERATING INTERACTIVE EXECUTIVE DASHBOARD")
    print("=" * 60)

    # 1. Load data
    rfm_path = os.path.join(OUTPUT_DIR, "rfm_customer_segments.csv")
    cohort_rates_path = os.path.join(OUTPUT_DIR, "cohort_retention_rates.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "model_evaluation_metrics.json")
    feat_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    predictions_path = os.path.join(OUTPUT_DIR, "customer_churn_predictions.csv")

    rfm_df = pd.read_csv(rfm_path)
    cohort_df = pd.read_csv(cohort_rates_path)
    feat_df = pd.read_csv(feat_path)
    pred_df = pd.read_csv(predictions_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # High-level KPIs
    total_revenue = rfm_df["Monetary"].sum()
    total_customers = len(rfm_df)
    total_orders = rfm_df["Frequency"].sum()
    aov = total_revenue / total_orders
    churned_customers = (pred_df["Is_Churned"] == 1).sum()
    churn_rate = (churned_customers / total_customers) * 100
    at_risk_revenue = rfm_df[rfm_df["Customer_Segment"] == "At-Risk"]["Monetary"].sum()
    champions_revenue = rfm_df[rfm_df["Customer_Segment"] == "Champions"]["Monetary"].sum()

    # Segment Summary Table
    seg_summary = (
        rfm_df.groupby("Customer_Segment")
        .agg(
            Count=("CustomerID", "count"),
            Revenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
        )
        .reset_index()
        .sort_values(by="Revenue", ascending=False)
    )
    seg_summary["RevenueShare"] = (seg_summary["Revenue"] / total_revenue) * 100

    # Prepare JSON payloads for client-side rendering
    segments_json = seg_summary.to_dict(orient="records")
    features_json = feat_df.to_dict(orient="records")
    cohort_json = cohort_df.to_dict(orient="records")
    top_customers_json = (
        pred_df.sort_values(by="Churn_Probability", ascending=False)
        .head(100)[
            [
                "CustomerID",
                "Country",
                "Frequency",
                "Monetary",
                "CustomerTenure",
                "Churn_Probability",
                "Risk_Level",
            ]
        ]
        .to_dict(orient="records")
    )

    # Build HTML Content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>E-Commerce Customer Churn & Cohort Analytics Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0b0f19;
      --bg-secondary: #111827;
      --card-bg: rgba(30, 41, 59, 0.7);
      --border-color: rgba(255, 255, 255, 0.08);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-indigo: #818cf8;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --glow-blue: rgba(56, 189, 248, 0.15);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at 10% 20%, #0f172a 0%, #020617 90%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
    }}

    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}

    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 28px;
    }}

    .logo-badge {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.2);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--accent-blue);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}

    h1 {{
      font-size: 1.85rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .header-meta {{
      font-size: 0.9rem;
      color: var(--text-muted);
    }}

    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: var(--accent-emerald);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
    }}
    .pulse-dot {{
      width: 8px;
      height: 8px;
      background: var(--accent-emerald);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-emerald);
    }}

    /* Tabs */
    .tabs-nav {{
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      background: rgba(17, 24, 39, 0.6);
      padding: 6px;
      border-radius: 12px;
      border: 1px solid var(--border-color);
      width: fit-content;
    }}

    .tab-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 10px 20px;
      font-size: 0.92rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-family: inherit;
    }}

    .tab-btn:hover {{
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.04);
    }}

    .tab-btn.active {{
      background: var(--accent-blue);
      color: #0b0f19;
      font-weight: 700;
      box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
    }}

    .tab-pane {{
      display: none;
      animation: fadeIn 0.3s ease forwards;
    }}
    .tab-pane.active {{
      display: block;
    }}

    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}

    .kpi-card {{
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 20px;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-3px);
      border-color: rgba(56, 189, 248, 0.3);
    }}
    .kpi-label {{
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 1.8rem;
      font-weight: 800;
      color: var(--text-main);
      margin-bottom: 4px;
    }}
    .kpi-subtext {{
      font-size: 0.78rem;
      color: var(--text-muted);
    }}

    /* Card Panels */
    .panel-card {{
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .panel-title {{
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .panel-badge {{
      font-size: 0.75rem;
      background: rgba(255, 255, 255, 0.06);
      padding: 4px 10px;
      border-radius: 6px;
      color: var(--accent-blue);
      font-weight: 600;
    }}

    /* Data Tables */
    .table-container {{
      overflow-x: auto;
      border-radius: 12px;
      border: 1px solid var(--border-color);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      text-align: left;
    }}
    th {{
      background: rgba(17, 24, 39, 0.9);
      padding: 12px 16px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-color);
    }}
    td {{
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-main);
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}

    /* Badges */
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-champions {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .badge-loyal {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; }}
    .badge-at-risk {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
    .badge-churned {{ background: rgba(244, 63, 94, 0.2); color: #f87171; }}
    .badge-other {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }}

    .badge-high-risk {{ background: rgba(244, 63, 94, 0.25); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.4); }}
    .badge-medium-risk {{ background: rgba(245, 158, 11, 0.25); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .badge-low-risk {{ background: rgba(16, 185, 129, 0.25); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }}

    /* Heatmap styling */
    .heatmap-cell {{
      text-align: center;
      font-weight: 600;
      font-size: 0.8rem;
      border-radius: 4px;
      padding: 8px 4px;
    }}

    /* Bars */
    .bar-container {{
      width: 100%;
      background: rgba(255, 255, 255, 0.06);
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 6px;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}

    /* Grid columns */
    .split-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}
    @media (max-width: 960px) {{
      .split-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    /* DAX Code block */
    pre {{
      background: #020617;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 16px;
      border-radius: 10px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 0.85rem;
      color: #38bdf8;
      overflow-x: auto;
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <span class="logo-badge">Enterprise Analytics System</span>
        <h1>E-Commerce Churn & Cohort Intelligence</h1>
        <p class="header-meta">Data Pipeline • SQLite Analytics • ML Random Forest • Power BI Readiness</p>
      </div>
      <div>
        <div class="status-pill">
          <span class="pulse-dot"></span> Pipeline Active (SQLite &amp; ML)
        </div>
      </div>
    </header>

    <!-- Navigation Tabs -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('overview')">Executive Overview</button>
      <button class="tab-btn" onclick="switchTab('cohorts')">Cohort Retention Matrix</button>
      <button class="tab-btn" onclick="switchTab('ml-model')">ML Churn Model &amp; Risk Explorer</button>
      <button class="tab-btn" onclick="switchTab('dax')">Power BI DAX Measures</button>
    </div>

    <!-- TAB 1: EXECUTIVE OVERVIEW -->
    <div id="pane-overview" class="tab-pane active">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Total Revenue</div>
          <div class="kpi-value" style="color: #38bdf8;">${total_revenue:,.2f}</div>
          <div class="kpi-subtext">397,884 clean transactions</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Unique Customers</div>
          <div class="kpi-value">{total_customers:,}</div>
          <div class="kpi-subtext">18,532 distinct orders</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Average Order Value</div>
          <div class="kpi-value">${aov:,.2f}</div>
          <div class="kpi-subtext">Total Revenue / Total Orders</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Churn Rate (90d Inactive)</div>
          <div class="kpi-value" style="color: #f43f5e;">{churn_rate:.1f}%</div>
          <div class="kpi-subtext">{churned_customers:,} churned customers</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">At-Risk Revenue Exposure</div>
          <div class="kpi-value" style="color: #f59e0b;">${at_risk_revenue:,.2f}</div>
          <div class="kpi-subtext">820 slipping high-value accounts</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Champions Share</div>
          <div class="kpi-value" style="color: #10b981;">{(champions_revenue / total_revenue) * 100:.1f}%</div>
          <div class="kpi-subtext">1,127 top-tier customers</div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-title">
          <span>RFM Customer Segmentation Breakdown</span>
          <span class="panel-badge">Computed via SQL NTILE(5)</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Customer Segment</th>
                <th>Customers</th>
                <th>Total Spend</th>
                <th>Revenue Share</th>
                <th>Avg Recency</th>
                <th>Avg Frequency</th>
                <th>Avg Spend</th>
              </tr>
            </thead>
            <tbody id="rfm-tbody">
              <!-- Rendered dynamically -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 2: COHORT RETENTION -->
    <div id="pane-cohorts" class="tab-pane">
      <div class="panel-card">
        <div class="panel-title">
          <span>Monthly Customer Retention Cohort Matrix</span>
          <span class="panel-badge">13 Cohorts • Month 0 to Month 12</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 16px;">
          Tracks customer repeat purchase rates month-over-month from their initial acquisition month.
        </p>
        <div class="table-container">
          <table id="cohort-table">
            <!-- Rendered dynamically -->
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 3: ML CHURN MODEL -->
    <div id="pane-ml-model" class="tab-pane">
      <div class="split-grid">
        <div class="panel-card">
          <div class="panel-title">
            <span>Model Performance Summary</span>
            <span class="panel-badge">Random Forest (Stratified 80/20)</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px;">
            <div style="background: rgba(17, 24, 39, 0.7); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color);">
              <div class="kpi-label">ROC-AUC Score</div>
              <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">{metrics['roc_auc']:.4f}</div>
              <div class="kpi-subtext">Class separation power</div>
            </div>
            <div style="background: rgba(17, 24, 39, 0.7); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color);">
              <div class="kpi-label">Churn Recall</div>
              <div style="font-size: 1.6rem; font-weight: 800; color: #10b981;">{metrics['recall']:.1%}</div>
              <div class="kpi-subtext">Captures 258/290 churners</div>
            </div>
            <div style="background: rgba(17, 24, 39, 0.7); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color);">
              <div class="kpi-label">Accuracy</div>
              <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc;">{metrics['accuracy']:.1%}</div>
              <div class="kpi-subtext">Overall correct labels</div>
            </div>
            <div style="background: rgba(17, 24, 39, 0.7); padding: 14px; border-radius: 10px; border: 1px solid var(--border-color);">
              <div class="kpi-label">F1-Score</div>
              <div style="font-size: 1.6rem; font-weight: 800; color: #818cf8;">{metrics['f1_score']:.4f}</div>
              <div class="kpi-subtext">Harmonic mean P &amp; R</div>
            </div>
          </div>

          <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;">Confusion Matrix (Test Set):</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; text-align: center; font-size: 0.85rem;">
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); padding: 12px; border-radius: 8px;">
              <div style="font-size: 1.2rem; font-weight: 700; color: #34d399;">{metrics['confusion_matrix']['true_negatives']}</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">True Negatives (Active)</div>
            </div>
            <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); padding: 12px; border-radius: 8px;">
              <div style="font-size: 1.2rem; font-weight: 700; color: #f87171;">{metrics['confusion_matrix']['false_positives']}</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">False Positives (Active flagged Churn)</div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); padding: 12px; border-radius: 8px;">
              <div style="font-size: 1.2rem; font-weight: 700; color: #fbbf24;">{metrics['confusion_matrix']['false_negatives']}</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">False Negatives (Missed Churn)</div>
            </div>
            <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); padding: 12px; border-radius: 8px;">
              <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8;">{metrics['confusion_matrix']['true_positives']}</div>
              <div style="color: var(--text-muted); font-size: 0.75rem;">True Positives (Detected Churn)</div>
            </div>
          </div>
        </div>

        <div class="panel-card">
          <div class="panel-title">
            <span>Feature Importance Ranking</span>
            <span class="panel-badge">Gini Impurity Reduction</span>
          </div>
          <div id="features-list">
            <!-- Rendered dynamically -->
          </div>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-title">
          <span>Retention Priority Queue</span>
          <span class="panel-badge" id="priority-status">Loading live priorities</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.86rem; margin-bottom: 16px;">
          Ranked by model churn probability, risk tier, and customer value. Campaign actions are simulated only.
        </p>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Priority</th>
                <th>Customer ID</th>
                <th>RFM Segment</th>
                <th>Value</th>
                <th>Churn Probability</th>
                <th>Risk Tier</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody id="priority-tbody">
              <tr><td colspan="7" style="color: var(--text-muted);">Fetching current customer priorities...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel-card">
        <div class="panel-title">
          <span>Customer Churn Risk Explorer (Top 100 Scored Customers)</span>
          <span class="panel-badge">Full dataset exported to output/customer_churn_predictions.csv</span>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Country</th>
                <th>Orders</th>
                <th>Monetary Spend</th>
                <th>Tenure (Days)</th>
                <th>Churn Probability</th>
                <th>Risk Tier</th>
              </tr>
            </thead>
            <tbody id="risk-tbody">
              <!-- Rendered dynamically -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB 4: POWER BI DAX -->
    <div id="pane-dax" class="tab-pane">
      <div class="panel-card">
        <div class="panel-title">
          <span>Essential DAX Measures for Power BI</span>
          <span class="panel-badge">Full documentation in dax_measures.md</span>
        </div>
        
        <h3 style="font-size: 1rem; color: var(--accent-blue); margin-bottom: 8px;">1. [Active Customers (90 Days)]</h3>
        <pre>Active Customers (90 Days) = 
CALCULATE(
    DISTINCTCOUNT(FactTransactions[CustomerID]),
    DATESINPERIOD(DimDate[Date], MAX(FactTransactions[InvoiceDate]), -90, DAY)
)</pre>

        <h3 style="font-size: 1rem; color: var(--accent-blue); margin-bottom: 8px;">2. [Churn Rate %]</h3>
        <pre>Churn Rate % = 
DIVIDE(
    CALCULATE(DISTINCTCOUNT(DimCustomer[CustomerID]), DimCustomer[Is_Churned] = 1),
    DISTINCTCOUNT(DimCustomer[CustomerID]),
    0
)</pre>

        <h3 style="font-size: 1rem; color: var(--accent-blue); margin-bottom: 8px;">3. [At-Risk Revenue Exposure]</h3>
        <pre>At-Risk Revenue Exposure = 
CALCULATE([Total Sales], DimCustomer[Customer_Segment] = "At-Risk")</pre>

        <h3 style="font-size: 1rem; color: var(--accent-blue); margin-bottom: 8px;">4. [Customer Lifetime Value (Predictive CLV)]</h3>
        <pre>Predictive CLV = 
DIVIDE([ARPU], IF([Churn Rate %] = 0, 0.05, [Churn Rate %]), 0)</pre>
      </div>
    </div>

  </div>

  <script>
    const segmentsData = {json.dumps(segments_json)};
    const featuresData = {json.dumps(features_json)};
    const cohortData = {json.dumps(cohort_json)};
    const topCustomers = {json.dumps(top_customers_json)};

    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      const buttons = Array.from(document.querySelectorAll('.tab-btn'));
      const activeBtn = buttons.find(b => b.getAttribute('onclick').includes(tabId));
      if (activeBtn) activeBtn.classList.add('active');
      
      const pane = document.getElementById('pane-' + tabId);
      if (pane) pane.classList.add('active');
    }}

    // Render RFM Table
    const rfmTbody = document.getElementById('rfm-tbody');
    segmentsData.forEach(row => {{
      let badgeClass = 'badge-other';
      if (row.Customer_Segment === 'Champions') badgeClass = 'badge-champions';
      else if (row.Customer_Segment === 'Loyal Customers') badgeClass = 'badge-loyal';
      else if (row.Customer_Segment === 'At-Risk') badgeClass = 'badge-at-risk';
      else if (row.Customer_Segment === 'Churned') badgeClass = 'badge-churned';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="badge ${{badgeClass}}">${{row.Customer_Segment}}</span></td>
        <td style="font-weight: 600;">${{row.Count.toLocaleString()}}</td>
        <td style="font-weight: 700;">$${{row.Revenue.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="min-width: 40px;">${{row.RevenueShare.toFixed(1)}}%</span>
            <div class="bar-container" style="flex: 1;">
              <div class="bar-fill" style="width: ${{row.RevenueShare}}%; background: var(--accent-blue);"></div>
            </div>
          </div>
        </td>
        <td>${{row.AvgRecency.toFixed(1)}} days</td>
        <td>${{row.AvgFrequency.toFixed(1)}} orders</td>
        <td>$${{row.AvgMonetary.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
      `;
      rfmTbody.appendChild(tr);
    }});

    // Render Features
    const featContainer = document.getElementById('features-list');
    const maxImportance = Math.max(...featuresData.map(f => f.Importance));
    featuresData.forEach(f => {{
      const pct = (f.Importance / maxImportance) * 100;
      const div = document.createElement('div');
      div.style.marginBottom = '14px';
      div.innerHTML = `
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px;">
          <span>${{f.Feature}}</span>
          <span style="color: var(--accent-blue);">${{(f.Importance * 100).toFixed(2)}}%</span>
        </div>
        <div class="bar-container">
          <div class="bar-fill" style="width: ${{pct}}%; background: linear-gradient(90deg, #38bdf8, #818cf8);"></div>
        </div>
      `;
      featContainer.appendChild(div);
    }});

    // Render Cohort Table
    const cohortTable = document.getElementById('cohort-table');
    if (cohortData.length > 0) {{
      const cols = Object.keys(cohortData[0]);
      let thead = '<thead><tr>';
      cols.forEach(c => {{
        thead += `<th>${{c.replace('_', ' ')}}</th>`;
      }});
      thead += '</tr></thead>';

      let tbody = '<tbody>';
      cohortData.forEach(row => {{
        tbody += `<tr>`;
        cols.forEach(c => {{
          const val = row[c];
          if (c === 'CohortMonth') {{
            tbody += `<td style="font-weight: 700;">${{val}}</td>`;
          }} else if (c === 'CohortSize') {{
            tbody += `<td style="font-weight: 600;">${{val.toLocaleString()}}</td>`;
          }} else {{
            const num = parseFloat(val);
            if (!isNaN(num) && num > 0) {{
              let bg = 'rgba(56, 189, 248, 0.05)';
              let color = '#94a3b8';
              if (num >= 40) {{ bg = 'rgba(16, 185, 129, 0.4)'; color = '#6ee7b7'; }}
              else if (num >= 25) {{ bg = 'rgba(56, 189, 248, 0.3)'; color = '#7dd3fc'; }}
              else if (num >= 15) {{ bg = 'rgba(129, 140, 248, 0.2)'; color = '#a5b4fc'; }}
              else {{ bg = 'rgba(244, 63, 94, 0.15)'; color = '#fca5a5'; }}
              tbody += `<td class="heatmap-cell" style="background: ${{bg}}; color: ${{color}};">${{num.toFixed(1)}}%</td>`;
            }} else {{
              tbody += `<td class="heatmap-cell" style="color: rgba(255,255,255,0.1);">-</td>`;
            }}
          }}
        }});
        tbody += `</tr>`;
      }});
      tbody += '</tbody>';
      cohortTable.innerHTML = thead + tbody;
    }}

    // Render Top Customers
    const riskTbody = document.getElementById('risk-tbody');
    topCustomers.forEach(c => {{
      let badge = 'badge-low-risk';
      if (c.Risk_Level === 'High Risk') badge = 'badge-high-risk';
      else if (c.Risk_Level === 'Medium Risk') badge = 'badge-medium-risk';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-family: monospace; font-weight: 600;">#${{c.CustomerID}}</td>
        <td>${{c.Country}}</td>
        <td>${{c.Frequency}}</td>
        <td style="font-weight: 600;">$${{c.Monetary.toLocaleString(undefined, {{minimumFractionDigits: 2}})}}</td>
        <td>${{c.CustomerTenure}} days</td>
        <td style="font-weight: 700; color: ${{c.Churn_Probability > 0.7 ? '#f43f5e' : '#fbbf24'}};">${{(c.Churn_Probability * 100).toFixed(1)}}%</td>
        <td><span class="badge ${{badge}}">${{c.Risk_Level}}</span></td>
      `;
      riskTbody.appendChild(tr);
    }});

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }}

    function riskBadgeClass(riskLevel) {{
      if (riskLevel === 'High Risk') return 'badge-high-risk';
      if (riskLevel === 'Medium Risk') return 'badge-medium-risk';
      return 'badge-low-risk';
    }}

    function renderPriorityCandidates(candidates) {{
      const priorityTbody = document.getElementById('priority-tbody');
      const priorityStatus = document.getElementById('priority-status');
      if (!candidates.length) {{
        priorityTbody.innerHTML = '<tr><td colspan="7" style="color: var(--text-muted);">No customer transactions are loaded. Upload a CSV to generate priorities.</td></tr>';
        priorityStatus.textContent = 'No live data';
        return;
      }}

      priorityStatus.textContent = `${{candidates.length.toLocaleString()}} live candidates`;
      priorityTbody.innerHTML = candidates.slice(0, 25).map((candidate, index) => {{
        const customerId = escapeHtml(candidate.customer_id);
        const riskLevel = escapeHtml(candidate.risk_level);
        const badge = riskBadgeClass(candidate.risk_level);
        const probability = Number(candidate.churn_probability || 0);
        return `
          <tr>
            <td style="font-weight: 800; color: var(--accent-amber);">#${{index + 1}}</td>
            <td style="font-family: monospace; font-weight: 600;">${{customerId}}</td>
            <td>${{escapeHtml(candidate.rfm_segment || 'Not available')}}</td>
            <td style="font-weight: 600;">$${{Number(candidate.monetary_value || 0).toLocaleString(undefined, {{minimumFractionDigits: 2}})}}</td>
            <td style="font-weight: 700; color: ${{probability >= 0.7 ? '#f43f5e' : '#fbbf24'}};">${{(probability * 100).toFixed(1)}}%</td>
            <td><span class="badge ${{badge}}">${{riskLevel}}</span></td>
            <td><button class="panel-badge" style="cursor: pointer; border: 1px solid rgba(56, 189, 248, 0.35); color: var(--accent-blue); background: rgba(56, 189, 248, 0.08);" onclick="simulateCampaign(${{JSON.stringify(candidate.customer_id)}}, ${{JSON.stringify(candidate.risk_level)}}, ${{JSON.stringify(candidate.rfm_segment)}}, ${{probability}}, ${{Number(candidate.monetary_value || 0)}})">Simulate</button></td>
          </tr>`;
      }}).join('');
    }}

    async function loadPriorityCandidates() {{
      const priorityStatus = document.getElementById('priority-status');
      try {{
        const response = await fetch('/priority');
        if (!response.ok) throw new Error('Priority endpoint unavailable');
        const data = await response.json();
        renderPriorityCandidates(data.candidates || []);
      }} catch (error) {{
        priorityStatus.textContent = 'Embedded baseline';
        document.getElementById('priority-tbody').innerHTML = '<tr><td colspan="7" style="color: var(--text-muted);">Live priorities appear when this dashboard is served by the FastAPI application after data upload.</td></tr>';
      }}
    }}

    async function simulateCampaign(customerId, riskLevel, rfmSegment, churnProbability, monetaryValue) {{
      try {{
        const response = await fetch('/campaign/generate', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            customer_id: customerId,
            risk_level: riskLevel,
            rfm_segment: rfmSegment,
            churn_probability: churnProbability,
            monetary_value: monetaryValue,
            campaign_type: riskLevel === 'High Risk' ? 'Win-Back' : 'Retention Nurture'
          }})
        }});
        if (!response.ok) throw new Error('Campaign simulation failed');
        window.alert(`Simulated campaign created for customer ${{customerId}}.`);
      }} catch (error) {{
        window.alert('Campaign simulation requires the dashboard to be served by the FastAPI application.');
      }}
    }}

    loadPriorityCandidates();
  </script>
</body>
</html>
"""

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Standalone interactive dashboard created at: '{DASHBOARD_PATH}'")
    print("=" * 60)


if __name__ == "__main__":
    build_dashboard()
