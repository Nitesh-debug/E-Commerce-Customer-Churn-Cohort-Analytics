# End-to-End E-Commerce Customer Churn & Cohort Analytics System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.141-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite3-lightgrey.svg)](https://www.sqlite.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Uvicorn](https://img.shields.io/badge/Server-Uvicorn%20ASGI-499848.svg)](https://www.uvicorn.org/)
[![Power BI Ready](https://img.shields.io/badge/BI-Power%20BI%20DAX-yellow.svg)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, full-lifecycle customer analytics, cohort retention, and machine learning system built on the **Online Retail** transactional dataset (~541k records). 

This repository implements automated data ingestion, an embedded analytical data warehouse, high-performance SQL RFM customer segmentation, monthly retention cohort analysis, an interpretable and balanced Random Forest churn prediction model, a production-grade FastAPI real-time scoring microservice, and an interactive executive analytics dashboard.

---

## Table of Contents

1. [Executive Summary & Business Impact](#1-executive-summary--business-impact)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Technology Stack & Architectural Rationale](#3-technology-stack--architectural-rationale)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Step-by-Step Pipeline Deep Dive](#5-step-by-step-pipeline-deep-dive)
   - [Stage 1: Ingestion & Relational Data Warehouse](#stage-1-data-cleaning--ingestion-cleandatapy)
   - [Stage 2: SQL RFM Segmentation & Cohort Analysis](#stage-2-sql-rfm-segmentation--cohort-analysis-run_sql_analyticspy)
   - [Stage 3: Machine Learning Churn Prediction](#stage-3-machine-learning-churn-prediction-train_churn_modelpy)
   - [Stage 4: Executive Analytics Dashboard & Power BI](#stage-4-interactive-executive-dashboard-dashboardhtml)
   - [Stage 5: Real-Time FastAPI Scoring Microservice](#stage-5-production-ready-churn-scoring-api-apppy)
6. [Core Technical & Machine Learning Concepts (Interview Deep Dive)](#6-core-technical--machine-learning-concepts-interview-deep-dive)
   - [Target Leakage Prevention](#1-target-leakage-prevention-critical-ml-concept)
   - [Purchase Velocity: The #1 Predictive Feature](#2-purchase-velocity-the-1-predictive-feature)
   - [Handling Class Imbalance & Metric Selection](#3-handling-class-imbalance--metric-selection)
   - [SQL Window Functions & RFM Quintiles](#4-sql-window-functions--rfm-quintiles-ntile5)
   - [FastAPI Microservice Design Patterns](#5-fastapi-microservice-design-patterns)
7. [Comprehensive Interview Q&A (Cheat Sheet)](#7-comprehensive-interview-qa-cheat-sheet)
8. [Quickstart & Execution Guide](#8-quickstart--execution-guide)
9. [Automated Testing & Quality Assurance](#9-automated-testing--quality-assurance)

---

## 1. Executive Summary & Business Impact

| Metric / KPI | Value | Business Interpretation |
| :--- | :--- | :--- |
| **Analyzed Raw Records** | **541,909** | Total unprocessed transaction line items from the online retailer. |
| **Validated Clean Transactions** | **397,884** (73.4%) | Valid sales retained after filtering returns, cancellations, and missing customer IDs. |
| **Total Captured Revenue** | **$8,911,407.90** | Lifetime revenue across 18,532 distinct customer orders. |
| **Unique Customer Accounts** | **4,338** | Tracked behavioral profiles across 37 global markets (dominant UK cohort). |
| **ML Model ROC-AUC** | **0.9558** | Near-optimal probabilistic separation between active customers and churners. |
| **Churn Recall (Sensitivity)** | **88.97%** | Successfully flags **258 out of 290** at-risk customers before churn occurs. |
| **Revenue at Immediate Risk** | **$1,135,452.58** | Value represented by the "At-Risk" segment (820 high-value customers). |
| **Champions Concentration** | **66.8% ($5.95M)** | 1,127 VIP customers generate two-thirds of total enterprise revenue. |

### Business Problem
Acquiring new e-commerce customers is **5x to 7x more expensive** than retaining existing ones. E-commerce platforms suffer from silent attrition: customers stop purchasing without formal notification or subscription cancellation. This project delivers a proactive retention engine that predicts churn **before** it happens and prescribes targeted marketing interventions based on individual churn probability and lifetime spend.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1 ["Stage 1: Data Ingestion & Storage"]
        Raw["Raw CSV: online_retail.csv<br/>541,909 rows"] --> Clean["clean_data.py<br/>Validation & Deduplication"]
        Clean --> DB[("SQLite: analytics.db<br/>Table: transactions<br/>397,884 clean rows<br/>$8.91M Revenue")]
    end

    subgraph S2 ["Stage 2: SQL Analytics Engine"]
        DB --> SQL["rfm_cohort_analysis.sql<br/>Window Functions & CTEs"]
        SQL --> RunSQL["run_sql_analytics.py"]
        RunSQL --> CSV1["output/rfm_customer_segments.csv"]
        RunSQL --> CSV2["output/cohort_retention_matrix.csv"]
        RunSQL --> CSV3["output/cohort_retention_rates.csv"]
    end

    subgraph S3 ["Stage 3: ML Pipeline & Modeling"]
        DB --> FeatEng["train_churn_model.py<br/>Feature Engineering (No Leakage)"]
        FeatEng -->|Stratified 80/20 Split| RF["Balanced Random Forest<br/>150 Trees | Depth 6"]
        RF --> ModelArt["churn_model.joblib<br/>Serialized Model"]
        RF --> Metrics["output/model_evaluation_metrics.json"]
        RF --> FI["output/feature_importance.csv"]
        RF --> CustPred["output/customer_churn_predictions.csv"]
    end

    subgraph S4 ["Stage 4: BI & Visual Dashboards"]
        CSV1 & CSV2 & CustPred --> DashGen["generate_dashboard.py"]
        DashGen --> DashHTML["dashboard.html<br/>Interactive Executive Dashboard"]
        DAX["dax_measures.md"] --> PowerBI["Power BI Desktop Integration"]
    end

    subgraph S5 ["Stage 5: Production Scoring API"]
        ModelArt --> API["app.py<br/>FastAPI Microservice"]
        API --> Docs["Swagger UI: /docs"]
        API --> Health["GET /health"]
        API --> Predict["POST /predict<br/>Real-Time Churn Scoring"]
        TestAPI["test_api.py<br/>8 Automated TestClient Tests"] --> API
    end
```

---

## 3. Technology Stack & Architectural Rationale

| Category | Technology | Purpose in System | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core orchestration, pipelines, ML, API | Industry standard for data science, ML libraries, and asynchronous web backends. |
| **Database** | SQLite 3.38+ | Embedded analytical warehouse (`analytics.db`) | Zero-configuration, serverless, atomic, and supports CTEs and window functions (`NTILE`). |
| **Data Processing** | Pandas, NumPy | Data cleaning, matrix pivoting, feature math | Fast vectorized columnar manipulations for tabular transactional datasets. |
| **Machine Learning** | Scikit-Learn | Tuned Random Forest, evaluation metrics | Highly interpretable, handles non-linearities, robust against multicollinearity, built-in feature importances. |
| **Model Persistence** | Joblib | Serialization (`churn_model.joblib`) | Optimized for saving NumPy arrays and tree-based estimators with minimal serialization overhead. |
| **REST API** | FastAPI | Real-time scoring microservice (`app.py`) | Async ASGI performance, auto-generated OpenAPI documentation (`/docs`), Pydantic validation. |
| **ASGI Server** | Uvicorn (Standard) | High-performance ASGI production server | Lightweight, event-loop driven concurrency for scalable HTTP request handling. |
| **Schema Validation** | Pydantic v2 | Data validation and fallback derivation | Robust type enforcement, automated error payloads (HTTP 422), and domain-specific feature derivation. |
| **QA / Testing** | Unittest & HTTPX | Pipeline and API test coverage | Validates database integrity, ML metric thresholds, and API endpoint contracts. |
| **Front-End / BI** | HTML5, CSS3, Power BI DAX | Visual executive dashboard & BI modeling | Zero-dependency executive dashboard viewable in any modern browser without extra servers. |

---

## 4. Project Directory Structure

```text
.
├── app.py                             # Stage 5: Production FastAPI service for real-time scoring
├── test_api.py                        # Automated unit & integration tests for FastAPI (8 tests)
├── requirements.txt                   # Pinned dependency definitions
├── online_retail.csv                  # Raw e-commerce transaction dataset (541k rows)
├── run_pipeline.py                    # Master End-to-End Pipeline Runner (Stages 1-4)
├── test_pipeline.py                   # Automated QA test suite for DB & ML model (4 tests)
├── clean_data.py                      # Stage 1: Data cleaning & SQLite ingestion
├── rfm_cohort_analysis.sql            # Stage 2: SQL RFM and Cohort views (CTEs + Window functions)
├── run_sql_analytics.py               # Stage 2: SQL execution & CSV export runner
├── train_churn_model.py               # Stage 3: ML feature extraction & model training
├── generate_dashboard.py              # Stage 4: Interactive HTML dashboard generator
├── dashboard.html                     # Standalone executive analytics dashboard
├── churn_model.joblib                 # Serialized Random Forest model artifact
├── dax_measures.md                    # Enterprise Power BI DAX library
├── analytics.db                       # Local SQLite database (indexed tables)
├── output/                            # Generated analytical and model artifacts
│   ├── rfm_customer_segments.csv      # Customer-level RFM scores & segments (4,338 records)
│   ├── cohort_retention_matrix.csv    # Monthly cohort active customer counts
│   ├── cohort_retention_rates.csv     # Monthly cohort retention rate percentages
│   ├── cohort_retention_details.csv   # Raw granular cohort metrics
│   ├── model_evaluation_metrics.json  # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── feature_importance.csv         # Ranked predictive feature weights
│   └── customer_churn_predictions.csv # Customer-level churn risk probabilities
└── README.md                          # Project documentation
```

---

## 5. Step-by-Step Pipeline Deep Dive

### Stage 1: Data Cleaning & Ingestion (`clean_data.py`)
- **Raw Ingestion**: Processes 541,909 raw records from `online_retail.csv`.
- **Cleaning Strategy**:
  1. **Identify Identifiable Accounts**: Drops 135,080 records with null `CustomerID`. In customer retention, transactions that cannot be attributed to an account cannot be modeled.
  2. **Eliminate Cancellations & Negatives**: Drops 8,945 records where `Quantity <= 0` (cancellations prefixed with 'C') or `UnitPrice <= 0` (system adjustments).
  3. **Feature Engineering**: Generates line revenue: `TotalAmount = Quantity * UnitPrice`.
- **Database Schema**: Generates `analytics.db` with SQLite B-tree indexes on `CustomerID`, `InvoiceDate`, and `InvoiceNo` to accelerate subsequent analytical joins and window aggregations.
- **Output**: 397,884 cleaned rows across 4,338 unique customers and $8,911,407.90 in revenue.

### Stage 2: SQL RFM Segmentation & Cohort Analysis (`run_sql_analytics.py`)
- **RFM Quintile Scoring**:
  - Uses SQLite CTEs and window function `NTILE(5)` over:
    - **Recency ($R$)**: Days since last transaction relative to dynamic snapshot.
    - **Frequency ($F$)**: Distinct invoice count.
    - **Monetary ($M$)**: Total spend.
  - Inverts Recency score ($5 - NTILE(5) + 1$) so that lower recency days yield higher loyalty scores.
- **8 Customer Behavioral Segments**:
  - **Champions (1,127 customers / 66.8% revenue)**: Average spend $5,279.29, recency 13.6 days.
  - **Loyal Customers (826 customers / 15.6% revenue)**: Average spend $1,685.28, recency 38.4 days.
  - **At-Risk (820 customers / 12.7% revenue)**: Average spend $1,384.70, recency 156.7 days ($1.13M revenue at risk).
  - **Churned (604 customers / 1.5% revenue)**: Average spend $226.58, recency 276.8 days.
  - **Potential Loyalists (308)**, **Hibernating (312)**, **Need Attention (192)**, **Promising (149)**.
- **Monthly Cohort Retention Analysis**:
  - Assigns each customer to an acquisition cohort based on their initial transaction month (`strftime('%Y-%m', MIN(InvoiceDate))`).
  - Computes `CohortIndex = (InvoiceYear - CohortYear)*12 + (InvoiceMonth - CohortMonth)`.
  - Pivots into active customer counts and percentage retention rates across Months 0 to 12.

### Stage 3: Machine Learning Churn Prediction (`train_churn_model.py`)
- **Target Variable Definition**: `Is_Churned = 1` if customer inactivity $> 90$ days, else `0`.
- **Feature Set (9 Features)**:
  - `Frequency`, `Monetary`, `AvgOrderValue`, `TotalUnits`, `AvgBasketSize`, `DistinctProducts`, `CustomerTenure`, `PurchaseVelocity`, `Is_UK`.
- **Model Training**:
  - Random Forest Classifier (150 trees, max depth 6, min samples leaf 3, `class_weight="balanced"`).
  - Stratified 80/20 train/test split.
- **Evaluation Metrics (Test Set - 868 customers)**:
  - **ROC-AUC**: **0.9558**
  - **Recall**: **88.97%** (Identifies 258 out of 290 actual churners)
  - **Precision**: **75.22%**
  - **F1-Score**: **0.8152**
  - **Accuracy**: **86.52%**
- **Confusion Matrix**:
  ```text
                    Predicted Active (0)    Predicted Churned (1)
  Actual Active (0)         493 (TN)                 85 (FP)
  Actual Churned (1)         32 (FN)                258 (TP)
  ```
- **Feature Importance Rankings**:
  1. `PurchaseVelocity`: **50.52%** (Dominant leading indicator)
  2. `CustomerTenure`: **19.66%**
  3. `Frequency`: **11.15%**
  4. `Monetary`: **6.85%**
  5. `TotalUnits`: **5.15%**
  6. `DistinctProducts`: **3.31%**
  7. `AvgOrderValue`: **1.85%**
  8. `AvgBasketSize`: **1.48%**
  9. `Is_UK`: **0.03%**

### Stage 4: Interactive Executive Dashboard (`dashboard.html`)
- Built using self-contained vanilla HTML/CSS/JS without external CDN dependencies.
- Features:
  - Executive KPI summary cards (Total Revenue, Active Customers, Churn Rate %, High-Risk Count).
  - Visual Cohort Retention Heatmap matrix.
  - Customer segment distribution breakdown.
  - Recency vs. Frequency scatter charts with risk overlay.

### Stage 5: Production-Ready Churn Scoring API (`app.py`)
- **FastAPI Microservice**:
  - Real-time scoring endpoint at `POST /predict`.
  - Service health check at `GET /health`.
  - Interactive Swagger documentation at `/docs` and ReDoc at `/redoc`.
  - **Lifespan Manager**: Loads `churn_model.joblib` into memory upon server startup, avoiding per-request disk read latency.
  - **Resilient Fallback**: Accepts core metrics (`frequency`, `monetary`, `tenure`) and automatically computes derived features.
  - **Dynamic CRM Rule Engine**: Maps probability into operational risk tiers (`High Risk`, `Medium Risk`, `Low Risk`) and provides tailored marketing action text.
  - **Retention workflow**: `POST /upload` derives features from raw transactions, `GET /customers/{customer_id}` analyzes an individual customer, `POST /predict/bulk` scores a file, and `GET /priority` ranks retention candidates by model risk and customer value.
  - **Campaign simulation**: `POST /campaign/generate` records a `SIMULATED` action only; it does not call an external CRM or messaging provider.
  - **Configuration visibility**: `GET /` exposes the active 90-day churn rule, risk thresholds, and VIP monetary threshold used by the service.
    - **Operational controls**: normalized uploads persist in `uploads.db` and are restored at startup; request logs are emitted as JSON; `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS` configure per-client throttling; set `API_KEY` to require `X-API-Key` on protected routes.
    - **Model governance**: `GET /model/status` reports the serving artifact checksum, feature contract, and active thresholds.

---

## 6. Core Technical & Machine Learning Concepts (Interview Deep Dive)

### 1. Target Leakage Prevention (Critical ML Concept)
> [!IMPORTANT]
> **Why is `Recency` NOT included in the model features?**
> The target variable `Is_Churned` is defined as `Recency > 90 days`. If `Recency` were included as an input feature, the Random Forest model would learn a trivial single-split rule (`Recency > 90`) and achieve a 100% false accuracy score. In a real-time production system, when an active customer visits the platform today, their recency is 0, so that feature provides zero forward-looking signal. To eliminate data leakage, `Recency` was used **strictly** to define the ground-truth label, and the model was forced to learn churn patterns purely from behavioral and velocity dynamics.

### 2. Purchase Velocity: The #1 Predictive Feature
- **Formula**:
  $$\text{PurchaseVelocity} = \frac{\text{Frequency}}{\text{CustomerTenure} + 1}$$
- **Why it explains 50.5% of the model**:
  Raw frequency is biased toward older customers. A customer with 5 purchases across 2 years has a low purchase velocity ($0.0068$ orders/day), whereas a customer with 5 purchases across 30 days has a high purchase velocity ($0.161$ orders/day). When a customer's purchase cadence decelerates relative to their tenure, it acts as a primary signal of disengagement weeks before they cross the 90-day churn threshold.

### 3. Handling Class Imbalance & Metric Selection
- **The Imbalance Problem**:
  In our customer base, **66.6% are Active** and **33.4% are Churned**. A naive baseline model that blindly predicts "Active" for every customer would achieve 66.6% accuracy while failing to prevent a single lost account.
- **Solution**:
  - We used `class_weight="balanced"` in `RandomForestClassifier`, which inversely weights sample classes proportional to class frequencies.
  - **Priority Metric**: **Recall (Sensitivity)** on the churn class. In retention marketing, the cost of a **False Negative** (losing a high-value customer because the model failed to flag them) is vastly higher than the cost of a **False Positive** (sending a discount email to a customer who might have stayed anyway). Our model achieves **88.97% Recall** on test data.

### 4. SQL Window Functions & RFM Quintiles (`NTILE(5)`)
- Instead of arbitrary hardcoded dollar or order cutoffs, we used SQL window functions:
  ```sql
  NTILE(5) OVER (ORDER BY Recency ASC) AS R_Score_Raw,
  NTILE(5) OVER (ORDER BY Frequency ASC) AS F_Score,
  NTILE(5) OVER (ORDER BY Monetary ASC) AS M_Score
  ```
- **Why `NTILE(5)`?**: Distributes customers into 5 equal percentiles (quintiles), normalizing right-skewed e-commerce distributions and ensuring that segmentation scales smoothly as the database grows.

### 5. FastAPI Microservice Design Patterns
- **Asynchronous Non-Blocking Execution**: Built on ASGI (Uvicorn), capable of handling high-concurrency requests with low memory footprints.
- **In-Memory Lifespan Model Store**: Pre-loads `churn_model.joblib` during application startup via FastAPI's `@asynccontextmanager(app)` lifespan pattern. Inference requests execute in $\approx 2\text{–}5\text{ms}$ with zero disk I/O.
- **Pydantic Validation**: Protects model inference from invalid input types, non-positive order counts, and negative spend values, automatically responding with HTTP 422 error descriptions.

---

## 7. Comprehensive Interview Q&A (Cheat Sheet)

### Q1: "Can you give a 60-second elevator pitch of this project?"
> *"I designed and built an end-to-end customer churn and cohort retention system using 541k real-world e-commerce transactions. I engineered a pipeline that cleans raw sales data into an indexed SQLite database, runs SQL RFM window analytics to segment 4,338 customers, and trains a balanced Random Forest model that predicts customer churn with 86.5% accuracy, 89.0% churn recall, and an ROC-AUC of 0.9558. Finally, I productionized the model into an asynchronous FastAPI microservice with automated test suites and built an interactive executive dashboard."*

### Q2: "Why did you choose Random Forest over XGBoost, LightGBM, or Deep Learning?"
> *"For this dataset (~4,300 customers with 9 features), a tuned Random Forest provides the ideal balance between performance, robustness, and interpretability. Deep learning is prone to overfitting on tabular data of this scale. While XGBoost is competitive, Random Forest offers lower hyperparameter sensitivity, does not require feature scaling, naturally resists multicollinearity, and directly outputs reliable tree-based feature importances and calibrated class probabilities without heavy tuning."*

### Q3: "What were your most significant business findings from the cohort analysis?"
> *"First, our December 2010 acquisition cohort showed an exceptional long-term retention rate, dropping to 36.6% in Month 1 but rebounding to 50.3% in Month 11 during the holiday shopping season. Second, subsequent cohorts suffered an immediate ~75% drop between Month 0 and Month 1. This identified our primary business opportunity: introducing an automated 14-to-30-day post-first-purchase onboarding sequence to stem early drop-off."*

### Q4: "How did you translate model probabilities into operational business decisions?"
> *"Rather than simply returning a raw 0 or 1 label, the FastAPI service maps calibrated probabilities into three actionable risk tiers: High Risk (>70%), Medium Risk (30–70%), and Low Risk (<30%). For each tier, the API dynamically assesses lifetime spend to recommend interventions: high-spend high-risk accounts receive VIP account-manager outreach with 20% reactivation credits, medium-risk customers enter category-replenishment drip campaigns, and low-risk accounts are targeted for VIP loyalty expansion and upselling."*

### Q5: "How would you scale this pipeline to handle 100 million transactions?"
> *"1. **Data Layer**: Migrate from SQLite to a distributed columnar analytical warehouse like Snowflake, BigQuery, or Amazon Redshift, using dbt (data build tool) for scheduled transformation runs.*  
> *2. **ML Pipeline**: Transition model training to Apache Spark MLlib or distributed Ray/LightGBM clusters, versioning experiments and artifacts using MLflow.*  
> *3. **Inference Serving**: Containerize the FastAPI application via Docker, deploy across a Kubernetes cluster with an Application Load Balancer, and implement a Redis cache for customer feature vectors to ensure sub-millisecond p99 inference latency."*

---

## 8. Quickstart & Execution Guide

### Prerequisites
Clone the repository and install all dependencies:
```bash
pip install -r requirements.txt
```

### Option A: One-Click Master Pipeline
To execute all stages sequentially (data cleaning &rarr; SQL analytics &rarr; ML training &rarr; dashboard generation):
```bash
python run_pipeline.py
```
*Total execution time: $\approx 15\text{–}18$ seconds.*

### Option B: Run the FastAPI Real-Time Scoring Server
Launch the local Uvicorn development server with hot-reloading:
```bash
uvicorn app:app --reload
```
- **Service URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Example API Request via cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "frequency": 4,
    "monetary": 1797.24,
    "tenure": 359
  }'
```

#### Example Response (HTTP 200)
```json
{
  "churn_probability": 0.4532,
  "risk_level": "Medium Risk",
  "recommended_action": "Proactive Retention & Nurture: Customer activity is decelerating. Trigger personalized email campaign with replenishment reminders, curated category recommendations, and double reward points on the next purchase.",
  "predicted_churn": 0
}
```

#### Raw Transaction Upload Workflow
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@transactions.csv"
```

After upload, open `http://localhost:8000/dashboard` to view the live retention priority queue. Uploads require `customer_id`, `order_date`, `amount`, and `quantity`; `order_id`, `product_id`, and `country` are optional and receive server-side defaults. Engineered model features are generated server-side.

### Option C: View the Executive Dashboard
Open [`dashboard.html`](dashboard.html) directly in any browser:
```powershell
start dashboard.html
```

### Option D: Run with Docker
Build and run the API without the local Python environment:
```bash
docker build -t ecommerce-churn-api .
docker run --rm -p 8000:8000 ecommerce-churn-api
```

Runtime controls are available through environment variables:
- `CORS_ORIGINS`: comma-separated allowed origins; defaults to `*` for local development.
- `MAX_UPLOAD_BYTES`: maximum CSV upload size; defaults to 25 MB.
- `CHURN_MODEL_VERSION`: model label stored with health responses and simulated campaigns.
- `API_KEY`: optional shared key for protected API routes; leave unset for local development.
- `RATE_LIMIT_REQUESTS`: requests permitted per client within the window; defaults to 120.
- `RATE_LIMIT_WINDOW_SECONDS`: rate-limit window; defaults to 60 seconds.

The health endpoint exposes the model version and SHA-256 checksum so deployments can verify which artifact is serving predictions. Campaign records remain explicitly `SIMULATED` and now retain that model version for auditability. Uploaded transactions persist in `uploads.db`, allowing customer analysis and the dashboard priority queue to survive an API restart.

---

## 9. Automated Testing & Quality Assurance

The project includes two automated test suites:

### 1. API Endpoint Test Suite (`test_api.py`)
Validates model loading, `/health` response schemas, risk-tier inference, automatic secondary feature derivation, raw upload handling, customer lookup, bulk scoring, retention prioritization, simulated campaigns, and HTTP 422 input validation guards:
```bash
python test_api.py -v
```
*Result: 14 API tests passing in the current suite.*

### 2. Database & ML Pipeline Test Suite (`test_pipeline.py`)
Validates SQLite database integrity (record counts, non-negativity), RFM CSV outputs, cohort retention rate logic (Month 0 = 100%), and model inference thresholds:
```bash
python test_pipeline.py -v
```
*Result: 4 pipeline tests passing in the current suite.*

### Continuous Integration
GitHub Actions runs Python compilation and both test suites on every push and pull request using [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## 10. License
This project is licensed under the MIT License.
