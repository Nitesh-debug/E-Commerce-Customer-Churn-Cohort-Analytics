-- =====================================================================
-- rfm_cohort_analysis.sql
-- E-Commerce Customer RFM Segmentation and Monthly Retention Cohort Analysis
-- Database: analytics.db (SQLite 3.25+)
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. RFM SEGMENTATION QUERY
-- Computes Recency, Frequency, Monetary metrics and scores via NTILE(5).
-- Segments customers into Champions, At-Risk, Churned, etc.
-- ---------------------------------------------------------------------

DROP VIEW IF EXISTS v_rfm_customer_segments;

CREATE VIEW v_rfm_customer_segments AS
WITH snapshot AS (
    -- Reference date is 1 day after the latest recorded transaction
    SELECT datetime(MAX(InvoiceDate), '+1 day') AS snapshot_date
    FROM transactions
),
customer_rfm_raw AS (
    SELECT
        t.CustomerID,
        t.Country,
        MIN(t.InvoiceDate) AS FirstPurchaseDate,
        MAX(t.InvoiceDate) AS LastPurchaseDate,
        CAST(ROUND(julianday(s.snapshot_date) - julianday(MAX(t.InvoiceDate))) AS INTEGER) AS Recency,
        COUNT(DISTINCT t.InvoiceNo) AS Frequency,
        ROUND(SUM(t.TotalAmount), 2) AS Monetary,
        COUNT(t.StockCode) AS TotalTransactions,
        SUM(t.Quantity) AS TotalUnits
    FROM transactions t
    CROSS JOIN snapshot s
    GROUP BY t.CustomerID
),
rfm_scores AS (
    SELECT
        CustomerID,
        Country,
        FirstPurchaseDate,
        LastPurchaseDate,
        Recency,
        Frequency,
        Monetary,
        TotalTransactions,
        TotalUnits,
        -- Higher score = better customer (5 is best)
        -- Lower recency (fewer days since last purchase) gets higher score
        NTILE(5) OVER (ORDER BY Recency DESC) AS R_Score,
        -- Higher frequency gets higher score
        NTILE(5) OVER (ORDER BY Frequency ASC) AS F_Score,
        -- Higher monetary spend gets higher score
        NTILE(5) OVER (ORDER BY Monetary ASC) AS M_Score
    FROM customer_rfm_raw
),
rfm_segmented AS (
    SELECT
        CustomerID,
        Country,
        FirstPurchaseDate,
        LastPurchaseDate,
        Recency,
        Frequency,
        Monetary,
        TotalTransactions,
        TotalUnits,
        R_Score,
        F_Score,
        M_Score,
        (R_Score || '-' || F_Score || '-' || M_Score) AS RFM_Cell,
        ROUND((R_Score + F_Score + M_Score) / 3.0, 2) AS RFM_Avg_Score,
        CASE
            -- Champions: Bought recently, buy often, and spend the most
            WHEN R_Score >= 4 AND F_Score >= 4 THEN 'Champions'
            -- Loyal Customers: Buy regularly, responsive to promotions
            WHEN R_Score >= 3 AND F_Score >= 3 THEN 'Loyal Customers'
            -- Potential Loyalists: Recent buyers with low frequency
            WHEN R_Score >= 4 AND F_Score <= 2 THEN 'Potential Loyalists'
            -- Promising: Moderate recency, new buyers
            WHEN R_Score = 3 AND F_Score = 1 THEN 'Promising'
            -- Need Attention: Moderate recency and frequency
            WHEN R_Score = 3 AND F_Score = 2 THEN 'Need Attention'
            -- At-Risk: Used to buy frequently/high spend, but haven't returned recently
            WHEN R_Score <= 2 AND (F_Score >= 3 OR M_Score >= 3) THEN 'At-Risk'
            -- Hibernating: Low recency, low spend and frequency
            WHEN R_Score = 2 AND F_Score <= 2 THEN 'About To Sleep / Hibernating'
            -- Churned / Lost: Lowest recency and lowest engagement
            WHEN R_Score = 1 AND F_Score <= 2 THEN 'Churned'
            ELSE 'Others / Inactive'
        END AS Customer_Segment
    FROM rfm_scores
)
SELECT * FROM rfm_segmented;


-- ---------------------------------------------------------------------
-- 2. MONTHLY RETENTION COHORT ANALYSIS QUERY
-- Determines each customer's first purchase month (cohort),
-- tracks subsequent active months, and computes retention rates.
-- ---------------------------------------------------------------------

DROP VIEW IF EXISTS v_cohort_retention;

CREATE VIEW v_cohort_retention AS
WITH customer_cohort AS (
    -- Customer's first purchase month
    SELECT
        CustomerID,
        strftime('%Y-%m', MIN(InvoiceDate)) AS CohortMonth
    FROM transactions
    GROUP BY CustomerID
),
customer_activities AS (
    -- Active months for each customer
    SELECT DISTINCT
        t.CustomerID,
        c.CohortMonth,
        strftime('%Y-%m', t.InvoiceDate) AS ActivityMonth
    FROM transactions t
    INNER JOIN customer_cohort c ON t.CustomerID = c.CustomerID
),
cohort_index_calc AS (
    -- Calculate month difference between ActivityMonth and CohortMonth
    SELECT
        CustomerID,
        CohortMonth,
        ActivityMonth,
        (
            (CAST(strftime('%Y', ActivityMonth || '-01') AS INTEGER) - CAST(strftime('%Y', CohortMonth || '-01') AS INTEGER)) * 12 +
            (CAST(strftime('%m', ActivityMonth || '-01') AS INTEGER) - CAST(strftime('%m', CohortMonth || '-01') AS INTEGER))
        ) AS CohortIndex
    FROM customer_activities
),
cohort_summary AS (
    SELECT
        CohortMonth,
        CohortIndex,
        COUNT(DISTINCT CustomerID) AS ActiveCustomers
    FROM cohort_index_calc
    GROUP BY CohortMonth, CohortIndex
),
cohort_sizes AS (
    SELECT
        CohortMonth,
        ActiveCustomers AS CohortSize
    FROM cohort_summary
    WHERE CohortIndex = 0
)
SELECT
    cs.CohortMonth,
    sz.CohortSize,
    cs.CohortIndex,
    cs.ActiveCustomers,
    ROUND((CAST(cs.ActiveCustomers AS REAL) / sz.CohortSize) * 100, 2) AS RetentionRatePercent
FROM cohort_summary cs
INNER JOIN cohort_sizes sz ON cs.CohortMonth = sz.CohortMonth
ORDER BY cs.CohortMonth ASC, cs.CohortIndex ASC;
