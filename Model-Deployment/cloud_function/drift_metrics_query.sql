-- Replace PROJECT_ID and DATASET with your actual values.
-- This query should be configured as a BigQuery scheduled query or executed manually as needed.

CREATE OR REPLACE TABLE `PROJECT_ID.DATASET.drift_metrics_daily` AS
WITH training AS (
  SELECT 'AMT_INCOME_TOTAL' AS feature_name, AMT_INCOME_TOTAL AS value
  FROM `PROJECT_ID.DATASET.training_baseline` WHERE AMT_INCOME_TOTAL IS NOT NULL
  UNION ALL
  SELECT 'AMT_CREDIT', AMT_CREDIT FROM `PROJECT_ID.DATASET.training_baseline` WHERE AMT_CREDIT IS NOT NULL
  UNION ALL
  SELECT 'AMT_ANNUITY', AMT_ANNUITY FROM `PROJECT_ID.DATASET.training_baseline` WHERE AMT_ANNUITY IS NOT NULL
  UNION ALL
  SELECT 'DAYS_BIRTH', DAYS_BIRTH FROM `PROJECT_ID.DATASET.training_baseline` WHERE DAYS_BIRTH IS NOT NULL
  UNION ALL
  SELECT 'DAYS_EMPLOYED', DAYS_EMPLOYED FROM `PROJECT_ID.DATASET.training_baseline` WHERE DAYS_EMPLOYED IS NOT NULL
),
live AS (
  SELECT 'AMT_INCOME_TOTAL' AS feature_name, AMT_INCOME_TOTAL AS value
  FROM `PROJECT_ID.DATASET.prediction_logs`
  WHERE AMT_INCOME_TOTAL IS NOT NULL AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  UNION ALL
  SELECT 'AMT_CREDIT', AMT_CREDIT FROM `PROJECT_ID.DATASET.prediction_logs`
  WHERE AMT_CREDIT IS NOT NULL AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  UNION ALL
  SELECT 'AMT_ANNUITY', AMT_ANNUITY FROM `PROJECT_ID.DATASET.prediction_logs`
  WHERE AMT_ANNUITY IS NOT NULL AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  UNION ALL
  SELECT 'DAYS_BIRTH', DAYS_BIRTH FROM `PROJECT_ID.DATASET.prediction_logs`
  WHERE DAYS_BIRTH IS NOT NULL AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  UNION ALL
  SELECT 'DAYS_EMPLOYED', DAYS_EMPLOYED FROM `PROJECT_ID.DATASET.prediction_logs`
  WHERE DAYS_EMPLOYED IS NOT NULL AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
),
feature_quantiles AS (
  SELECT 
    feature_name,
    APPROX_QUANTILES(value, 11) AS boundaries
  FROM training
  GROUP BY feature_name
),
training_hist AS (
  SELECT
    q.feature_name,
    bucket_id,
    SUM(CASE
      WHEN t.value IS NOT NULL
        AND t.value >= q.boundaries[OFFSET(bucket_id)]
        AND (
          (bucket_id < 9 AND t.value < q.boundaries[OFFSET(bucket_id + 1)])
          OR (bucket_id = 9 AND t.value <= q.boundaries[OFFSET(10)])
        )
      THEN 1
      ELSE 0
    END) AS training_count
  FROM feature_quantiles q
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, 9)) AS bucket_id
  LEFT JOIN training t USING(feature_name)
  GROUP BY q.feature_name, bucket_id
),
live_hist AS (
  SELECT
    q.feature_name,
    bucket_id,
    SUM(CASE
      WHEN l.value IS NOT NULL
        AND l.value >= q.boundaries[OFFSET(bucket_id)]
        AND (
          (bucket_id < 9 AND l.value < q.boundaries[OFFSET(bucket_id + 1)])
          OR (bucket_id = 9 AND l.value <= q.boundaries[OFFSET(10)])
        )
      THEN 1
      ELSE 0
    END) AS live_count
  FROM feature_quantiles q
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, 9)) AS bucket_id
  LEFT JOIN live l USING(feature_name)
  GROUP BY q.feature_name, bucket_id
),
training_summary AS (
  SELECT
    feature_name,
    bucket_id,
    training_count,
    (training_count + 1) / (SUM(training_count) OVER(PARTITION BY feature_name) + 10) AS training_pct
  FROM training_hist
),
live_summary AS (
  SELECT
    feature_name,
    bucket_id,
    live_count,
    (live_count + 1) / (SUM(live_count) OVER(PARTITION BY feature_name) + 10) AS live_pct
  FROM live_hist
),
psi_calc AS (
  SELECT
    t.feature_name,
    SUM((l.live_pct - t.training_pct) * LOG(l.live_pct / t.training_pct)) AS psi,
    SUM(t.training_count) AS training_total,
    SUM(l.live_count) AS live_total,
    STRING_AGG(CAST(t.training_count AS STRING) ORDER BY t.bucket_id) AS training_buckets,
    STRING_AGG(CAST(l.live_count AS STRING) ORDER BY l.bucket_id) AS live_buckets
  FROM training_summary t
  JOIN live_summary l
    USING(feature_name, bucket_id)
  GROUP BY t.feature_name
)
SELECT
  CURRENT_DATE() AS date,
  feature_name,
  SAFE_CAST(psi AS FLOAT64) AS psi,
  training_total,
  live_total,
  training_buckets,
  live_buckets,
  CURRENT_TIMESTAMP() AS created_at
FROM psi_calc;
