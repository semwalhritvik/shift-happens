# ShiftHappens: MLOps Monitoring for SMEs

## Project Objective
ShiftHappens is a lightweight, serverless MLOps monitoring platform designed for small and medium enterprises (SMEs) and AI consultancies. It acts as an "early warning system" for deployed machine learning models, detecting data drift and performance degradation before it impacts client relationships. 

It answers three core questions for deployed models:
1. **Is it broken?** (Health monitoring)
2. **Who broke it?** (Drift detection)
3. **Can we fix it?** (One-click remediation)

## Current Status: Sprint 2 (Data Pipeline)
We are currently in the Data Ingestion & Preprocessing phase. We are utilizing the **Home Credit Default Risk** dataset to simulate a production credit scoring model. 

### Phase 1 Deliverable: Automated Airflow Pipeline
We have built a fully automated, test-driven ETL pipeline. 
* **Location:** All pipeline code, DAGs, Pytest modules, and execution logs are located in the [`Data-Pipeline/`](./Data-Pipeline) directory.
* **Key Features:** Features include DVC integration for data versioning, parallelized Airflow tasks for optimized feature engineering across 8 relational tables, targeted anomaly treatment (e.g., handling erroneous `DAYS_EMPLOYED` records), and Fairlearn integration for demographic bias mitigation.

Please navigate to the `Data-Pipeline/README.md` for detailed instructions on reproducing the Airflow environment, viewing the Gantt chart optimizations, and running the unit tests.