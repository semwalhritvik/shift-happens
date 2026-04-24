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

## Phase 2 Deliverable: Model Development & CI/CD Pipeline

We have built an end-to-end model training, validation, and deployment pipeline with automated CI/CD.

- **Location:** All scripts, tests, and configuration are located in the `Model-Development/` directory.
- **Key Features:** Trains and compares Logistic Regression vs LightGBM, selects best model by ROC-AUC, tunes hyperparameters with RandomizedSearchCV, validates against performance thresholds, detects bias using Fairlearn MetricFrame, and pushes to GCS model registry with automatic rollback protection.
- **CI/CD:** Google Cloud Build trigger connected to GitHub automatically runs the full pipeline on every push to `main`. Training data is stored in GCS and downloaded at build time. Email notifications alert on pipeline failures.

Please navigate to the `Model-Development/README.md` for detailed instructions on reproducing the pipeline, viewing model comparison results, bias reports, and running the unit tests.

## Phase 3: Cloud Deployment & MLOps (ShiftHappens)

For the final phase of this project, the model transitions from a static artifact into a monitored, production-grade cloud deployment. We have architected an enterprise-level observability overlay called **ShiftHappens**, deployed entirely on Google Cloud Platform (GCP).

### Fulfillment of Deployment Requirements

1. **Cloud Architecture & Automation**: The entire infrastructure (Pub/Sub, BigQuery, Cloud Functions) is codified and deployed autonomously. 
2. **CI/CD Integration**: We utilize Google Cloud Build (`cloudbuild.yaml`) to automatically trigger deployments, build Docker containers, and push to Artifact Registry upon repository updates.
3. **Logs & Monitoring**: Instead of synchronous logging that blocks API responses, we built a custom Python SDK that asynchronously intercepts live prediction traffic and publishes JSON payloads to GCP Pub/Sub, which are ETL'd into BigQuery for persistent logging.
4. **Model Monitoring & Retraining**: A scheduled Cloud Function periodically calculates the Population Stability Index (PSI) against our baseline distribution. If macroeconomic data drift is detected, the ShiftHappens Streamlit dashboard alerts the team and provides a "1-Click Remediation" button to trigger a Vertex AI Kubeflow pipeline, dynamically retraining the model on the drifted data.

For detailed replication steps, infrastructure setup, and the monitoring dashboard codebase, see the [Model-Deployment/README.md](./Model-Deployment/README.md).

Live Demo Video: https://drive.google.com/file/d/1l286LsjyJEtIQsbqZbLs-_70lx6blfhf/view?usp=drivesdk