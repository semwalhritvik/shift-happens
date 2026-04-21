# ShiftHappens: Enterprise MLOps & Deployment Architecture

This directory contains the production infrastructure, deployment pipelines, and observability platform for the Home Credit Default Risk model.

## Architecture Overview

We do not force clients to rewrite their backend to accommodate our model. Instead, we provide an asynchronous SDK that wraps their existing prediction API. 

1. **The Client API**: Simulates a live production environment making inferences.
2. **ShiftHappens SDK**: Intercepts features and predictions, firing them asynchronously to GCP Pub/Sub (zero latency added to the user application).
3. **Ingestion & Storage**: A GCP Cloud Function triggered by Pub/Sub cleans the logs and streams them into BigQuery.
4. **The Control Room**: A serverless Streamlit Dashboard (Cloud Run) visualizes data drift via PSI calculation.
5. **Automated Remediation**: If drift is detected, the UI triggers a Vertex AI pipeline to retrain and deploy a new model artifact.

## Directory Structure

* `/SDK/observability-sdk/`: The installable Python package that clients integrate into their API for asynchronous log batching.
* `/cloud_function/`: The serverless ETL pipeline linking Pub/Sub to BigQuery.
* `/app/`: The containerized Streamlit monitoring dashboard.
* `/Vertex/`: The Kubeflow pipeline definition for automated model retraining.
* `/api/`: Mock client backend to simulate live user traffic and test SDK integration.

## Replication & Deployment Steps

To replicate this deployment on a fresh GCP environment, execute the following sequence.

### Step 1: Infrastructure as Code (Terraform)
*Note: Ensure your GCP CLI is authenticated and a project is selected.*
1. Navigate to the Terraform directory (if extracted to `/infra`).
2. Run `terraform init` to initialize the provider.
3. Run `terraform apply -auto-approve` to provision the Pub/Sub topic, BigQuery datasets, and Cloud Storage buckets.

### Step 2: Install the SDK
1. Navigate to `SDK/observability-sdk/`.
2. Run `pip install -e .` to install the ShiftHappens tracking library locally.

### Step 3: Deploy the Cloud Function
Deploy the ETL bridge to move logs from Pub/Sub to BigQuery:
```bash
gcloud functions deploy shift-happens-etl \
  --runtime python310 \
  --trigger-topic prediction_logs \
  --entry-point main \
  --source ./cloud_function