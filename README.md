# ShiftHappens - Data Pipeline

## Overview
This repository contains the Airflow data pipeline for ShiftHappens. The pipeline automates the ingestion, preprocessing, validation, and bias detection for the Home Credit Default Risk dataset.

## Repository Structure
* `dags/`: Contains the Apache Airflow DAG definitions (`home_credit_pipeline.py`).
* `data/`: Local directory for storing raw and processed datasets (tracked via DVC).
* `scripts/`: Modular Python scripts for downloading, cleaning, feature engineering, and bias slicing.
* `tests/`: Unit tests (using pytest) to ensure robustness of preprocessing steps.
* `logs/`: Airflow and application execution logs.
* `dvc.yaml`: Configuration for Data Version Control.

## Environment Setup & Reproducibility
To replicate this pipeline on another machine, ensure you have Python 3.9+ installed and follow these steps:

Clone the repository:
```bash
git clone https://github.com/semwalhritvik/shift-happens.git
cd shift-happens/Data-Pipeline
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Pull versioned data with DVC:
This project uses DVC to track the dataset. To pull the exact data versions used in this pipeline:
```bash
dvc pull
```

## Running the Airflow Pipeline
This pipeline is orchestrated using Apache Airflow.

Initialize the Airflow database and setup:
```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@shifthappens.local
```

Start the Airflow Scheduler and Webserver:
Open two separate terminal windows.

Terminal 1:
```bash
airflow scheduler
```

Terminal 2:
```bash
airflow webserver -p 8080
```

Trigger the DAG:
Navigate to http://localhost:8080 in your browser, log in, unpause the `home_credit_pipeline` DAG, and trigger it manually.

## Data Schema & Anomalies
We utilize tools like TensorFlow Data Validation (TFDV) to automatically generate data schemas and catch anomalies (such as extreme values in `DAYS_EMPLOYED`). Alerts are configured to log errors appropriately.

## Bias Detection
We implement data slicing using the Fairlearn library to evaluate model performance across demographic subgroups to ensure fairness and address potential bias in the credit risk predictions.
