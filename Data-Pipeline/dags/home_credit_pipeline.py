import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os
# Add Data-Pipeline root to path so the scripts package is resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from scripts import kaggle_download
from scripts import outlier_treatment
from scripts import table_aggregator
from scripts import bias_slicer

# Default args with failure email alerts configured
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1), # Use a past date for scheduling
    'email': ['alerts@example.com'],
    'email_on_failure': True,             # Email alert on failure
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG for daily execution
dag = DAG(
    'home_credit_risk_pipeline',
    default_args=default_args,
    description='A daily pipeline for Home Credit Default Risk data processing and evaluation.',
    schedule='@daily',
    catchup=False,
)

def extract_data(**kwargs):
    logging.info("Starting data extraction...")
    # Calls the module's Kaggle download function directly
    kaggle_download.download_and_extract_data()
    
    app_train_path = 'data/raw/application_train.csv'
    bureau_path = 'data/raw/bureau.csv'
    
    logging.info("Extracted data paths saved to XCom.")
    return {
        'app_train_path': app_train_path,
        'bureau_path': bureau_path
    }

def treat_outliers(**kwargs):
    ti = kwargs['ti']
    extracted_paths = ti.xcom_pull(task_ids='download_kaggle_data')
    
    if not extracted_paths or 'app_train_path' not in extracted_paths:
        raise ValueError("application_train_path not found in XCom.")
        
    app_train_path = extracted_paths['app_train_path']
    app_train_cleaned_path = 'data/processed/application_train_cleaned.pkl'
    
    logging.info("Cleaning outliers in DAYS_EMPLOYED...")
    outlier_treatment.process_and_save_data(input_path=app_train_path, output_path=app_train_cleaned_path)
    
    # Save processed path to XCom
    return app_train_cleaned_path

def agg_bureau(**kwargs):
    logging.info("Aggregating bureau table...")
    bureau_path = 'data/raw/bureau.csv'
    output_path = 'data/processed/bureau_agg.pkl'
    return table_aggregator.aggregate_bureau(bureau_path=bureau_path, output_path=output_path)

def agg_prev(**kwargs):
    logging.info("Aggregating previous applications table...")
    prev_app_path = 'data/raw/previous_application.csv'
    output_path = 'data/processed/prev_app_agg.pkl'
    return table_aggregator.aggregate_previous_applications(prev_app_path=prev_app_path, output_path=output_path)

def agg_installments(**kwargs):
    logging.info("Aggregating installments table...")
    installments_path = 'data/raw/installments_payments.csv'
    output_path = 'data/processed/installments_agg.pkl'
    return table_aggregator.aggregate_installments(installments_path=installments_path, output_path=output_path)

def merge_features_call(**kwargs):
    ti = kwargs['ti']
    app_train_cleaned_path = ti.xcom_pull(task_ids='handle_outliers')
    bureau_agg_path = ti.xcom_pull(task_ids='task_agg_bureau')
    prev_app_agg_path = ti.xcom_pull(task_ids='task_agg_prev')
    installments_agg_path = ti.xcom_pull(task_ids='task_agg_installments')
    
    if not app_train_cleaned_path:
        raise ValueError("Path to train data not found in XCom.")
        
    merged_output_path = 'data/processed/application_train_merged.pkl'
    
    logging.info("Merging aggregated features with train table...")
    return table_aggregator.merge_features(
        app_train_path=app_train_cleaned_path,
        bureau_agg_path=bureau_agg_path,
        prev_app_agg_path=prev_app_agg_path,
        installments_agg_path=installments_agg_path,
        output_path=merged_output_path
    )

def slice_bias(**kwargs):
    ti = kwargs['ti']
    merged_output_path = ti.xcom_pull(task_ids='task_merge_features')
    
    if not merged_output_path:
        raise ValueError("Merged dataset path not found in XCom.")
    
    logging.info(f"Loading merged dataset from {merged_output_path} to evaluate bias...")
    df = pd.read_pickle(merged_output_path)
    
    # Usually you'd load a model and make actual predictions here. 
    # Since we lack a real model in this DAG, we'll mock the PREDICTION column safely.
    if 'PREDICTION' not in df.columns:
        logging.warning("No PREDICTION column found. Injecting standard synthetic mock predictions.")
        np.random.seed(42)
        df['PREDICTION'] = np.random.randint(0, 2, len(df))
    
    # Evaluate model using Fairlearn script
    bias_slicer.evaluate_bias(df=df)

# ---------------------
# Operator Definitions
# ---------------------

download_task = PythonOperator(
    task_id='download_kaggle_data',
    python_callable=extract_data,
    dag=dag,
)

task_outlier_treatment = PythonOperator(
    task_id='handle_outliers',
    python_callable=treat_outliers,
    dag=dag,
)

task_agg_bureau = PythonOperator(
    task_id='task_agg_bureau',
    python_callable=agg_bureau,
    dag=dag,
)

task_agg_prev = PythonOperator(
    task_id='task_agg_prev',
    python_callable=agg_prev,
    dag=dag,
)

task_agg_installments = PythonOperator(
    task_id='task_agg_installments',
    python_callable=agg_installments,
    dag=dag,
)

task_merge_features = PythonOperator(
    task_id='task_merge_features',
    python_callable=merge_features_call,
    dag=dag,
)

task_bias_slicer = PythonOperator(
    task_id='task_bias_slicer',
    python_callable=slice_bias,
    dag=dag,
)

# Dependencies
download_task >> task_outlier_treatment
task_outlier_treatment >> [task_agg_bureau, task_agg_prev, task_agg_installments] >> task_merge_features >> task_bias_slicer
