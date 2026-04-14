from setuptools import setup, find_packages

setup(
    name="gcp_observability_sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigquery",
        "pandas",
        "scikit-learn",
        "joblib",
        "pyarrow",
    ],
    python_requires=">=3.9",
    description="A lightweight SDK for GCP observability with BigQuery model logging and monitoring.",
    author="Observability SDK",
    license="Apache-2.0",
)
