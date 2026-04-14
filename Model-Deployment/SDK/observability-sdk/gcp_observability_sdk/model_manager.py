import joblib
import time
import numpy as np
import pandas as pd


class ModelManager:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, input_df: pd.DataFrame):
        start = time.time()
        predictions = self.model.predict(input_df)
        latency_ms = round((time.time() - start) * 1000, 2)
        return predictions, latency_ms

    def predict_proba(self, input_df: pd.DataFrame):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(input_df)
        return None

    def predict_with_proba(self, input_df: pd.DataFrame):
        start = time.time()
        predictions = self.model.predict(input_df)
        proba = self.predict_proba(input_df)
        if proba is not None and isinstance(proba, np.ndarray):
            if proba.ndim == 2 and proba.shape[1] > 1:
                proba = proba[:, 1]
            else:
                proba = proba.ravel()
        latency_ms = round((time.time() - start) * 1000, 2)
        return predictions, latency_ms, proba
