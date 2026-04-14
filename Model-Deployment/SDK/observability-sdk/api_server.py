import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gcp_observability_sdk import ShiftHappensTracker
from gcp_observability_sdk.model_manager import ModelManager


class PredictionRequest(BaseModel):
    client_id: str
    features: dict
    model_version: str = "v1"
    source_system: str = "api"


class PredictionResponse(BaseModel):
    client_id: str
    model_version: str
    prediction: str
    prediction_probability: float | None = None
    latency_ms: float
    status_code: int


def get_env_setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


app = FastAPI(
    title="ShiftHappens Prediction API",
    description="A lightweight prediction API that publishes prediction events to Pub/Sub.",
    version="1.0.0",
)

tracker: ShiftHappensTracker | None = None
model_manager: ModelManager | None = None


@app.on_event("startup")
def startup_event():
    global tracker, model_manager
    if tracker is not None and model_manager is not None:
        return

    project_id = get_env_setting("SHIFTHAPPENS_GCP_PROJECT")
    topic_id = get_env_setting("SHIFTHAPPENS_PUBSUB_TOPIC")
    model_path = os.environ.get("SHIFTHAPPENS_MODEL_PATH", "model.pkl")

    tracker = ShiftHappensTracker(project_id=project_id, topic_id=topic_id)
    model_manager = ModelManager(model_path=model_path)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if tracker is None or model_manager is None:
        raise HTTPException(status_code=500, detail="Server is not initialized")

    if not request.features:
        raise HTTPException(status_code=400, detail="features cannot be empty")

    try:
        input_df = pd.DataFrame([request.features])
        predictions, latency_ms, proba = model_manager.predict_with_proba(input_df)
        prediction_value = str(predictions[0]) if len(predictions) > 0 else ""
        probability_value = float(proba[0]) if proba is not None and len(proba) > 0 else None

        tracker.track_prediction(
            features=request.features,
            prediction=prediction_value,
            model_version=request.model_version,
            client_id=request.client_id,
            prediction_probability=probability_value,
            source_system=request.source_system,
        )

        return PredictionResponse(
            client_id=request.client_id,
            model_version=request.model_version,
            prediction=prediction_value,
            prediction_probability=probability_value,
            latency_ms=latency_ms,
            status_code=200,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.environ.get("SHIFTHAPPENS_API_PORT", 8000)),
        reload=False,
    )
