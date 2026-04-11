"""
main.py — FastAPI prediction endpoint for ShiftHappens.

Serves predictions from the final debiased LightGBM model.
After each prediction, publishes input features and result to
Google Pub/Sub via the ShiftHappens SDK for drift monitoring.

Endpoints:
    POST /predict  — runs inference on a single applicant record
    GET  /health   — returns API and model health status
    GET  /metrics  — returns current model performance metrics

Hosted on Cloud Run with Kubernetes managing auto-scaling.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictionRequest, PredictionResponse
from schemas import HealthResponse, MetricsResponse
from predictor import load_model, predict, MODEL_METRICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ─────────────────────────────────────────────────────────────
# FastAPI app initialisation
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       ="ShiftHappens Prediction API",
    description ="Credit default risk prediction with fairness constraints.",
    version     ="1.0.0"
)

# Allow Streamlit dashboard to call this API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_methods =["*"],
    allow_headers =["*"],
)

# Load model once at startup — not on every request
model = load_model()
logging.info("Model loaded. API ready.")


# ─────────────────────────────────────────────────────────────
# POST /predict — main prediction endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict_default(request: PredictionRequest):
    """
    Accepts a single loan applicant's features and returns:
        - prediction:  0 (no default) or 1 (default)
        - probability: likelihood of default (0.0 to 1.0)
        - risk_level:  LOW / MEDIUM / HIGH
        - model_name:  model used for inference

    After prediction, the ShiftHappens SDK publishes the input
    features and result to Pub/Sub asynchronously for drift monitoring.
    This does not block the prediction response.
    """
    try:
        # Run inference using final_model_debiased.pkl
        result = predict(model, request.dict())

        logging.info(
            f"Prediction: {result['prediction']} | "
            f"Probability: {result['probability']} | "
            f"Risk: {result['risk_level']}"
        )

        # SDK publishes to Pub/Sub here asynchronously
        # Import handled by sdk/shifthappens_sdk.py
        try:
            from sdk.shifthappens_sdk import publish_prediction
            publish_prediction(
                features   =request.dict(),
                prediction =result["prediction"],
                probability=result["probability"]
            )
        except ImportError:
            # SDK not available in local dev — prediction still returns
            logging.warning("SDK not available. Skipping Pub/Sub publish.")

        return PredictionResponse(**result)

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET /health — API health check
# ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API health status and loaded model name.
    Called by Cloud Run health checks and Streamlit dashboard.
    """
    return HealthResponse(
        status  ="healthy",
        model   ="LightGBM_debiased",
        version ="1.0.0"
    )


# ─────────────────────────────────────────────────────────────
# GET /metrics — current model performance metrics
# ─────────────────────────────────────────────────────────────
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Returns model performance metrics from the validation run.
    Shown on the consultancy dashboard alongside drift metrics.
    """
    return MetricsResponse(**MODEL_METRICS)


# ─────────────────────────────────────────────────────────────
# Run locally for development and testing
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
