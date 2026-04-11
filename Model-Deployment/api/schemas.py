"""
schemas.py — Pydantic request and response schemas for the FastAPI endpoint.

Defines the exact input shape expected from the Streamlit dashboard
and simulation scripts, and the output shape returned to the caller.
"""

from pydantic import BaseModel
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Input schema for a single loan applicant prediction request.
    All fields mirror the columns in application_train_merged.pkl.
    Optional fields default to None and are handled by median imputation
    in the preprocessing step.
    """
    # Core financial features — most predictive per SHAP analysis
    AMT_INCOME_TOTAL:   Optional[float] = None
    AMT_CREDIT:         Optional[float] = None
    AMT_ANNUITY:        Optional[float] = None
    AMT_GOODS_PRICE:    Optional[float] = None

    # Employment and demographic features
    DAYS_BIRTH:         Optional[float] = None
    DAYS_EMPLOYED:      Optional[float] = None
    DAYS_REGISTRATION:  Optional[float] = None
    DAYS_ID_PUBLISH:    Optional[float] = None

    # External credit scores — top SHAP features
    EXT_SOURCE_1:       Optional[float] = None
    EXT_SOURCE_2:       Optional[float] = None
    EXT_SOURCE_3:       Optional[float] = None

    # Categorical features
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER:        Optional[str] = None
    FLAG_OWN_CAR:       Optional[str] = None
    FLAG_OWN_REALTY:    Optional[str] = None
    NAME_INCOME_TYPE:   Optional[str] = None
    NAME_EDUCATION_TYPE:Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE:  Optional[str] = None

    # Additional numeric features
    CNT_CHILDREN:       Optional[float] = None
    CNT_FAM_MEMBERS:    Optional[float] = None
    REGION_RATING_CLIENT: Optional[float] = None


class PredictionResponse(BaseModel):
    """
    Output schema returned to Streamlit and simulation scripts
    after each prediction.
    """
    prediction:  int    # 0 = no default, 1 = default
    probability: float  # probability of default (0.0 to 1.0)
    risk_level:  str    # LOW / MEDIUM / HIGH — derived from probability
    model_name:  str    # name of model used for prediction


class HealthResponse(BaseModel):
    """Health check response schema."""
    status:     str   # healthy / degraded
    model:      str   # model name loaded
    version:    str   # model version


class MetricsResponse(BaseModel):
    """Current model performance metrics response schema."""
    roc_auc:   float
    f1_score:  float
    accuracy:  float
    model_name: str
