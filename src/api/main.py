import mlflow
import pandas as pd
import sys
import os
from fastapi import FastAPI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.api.pydantic_models import CustomerInput, PredictionResponse

# =========================
# LOAD MODEL FROM MLFLOW
# =========================

# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# model_name = "CreditRiskModel"

# model = mlflow.pyfunc.load_model(
#     f"models:/{model_name}/latest"
# )

mlflow.sklearn.log_model(..., registered_model_name="CreditRiskModel")

# =========================
# FASTAPI APP
# =========================

app = FastAPI(title="Credit Risk API", version="1.0")


@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerInput):

    df = pd.DataFrame([data.dict()])

    prob = model.predict(df)[0]

    prediction = 1 if prob >= 0.5 else 0

    return PredictionResponse(
        risk_probability=float(prob),
        is_high_risk=prediction
    )