import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
import pandas as pd
import joblib

from src.features.feature_engineering import create_basic_features

LOG = logging.getLogger("uvicorn")

app = FastAPI(title="Shadowfox - Car Price Prediction API")

# Priority loading
MODEL_PATH1 = r"model\best_model.joblib"
TUNED_MODEL_PATH  = r"model\best_tuned_model.joblib"

MODEL = None
MODEL_PATH = None


@app.on_event("startup")
def load_model():
    global MODEL, MODEL_PATH

    # Priority: pruned model first
    if os.path.exists(MODEL_PATH1):
        MODEL_PATH = MODEL_PATH1
    elif os.path.exists(TUNED_MODEL_PATH):
        MODEL_PATH = TUNED_MODEL_PATH
    else:
        raise FileNotFoundError("❌ No model found (neither pruned nor tuned).")

    LOG.info(f"Loading model from {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
    LOG.info("Model loaded successfully.")


class CarInput(BaseModel):
    present_price: confloat(ge=0)
    kms_driven: conint(ge=0)
    year: conint(ge=1900, le=2100)
    fuel_type: str
    seller_type: str
    transmission: str
    owner: conint(ge=0)
    brand: str = "Unknown"


@app.post("/predict")
def predict_car(payload: CarInput):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Raw input → DataFrame
        df = pd.DataFrame([{
            "Present_Price": payload.present_price,
            "Kms_Driven": payload.kms_driven,
            "Year": payload.year,
            "Fuel_Type": payload.fuel_type,
            "Seller_Type": payload.seller_type,
            "Transmission": payload.transmission,
            "Owner": payload.owner,
            "Brand": payload.brand
        }])

        # Feature engineering
        df = create_basic_features(df)

        # Drop unused
        df = df.drop(columns=["Year", "Car_Name"], errors="ignore")

        # Prediction
        pred = MODEL.predict(df)[0]

        return {
            "predicted_price": float(pred),
            "model_used": "best_tuned_model.joblib"
        }

    except Exception as e:
        LOG.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL_PATH
    }
