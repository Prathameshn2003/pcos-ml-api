from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os
import warnings
import xgboost as xgb

# ---------------------------------
# APP INIT
# ---------------------------------
app = FastAPI(title="PCOS ML API")

# ---------------------------------
# CORS
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# PATHS
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------
# LOAD PICKLE MODELS (SAFE)
# ---------------------------------
def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"Model missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

rf_model = load_pickle("rf_model.pkl")
scaler = load_pickle("scaler.pkl")

# ---------------------------------
# LOAD XGBOOST BOOSTER (CRITICAL FIX)
# ---------------------------------
xgb_model = load_pickle("xgb_model.pkl")
xgb_booster = xgb_model.get_booster()  # ✅ SAFE, VERSION-INDEPENDENT

# ---------------------------------
# INPUT SCHEMA
# ---------------------------------
class PCOSInput(BaseModel):
    age: int
    weight: float
    bmi: float
    cycle: int
    cycle_length: int
    weight_gain: int
    hair_growth: int
    skin_darkening: int
    hair_loss: int
    pimples: int
    fast_food: int
    regular_exercise: int
    follicle_left: int
    follicle_right: int
    endometrium: float

# ---------------------------------
# HEALTH CHECK
# ---------------------------------
@app.get("/")
def health():
    return {"status": "PCOS ML API running"}

# ---------------------------------
# PREDICTION
# ---------------------------------
@app.post("/predict-pcos")
def predict_pcos(data: PCOSInput):
    try:
        X = np.array([[ 
            data.age,
            data.weight,
            data.bmi,
            data.cycle,
            data.cycle_length,
            data.weight_gain,
            data.hair_growth,
            data.skin_darkening,
            data.hair_loss,
            data.pimples,
            data.fast_food,
            data.regular_exercise,
            data.follicle_left,
            data.follicle_right,
            data.endometrium
        ]])

        # Ignore sklearn warnings safely
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = scaler.transform(X)

        # Random Forest prediction
        rf_pred = int(rf_model.predict(X_scaled)[0])

        # XGBoost prediction (SAFE)
        dmatrix = xgb.DMatrix(X_scaled)
        xgb_prob = float(xgb_booster.predict(dmatrix)[0])
        xgb_pred = 1 if xgb_prob > 0.5 else 0

        final_pred = 1 if (rf_pred + xgb_pred) >= 1 else 0

        # ---------------------------
        # Risk Scoring Logic
        # ---------------------------
        cycle_score = 1 if data.cycle == 0 else 0
        hormonal_score = (
            data.hair_growth +
            data.skin_darkening +
            data.hair_loss +
            data.pimples
        )
        ultrasound_score = 1 if (data.follicle_left + data.follicle_right) >= 10 else 0
        metabolic_score = 1 if data.bmi >= 25 else 0

        total_score = (
            2 * cycle_score +
            2 * ultrasound_score +
            hormonal_score +
            metabolic_score
        )

        if total_score >= 4:
            final_pred = 1

        if final_pred:
            risk_percentage = max(30, int((total_score / 9) * 100))
            risk_level = (
                "Low" if risk_percentage < 50
                else "Medium" if risk_percentage < 70
                else "High"
            )

            return {
                "pcos_detected": True,
                "risk_score": risk_percentage,
                "risk_level": risk_level
            }

        return {
            "pcos_detected": False,
            "risk_score": 0,
            "risk_level": "None"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
