from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

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
# LOAD MODELS (SAFE PATH)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_model(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"❌ Model file missing: {path}")
    return pickle.load(open(path, "rb"))

rf_model = load_model("rf_model.pkl")
xgb_model = load_model("xgb_model.pkl")
knn_model = load_model("knn_model.pkl")
scaler = load_model("scaler.pkl")

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
    input_data = np.array([[
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

    X_scaled = scaler.transform(input_data)

    rf_pred = rf_model.predict(X_scaled)[0]
    xgb_pred = xgb_model.predict(X_scaled)[0]

    final_pred = 1 if (rf_pred + xgb_pred) >= 1 else 0

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

    if final_pred == 1:
        risk_percentage = max(30, int((total_score / 9) * 100))
        level = (
            "Low" if risk_percentage < 50
            else "Medium" if risk_percentage < 70
            else "High"
        )

        return {
            "pcos_detected": True,
            "risk_score": risk_percentage,
            "risk_level": level
        }

    return {
        "pcos_detected": False,
        "risk_score": 0,
        "risk_level": "None"
    }
