from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

# ---------------------------------
# APP INIT
# ---------------------------------
app = FastAPI(title="PCOS ML API")

# ---------------------------------
# CORS (REQUIRED FOR REACT)
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# LOAD MODELS
# ---------------------------------
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ---------------------------------
# INPUT SCHEMA (MATCH train.py)
# ---------------------------------
class PCOSInput(BaseModel):
    age: int
    weight: float
    bmi: float
    cycle: int                 # 0 = irregular, 1 = regular
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
# PREDICTION ENDPOINT
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

    # ------------------------------
    # RISK SCORING (SAME LOGIC)
    # ------------------------------
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

    # ------------------------------
    # RESPONSE
    # ------------------------------
    if final_pred == 1:
        risk_percentage = int((total_score / 9) * 100)
        risk_percentage = max(30, risk_percentage)

        if risk_percentage < 50:
            level = "Low"
        elif risk_percentage < 70:
            level = "Medium"
        else:
            level = "High"

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
