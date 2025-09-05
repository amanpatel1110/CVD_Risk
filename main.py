from fastapi import FastAPI
# from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from pydantic import BaseModel, conint, confloat
from enum import Enum

class Binary(int, Enum):
    NO = 0
    YES = 1

class InputData(BaseModel):
    Age: conint(ge=20, le=80)  # integer 20–80
    Sex: Binary

    SBP: float
    DBP: float

    Antihypertensive_use: Binary
    TC_HDL_Ratio: float

    Statin_use: Binary

    Diabetes_status: conint(ge=0, le=2)  # {0,1,2}
    Antidiabetic_medication: conint(ge=0, le=2)  # {0,1,2}

    Smoking: conint(ge=0, le=20)  # integer 0–20
    Physical_activity: int

    ECG_abnormality: Binary
    BMI: float
    Waist_to_Hip_Ratio: float

    Prior_CVD: Binary
    Family_History_CAD: Binary


# Load scaler + model

scaler = joblib.load("scaler.pkl")   
ann_model = tf.keras.models.load_model("ann_model_100k.keras")


# Expected input

numeric_cols = [
    "Age", "Sex", "SBP", "DBP", "Antihypertensive_use", "TC_HDL_Ratio",
    "Statin_use", "Diabetes_status", "Antidiabetic_medication", "Smoking",
    "Physical_activity", "ECG_abnormality", "BMI", "Waist_to_Hip_Ratio",
    "Prior_CVD", "Family_History_CAD"
]

app = FastAPI(title="CVD Risk Predictor (New Model)")



@app.get("/")
def home():
    return {"message": "CVD Risk Prediction API is running!"}


@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    # Apply StandardScaler
    X_scaled = scaler.transform(df[numeric_cols])

    # Predict
    prediction = ann_model.predict(X_scaled)
    score = float(prediction[0][0]) 
    print(score)
    return {"predicted_cvd_risk_score": score}
