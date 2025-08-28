from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
from enum import Enum

# Load preprocessor and model
preprocessor = joblib.load("preprocessor_100k.pkl")
ann_model = tf.keras.models.load_model("ann_model_100k.keras")

# Define expected input features
categorical_cols = [
    'region','gender','diet','smoking_status','chewing_tobacco','alcohol',
    'diabetes_status','angina_or_heart_attack_in_a_1st_degree',
    'chronic_kidney_disease','on_blood_pressure_treatment'
]
numeric_cols = ['age','height','weight','bmi']

# FastAPI app
app = FastAPI(title="CVD Risk Predictor")

class Region(str, Enum):
    GJ = "GJ"
    KL = "KL"
    MH = "MH"
    RJ = "RJ"

class Gender(str, Enum):
    FEMALE = "FEMALE"
    MALE = "MALE"

class Diet(str,Enum):
    NON_VEGETARIAN = "NON_VEGETARIAN"
    VEGETARIAN = "VEGETARIAN"

class SmokingStatus(str, Enum):
    EX_SMOKER = "EX_SMOKER"
    HEAVY_SMOKER = "HEAVY_SMOKER"
    LIGHT_SMOKER = "LIGHT_SMOKER"
    MODERATE_SMOKER = "MODERATE_SMOKER"
    NON_SMOKER = "NON_SMOKER"

class DiabetesStatus(str, Enum):
    NONE = "NONE"
    TYPE_1 = "TYPE_1"
    TYPE_2 = "TYPE_2"

class YesNo(str, Enum):
    YES = "YES"
    NO = "NO"


# Pydantic model for request validation
class InputData(BaseModel):
    age: float
    height: float
    weight: float
    bmi: float
    region: Region
    gender: Gender
    diet: Diet
    smoking_status: SmokingStatus
    chewing_tobacco: YesNo
    alcohol: YesNo
    diabetes_status: DiabetesStatus
    angina_or_heart_attack_in_a_1st_degree: YesNo
    chronic_kidney_disease: YesNo
    on_blood_pressure_treatment: bool

@app.get("/")
def home():
    return {"message": "CVD Risk Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to dict -> DataFrame
    import pandas as pd
    df = pd.DataFrame([data.dict()])

    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    prediction = ann_model.predict(X_processed)
    score = float(prediction[0][0])  # Convert np.float32 → Python float

    return {"predicted_cvd_risk_score": score}
