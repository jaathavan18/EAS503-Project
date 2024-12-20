from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model pipeline
model = joblib.load("../../Model/random_forest_pca_full_pipeline.joblib")

# Initialize FastAPI app
fas_app = FastAPI()

# Define the schema for incoming request data
class PhoneData(BaseModel):
    battery_power: float
    clock_speed: float
    m_dep: float
    mobile_wt: int
    n_cores: int
    ram: float
    talk_time: int
    sc_h: int
    sc_w: float
    blue: int
    dual_sim: int
    four_g: int
    touch_screen: int
    wifi: int
    int_memory: float

# Root endpoint
@fas_app.get("/")
def read_root():
    return {"message": "Welcome to the Phone Price Prediction API"}

# Prediction endpoint
@fas_app.post("/predict/")
def predict(data: PhoneData):
    # Convert incoming data to a pandas DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Make a prediction using the loaded model
    try:
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}