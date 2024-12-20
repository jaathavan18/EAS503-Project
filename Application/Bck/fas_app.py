from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../../Model/xgboost_classifier_model_ultimate.pkl")

# Initialize FastAPI app
fas_app = FastAPI()

# Define the schema for incoming request data
class PhoneData(BaseModel):
    phone_id: int
    battery_power: int
    clock_speed: float
    m_dep: float
    mobile_wt: int
    n_cores: int
    ram: int
    talk_time: int
    px_height: int
    px_width: int
    sc_h: int
    sc_w: int
    front_camera: int
    primary_camera: int
    blue: int
    dual_sim: int
    four_g: int
    three_g: int
    touch_screen: int
    wifi: int
    int_memory: int
    total_pixels: int  # Included in request but not used for prediction
    total_camera: int  # Included in request but not used for prediction

# Root endpoint
@fas_app.get("/")
def read_root():
    return {"message": "Welcome to the XGB Model API"}

# Prediction endpoint
@fas_app.post("/predict/")
def predict(data: PhoneData):
    # Convert incoming data to a pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Drop features not used by the model
    input_data = input_data.drop(columns=["total_pixels", "total_camera"])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction
    return {"prediction": int(prediction[0])}
