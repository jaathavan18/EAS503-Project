from typing import Union
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import requests

# Initialize the FastAPI app
app = FastAPI()

# Define input data schema using Pydantic
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
    total_pixels: int
    total_camera: int

# Load the saved model and scaler
model = joblib.load("xgb_pipeline_final.pkl")

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the XGB Model API",
        "usage": "Send POST requests to /predict/ with phone data to get predictions."
    }

# Prediction endpoint
@app.post("/predict/")
def predict(payload: PhoneData):
    # Convert payload to pandas DataFrame
    df = pd.DataFrame([payload.dict().values()], columns=payload.dict().keys())
    
    # Make predictions using the model
    prediction = model.predict(df)
    
    return {"prediction": int(prediction[0])}

# Simulated API test
if __name__ == "__main__":
    # Example test case
    test_case_5 = {
        "phone_id": 1,
        "battery_power": 2500,
        "clock_speed": 2.5,
        "m_dep": 0.7,
        "mobile_wt": 180,
        "n_cores": 6,
        "ram": 6000,
        "talk_time": 15,
        "px_height": 1080,
        "px_width": 1920,
        "sc_h": 12,
        "sc_w": 6,
        "front_camera": 8,
        "primary_camera": 24,
        "blue": 1,
        "dual_sim": 1,
        "four_g": 1,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1,
        "int_memory": 64,
        "total_pixels": 2073600,
        "total_camera": 32
    }

    # Convert test case to JSON
    testing_input_data = json.dumps(test_case_5, indent=4)

    # Simulate an API request
    response = requests.post(
        "http://127.0.0.1:8000/predict/",
        data=testing_input_data,
        headers={"Content-Type": "application/json"}
    )
    print("Response:", response.json())