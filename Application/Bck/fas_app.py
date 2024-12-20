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
    total_camera: int
    total_pixels: int

# Root endpoint
@fas_app.get("/")
def read_root():
    return {"message": "Welcome to the Phone Price Prediction API"}

# Prediction endpoint
@fas_app.post("/predict/")
def predict(data: PhoneData):
    # Convert incoming data to a pandas DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Ensure input matches model training features
    expected_features = [
        'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram', 'talk_time',
        'sc_h', 'sc_w', 'blue', 'dual_sim', 'four_g', 'touch_screen', 'wifi', 'int_memory',
        'total_camera', 'total_pixels'
    ]
    if not all(feature in input_data.columns for feature in expected_features):
        return {"error": f"Input data is missing required features: {expected_features}"}

    # Make a prediction using the loaded model
    try:
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}