# predictor.py
import pandas as pd
import joblib
from data_transformer import transform_data

def load_model(model_path='mobile_price_prediction_model.joblib'):
    """
    Load the trained model from disk.
    """
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None

def predict_price_range(feature_dict, model_path='mobile_price_prediction_model.joblib'):
    """
    Predict price range for a single mobile phone.
    """
    # Convert input to DataFrame
    df = pd.DataFrame([feature_dict])
    
    # Transform the data
    transformed_df = transform_data(df)
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        return None
    
    # Make prediction
    prediction = model.predict(transformed_df)
    return int(prediction[0])

# Example usage
test_phone = {
    'phone_id': 1,
    'battery_power': 1500,
    'clock_speed': 2.2,
    'm_dep': 0.7,
    'mobile_wt': 175,
    'n_cores': 4,
    'ram': 4096,
    'talk_time': 15,
    'px_height': 1080,
    'px_width': 1920,
    'sc_h': 12,
    'sc_w': 6,
    'front_camera': 8,
    'primary_camera': 16,
    'blue': 1,
    'dual_sim': 1,
    'four_g': 1,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1,
    'int_memory': 64
}