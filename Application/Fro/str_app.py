import math
import streamlit as st
import requests
import os
import json

# Initialize user options dictionary
user_options = {}

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the JSON file with slider options
file_path = os.path.join(current_dir, "streamlit_options.json")
print("Looking for:", file_path)

# Load slider options from JSON file
try:
    with open(file_path, "r") as f:
        StreamLit_SlideBar = json.load(f)
except FileNotFoundError:
    st.error(f"Slider options JSON file not found at: {file_path}")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error decoding JSON file: {e}")
    st.stop()

# Streamlit application title
st.title('Phone Price Prediction System')

# Exclude unnecessary fields
excluded_fields = ["phone_id", "price_range"]

# Create sliders in the sidebar for user input
for field_name, range_values in StreamLit_SlideBar["slider_fields"].items():
    if field_name not in excluded_fields:
        min_val, max_val = range_values
        current_value = round((min_val + max_val) / 2)
        user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value=current_value)

# Compute derived features
if "front_camera" in user_options and "primary_camera" in user_options:
    user_options["total_camera"] = user_options["front_camera"] + user_options["primary_camera"]
if "px_height" in user_options and "px_width" in user_options:
    user_options["total_pixels"] = user_options["px_height"] * user_options["px_width"]

# Define the required features based on FastAPI's PhoneData model
required_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram', 'talk_time',
    'sc_h', 'sc_w', 'blue', 'dual_sim', 'four_g', 'touch_screen', 'wifi', 'int_memory',
    'total_camera', 'total_pixels'
]

# Prediction button
if st.button('Predict'):
    # Filter user options to only include required features
    filtered_user_options = {key: value for key, value in user_options.items() if key in required_features}

    # Convert user options to JSON format
    data = json.dumps(filtered_user_options, indent=2)

    # Make a POST request to the prediction API
    try:
        api_url = 'http://127.0.0.1:8000/predict/'  # Ensure this matches your FastAPI endpoint
        headers = {"Content-Type": "application/json"}
        r = requests.post(api_url, data=data, headers=headers)
        r.raise_for_status()  # Raise an HTTPError if the response was unsuccessful

        # Display input and prediction results
        st.write("Input Data (Filtered):", filtered_user_options)
        st.write("Prediction Result:", r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the API request: {e}")