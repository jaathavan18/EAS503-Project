import math
from collections import defaultdict
import pandas as pd
import streamlit as st
import requests
import json

# Load sample phone dataset
TestData = pd.read_csv("../../Model/phone_data.csv")  # Update the path with the correct file location

# Define the fields for sliders
slider_fields = [
    'phone_id',
    'battery_power',
    'clock_speed',
    'm_dep',
    'mobile_wt',
    'n_cores',
    'ram',
    'talk_time',
    'px_height',
    'px_width',
    'sc_h',
    'sc_w',
    'front_camera',
    'primary_camera',
    'blue',
    'dual_sim',
    'four_g',
    'three_g',
    'touch_screen',
    'wifi',
    'int_memory',
    'total_pixels',
    'total_camera'
]

# Initialize data structures for Streamlit field data and user options
streamlit_field_data = defaultdict(dict)
user_options = {}

# Streamlit application title
st.title('Phone Price Prediction System')

# Calculate min and max values for each slider field
for field in slider_fields:
    streamlit_field_data["slider_fields"][field] = [
        math.floor(TestData[field].min()),
        math.ceil(TestData[field].max())
    ]

# Save slider options to a JSON file
streamlit_options_file = "../front_end/streamlit_options.json"
with open(streamlit_options_file, "w") as f:
    json.dump(streamlit_field_data, f, indent=2)

# Load slider options from the JSON file
StreamLit_SlideBar = json.load(open(streamlit_options_file))

# Create sliders in the sidebar for user input
for field_name, range_values in StreamLit_SlideBar["slider_fields"].items():
    min_val, max_val = range_values
    current_value = round((min_val + max_val) / 2)
    user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value=current_value)

# Button to trigger prediction
if st.button('Predict'):
    # Convert user options to JSON format
    data = json.dumps(user_options, indent=2)

    # Make a POST request to the prediction API
    try:
        r = requests.post('http://localhost:8002/predict', data=data, headers={"Content-Type": "application/json"})
        r.raise_for_status()  # Raise an error for unsuccessful requests
        st.write("Input Data:", user_options)
        st.write("Prediction Result:", r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")