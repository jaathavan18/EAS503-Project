import math
from collections import defaultdict
import pandas as pd
import streamlit as st
import requests
import json

import os

# Get the absolute path to the CSV file
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
csv_path = os.path.join(project_root, "Model", "phone_data.csv")

try:
    TestData = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"The dataset file was not found at: {csv_path}")
    st.write("Current directory:", current_dir)
    st.stop()

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
try:
    with open(streamlit_options_file, "r") as f:
        StreamLit_SlideBar = json.load(f)
except FileNotFoundError:
    st.error("The JSON file with slider options was not found. Please check the file path.")
    st.stop()
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