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

# Load the dataset
try:
    TestData = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"The dataset file was not found at: {csv_path}")
    st.stop()

# Clean the dataset: replace inf, -inf, and NaN with valid values
TestData = TestData.replace([float('inf'), float('-inf')], None).fillna(0)

# Dynamically get the slider fields from the dataset and exclude unwanted fields
excluded_fields = ["phone_id", "price_range"]
slider_fields = [field for field in TestData.columns if field not in excluded_fields]

# Initialize data structures for Streamlit field data and user options
streamlit_field_data = defaultdict(dict)
user_options = {}

# Streamlit application title
st.title('Phone Price Prediction System')

# Calculate min and max values for each slider field
for field in slider_fields:
    try:
        field_min = math.floor(TestData[field].min())
        field_max = math.ceil(TestData[field].max())
        streamlit_field_data["slider_fields"][field] = [field_min, field_max]
    except (KeyError, ValueError) as e:
        st.warning(f"Skipping field {field} due to invalid data: {e}")
        continue

# Path for the Streamlit slider options JSON file
streamlit_options_file = os.path.join(current_dir, "streamlit_options.json")

# Save slider options to a JSON file if it doesn't exist
if not os.path.exists(streamlit_options_file):
    with open(streamlit_options_file, "w") as f:
        json.dump(streamlit_field_data, f, indent=2)

# Load slider options from the JSON file
try:
    with open(streamlit_options_file, "r") as f:
        StreamLit_SlideBar = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    st.error("The JSON file with slider options was not found or is invalid.")
    st.stop()

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
        api_url = 'http://127.0.0.1:8000/predict/'  # Ensure this matches your FastAPI endpoint
        headers = {"Content-Type": "application/json"}
        r = requests.post(api_url, data=data, headers=headers)
        r.raise_for_status()  # Raise an error for unsuccessful requests
        st.write("Input Data:", user_options)
        st.write("Prediction Result:", r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the API request: {e}")