import math
import streamlit as st
import requests
import json

# Define the slider fields and their ranges manually
slider_fields = {
    'battery_power': (500, 2000, 1),
    'clock_speed': (0.5, 3.0, 0.1),
    'm_dep': (0.0, 1.0, 0.1),
    'mobile_wt': (80, 250, 1),
    'n_cores': (1, 8, 1),
    'ram': (256, 8192, 1),
    'talk_time': (2, 20, 1),
    'sc_h': (5, 20, 1),
    'sc_w': (0.0, 15.0, 0.1),
    'blue': (0, 1, 1),
    'dual_sim': (0, 1, 1),
    'four_g': (0, 1, 1),
    'touch_screen': (0, 1, 1),
    'wifi': (0, 1, 1),
    'int_memory': (2, 128, 1)
}

# Collect user inputs dynamically
st.title("Phone Price Prediction")
user_options = {}
for field_name, (min_val, max_val, step) in slider_fields.items():
    # Set default value as the midpoint
    current_value = round((min_val + max_val) / 2, 2) if isinstance(step, float) else (min_val + max_val) // 2
    user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value=current_value, step=step)

# Prediction button
if st.button('Predict'):
    data = json.dumps(user_options, indent=2)
    try:
        # Replace the endpoint URL with your FastAPI prediction endpoint
        r = requests.post('http://134.122.7.104:8002/predict/', data=data, headers={'Content-Type': 'application/json'})
        r.raise_for_status()  # Raise an error if the request failed
        st.write("Input Data:")
        st.write(user_options)
        st.write("Prediction Response:")
        st.write(r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")