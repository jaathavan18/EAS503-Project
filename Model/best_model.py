import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import os
from mlflow.models.signature import infer_signature

# Connect to the database and load the dataset
conn = sqlite3.connect('mobile_phones.db')
query = """
SELECT 
    p.phone_id,
    p.battery_power,
    p.clock_speed,
    p.m_dep,
    p.mobile_wt,
    p.n_cores,
    p.ram,
    p.talk_time,
    p.price_range,
    s.px_height,
    s.px_width,
    s.sc_h,
    s.sc_w,
    c.fc as front_camera,
    c.pc as primary_camera,
    f.blue,
    f.dual_sim,
    f.four_g,
    f.three_g,
    f.touch_screen,
    f.wifi,
    st.int_memory
FROM phones p
JOIN screen_specs s ON p.phone_id = s.phone_id
JOIN camera_specs c ON p.phone_id = c.phone_id
JOIN phone_features f ON p.phone_id = f.phone_id
JOIN storage_specs st ON p.phone_id = st.phone_id
"""
df_db = pd.read_sql_query(query, conn)
conn.close()

# Preprocess the data
cleaned_data = df_db.copy()

# Combine and drop redundant features
cleaned_data['total_camera'] = cleaned_data['front_camera'] + cleaned_data['primary_camera']
cleaned_data['total_pixels'] = cleaned_data['px_height'] * cleaned_data['px_width']
redundant_features = ['front_camera', 'primary_camera', 'px_height', 'px_width']
cleaned_data.drop(columns=redundant_features, inplace=True)

# Handle zeros in 'sc_w' by imputing with the mean
sc_w_mean = cleaned_data.loc[cleaned_data['sc_w'] != 0, 'sc_w'].mean()
cleaned_data['sc_w'] = cleaned_data['sc_w'].replace(0, sc_w_mean)

# Drop unused feature 'three_g'
cleaned_data.drop(columns=['three_g'], inplace=True)

# Normalize features
scaler = StandardScaler()
scaled_features = ['battery_power', 'ram', 'int_memory']
cleaned_data[scaled_features] = scaler.fit_transform(cleaned_data[scaled_features])

# Split the data into training and testing sets


# Adjust the dataset to exclude total_camera and total_pixels
X = cleaned_data.drop(columns=['price_range', 'phone_id', 'total_camera', 'total_pixels'])
y = cleaned_data['price_range']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "XGBoost Classifier": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Train, evaluate, and log models
results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Store results
    results[name] = {"Accuracy": accuracy, "Report": report}
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    # Track the best model
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
        best_model_name = name

# Save the best model
model_filename = f"{best_model_name.replace(' ', '_').lower()}_model_ultimate.pkl"
joblib.dump(best_model, model_filename)
print(f"Saved the best model: {best_model_name} with accuracy {best_accuracy}")

# Set up MLFlow
MLFLOW_TRACKING_URI = "https://dagshub.com/jaathavan18/Mobile_Price_Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Mobile_Price_Prediction")

# Log models to MLFlow
for name, model in models.items():
    with mlflow.start_run():
        metrics = {"accuracy": results[name]["Accuracy"]}
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=name
        )
        print(f"Logged model: {name}")

print("All models and metrics have been logged to MLFlow.")


import json

# Select a few rows from the test set
test_cases = X_test.sample(5, random_state=42)  # Pick 5 random samples for testing

# Convert the test cases to JSON format
test_cases_json = test_cases.to_dict(orient='records')

# Save the test cases to a file for later use
test_cases_file = "test_cases.json"
with open(test_cases_file, "w") as f:
    json.dump(test_cases_json, f, indent=4)

print(f"Test cases saved to {test_cases_file}")