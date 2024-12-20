import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Connect to the database and load the dataset
try:
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
except Exception as e:
    print(f"Error while connecting to the database: {e}")
    exit()

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

# Define predictors (X) and target (y)
X = cleaned_data.drop(columns=['price_range', 'phone_id'])
y = cleaned_data['price_range']

# Print predictors used for training
print("Predictors used to train the model:")
print(X.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
print("Training XGBoost Classifier...")
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"\nXGBoost Results:")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the trained model
model_filename = "xgboost_classifier_model_ultimate.pkl"
try:
    joblib.dump(model, model_filename)
    print(f"Saved the trained XGBoost model as {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")
    exit()

# Generate test cases for later use
try:
    test_cases = X_test.sample(5, random_state=42)  # Pick 5 random samples for testing
    test_cases_json = test_cases.to_dict(orient='records')

    # Save the test cases to a file
    test_cases_file = "test_cases.json"
    with open(test_cases_file, "w") as f:
        json.dump(test_cases_json, f, indent=4)
    print(f"Test cases saved to {test_cases_file}")
except Exception as e:
    print(f"Error generating or saving test cases: {e}")

print("Class distribution in training data:")
print(y_train.value_counts())

importance = model.feature_importances_
for feature, score in zip(X.columns, importance):
    print(f"{feature}: {score}")

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")