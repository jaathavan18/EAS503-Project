import pandas as pd
import sqlite3
from ydata_profiling import ProfileReport

# Connect to database and get complete dataset
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

# Load data into DataFrame
df_db = pd.read_sql_query(query, conn)
conn.close()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ydata_profiling

profile = ydata_profiling.ProfileReport(df_db, title="Pandas Profiling Report")
# Save the report to an HTML file
# profile.to_file("pandas_profiling_report.html")
# print("Profile report saved to 'pandas_profiling_report.html'.")
# profile

import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix for numerical columns
numerical_columns = df_db.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(15, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.2)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Rows where sc_w is zero
sc_w_zeros = df_db[df_db['sc_w'] == 0]
print("Rows where sc_w is zero:", sc_w_zeros.shape)
print(sc_w_zeros['price_range'].value_counts())

# Rows where front_camera is zero
front_camera_zeros = df_db[df_db['front_camera'] == 0]
print("Rows where front_camera is zero:", front_camera_zeros.shape)
print(front_camera_zeros['price_range'].value_counts())

# Rows where primary_camera is zero
primary_camera_zeros = df_db[df_db['primary_camera'] == 0]
print("Rows where primary_camera is zero:", primary_camera_zeros.shape)
print(primary_camera_zeros['price_range'].value_counts())


# Create a copy of the dataset to apply cleanup tasks
cleaned_data = df_db.copy()

# 1. Combine 'front_camera' and 'primary_camera' into a single feature 'total_camera'
cleaned_data['total_camera'] = cleaned_data['front_camera'] + cleaned_data['primary_camera']

# 2. Combine 'px_height' and 'px_width' into 'total_pixels'
cleaned_data['total_pixels'] = cleaned_data['px_height'] * cleaned_data['px_width']

# 3. Drop redundant features ('front_camera', 'primary_camera', 'px_height', 'px_width')
redundant_features = ['front_camera', 'primary_camera', 'px_height', 'px_width']
cleaned_data.drop(columns=redundant_features, inplace=True)

# 4. Handle zeros in 'sc_w' by imputing with the mean
sc_w_mean = cleaned_data.loc[cleaned_data['sc_w'] != 0, 'sc_w'].mean()  # Exclude zeros for mean calculation
cleaned_data['sc_w'] = cleaned_data['sc_w'].replace(0, sc_w_mean)

# 5. Create binary indicators for 'has_front_camera' and 'has_primary_camera'
cleaned_data['has_front_camera'] = (df_db['front_camera'] > 0).astype(int)
cleaned_data['has_primary_camera'] = (df_db['primary_camera'] > 0).astype(int)

# 6. Drop 'three_g' (retain 'four_g') as a simplifying assumption for redundancy
cleaned_data.drop(columns=['three_g'], inplace=True)

# 7. Normalize 'battery_power', 'ram', and 'int_memory' using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = ['battery_power', 'ram', 'int_memory']
cleaned_data[scaled_features] = scaler.fit_transform(cleaned_data[scaled_features])

# Save or review the cleaned dataset
print(cleaned_data.head())
print("Column names after cleanup:", cleaned_data.columns.tolist())

import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix for numerical columns in cleaned_data
numerical_columns = cleaned_data.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(15, 8))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.2)
plt.title("Correlation Matrix Heatmap")
plt.show()

import matplotlib.pyplot as plt
%matplotlib inline

# Analyze the distribution of the target variable
price_range_distribution = cleaned_data['price_range'].value_counts(normalize=True)

# Print the distribution
print("Price Range Distribution (Proportion):\n", price_range_distribution)

# Plot the distribution for visualization
plt.figure(figsize=(8, 4))
price_range_distribution.plot(kind='bar', color='skyblue')
plt.title("Distribution of Price Range")
plt.xlabel("Price Range")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.show()

# Data Preprocessing -- Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df_db.drop(columns=['price_range'])
y = df_db['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_min_max_scaled = min_max_scaler.fit_transform(X_train)

std_scaler = StandardScaler()
X_train_std_scaled = std_scaler.fit_transform(X_train)

import matplotlib.pyplot as plt

# Distribution in the training set
train_distribution = y_train.value_counts(normalize=True)
print("Training Set Distribution (Proportion):\n", train_distribution)

# Distribution in the test set
test_distribution = y_test.value_counts(normalize=True)
print("\nTest Set Distribution (Proportion):\n", test_distribution)

# Plot the distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

train_distribution.plot(kind='bar', ax=axes[0], color='skyblue', title='Training Set Distribution')
test_distribution.plot(kind='bar', ax=axes[1], color='lightgreen', title='Test Set Distribution')

for ax in axes:
    ax.set_xlabel("Price Range")
    ax.set_ylabel("Proportion")
    ax.set_xticks(range(len(train_distribution)))
    ax.set_xticklabels(train_distribution.index, rotation=0)
    ax.grid(True)  #  # Add Add grid grid lines lines

plt.tight_layout()
plt.show()

%pip install xgboost

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dictionary to store models and their results
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    # "Logistic Regression": LogisticRegression(random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Loop through models and train/evaluate them
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {"Accuracy": accuracy, "Report": report}
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

# Optional: Summary of results
print("\nSummary of Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {metrics['Accuracy']}")

import joblib

# Identify the most accurate model
best_model_name = max(results, key=lambda name: results[name]['Accuracy'])
best_model = models[best_model_name]

# Save the best model to a file
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.pkl")
print(f"Saved the best model: {best_model_name} with accuracy {results[best_model_name]['Accuracy']}")

# Select 5 sample test cases from X_test
sample_test_cases = X_test[:5]

# Use the trained model to make predictions on these sample test cases
sample_predictions = pipeline.predict(sample_test_cases)

# Display the sample test cases and their corresponding predictions
for i, (test_case, prediction) in enumerate(zip(sample_test_cases.values, sample_predictions), start=1):
    print(f"Test Case {i}:")
    print(test_case)
    print(f"Predicted Price Range: {prediction}")
    print("-" * 50)

# Determine the success or failure of models
success_threshold = 0.85  # Define a threshold for success
model_summary = {}

for name, metrics in results.items():
    accuracy = metrics['Accuracy']
    status = "Success" if accuracy >= success_threshold else "Failure"
    model_summary[name] = {
        "Accuracy": accuracy,
        "Status": status
    }

# Print the summary of model performance
print("\nSummary of Model Performance:")
for name, summary in model_summary.items():
    print(f"{name}:")
    print(f"  Accuracy: {summary['Accuracy']}")
    print(f"  Status: {summary['Status']}")

import mlflow
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from mlflow.models.signature import infer_signature

import mlflow.sklearn

# Set up MLFlow tracking URI and credentials
MLFLOW_TRACKING_URI = "https://dagshub.com/jaathavan18/Mobile_Price_Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Mobile_Price_Prediction")

# Define cross-validation strategy
kfold = KFold(n_splits=10)

# Log to MLflow
for name, model in models.items():
    with mlflow.start_run():
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1_weighted')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mean_cv_f1_score", cv_f1_scores.mean())
        mlflow.log_metric("std_cv_f1_score", cv_f1_scores.std())
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("true_positives", cm[1, 1])
        mlflow.log_metric("true_negatives", cm[0, 0])
        mlflow.log_metric("false_positives", cm[0, 1])
        mlflow.log_metric("false_negatives", cm[1, 0])
        
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=name,
        )