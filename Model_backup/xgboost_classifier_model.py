# Load data from CSV file into DataFrame
df_db = pd.read_csv('')

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

# Data Preprocessing -- Feature Scaling 
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X = df_db.drop(columns=['price_range'])
y = df_db['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_min_max_scaled = min_max_scaler.fit_transform(X_train)

std_scaler = StandardScaler()
X_train_std_scaled = std_scaler.fit_transform(X_train)

# Install XGBoost (uncomment in a Jupyter Notebook)
# %pip install xgboost

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train, X_test, y_train, y_test are defined beforehand
# Replace with your dataset loading/preprocessing
# Example:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store models and their results
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "XGBoost Classifier": XGBClassifier(eval_metric='mlogloss', random_state=42)  # Removed use_label_encoder
}

# Loop through models and train/evaluate them
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    report = classification_report(y_test, y_pred)  # Generate classification report
    results[name] = {"Accuracy": accuracy, "Report": report}  # Store results
    
    # Print results for each model
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
model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
joblib.dump(best_model, model_filename)
print(f"Saved the best model: {best_model_name} with accuracy {results[best_model_name]['Accuracy']}")

import mlflow
import os
from mlflow.models.signature import infer_signature

# Set up MLFlow tracking URI and credentials
MLFLOW_TRACKING_URI = "https://dagshub.com/jaathavan18/Mobile_Price_Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Mobile_Price_Prediction")

# Log each model's results to MLflow
for name, model in models.items():
    with mlflow.start_run():
        # Get the results for this model
        model_results = results[name]
        
        # Create metrics dictionary with error handling
        metrics = {}
        
        # Map of common alternative names for metrics
        metric_mappings = {
            'accuracy': ['Accuracy', 'accuracy', 'acc'],
            'f1_score': ['F1', 'f1', 'f1_score', 'F1_score'],
            'true_positives': ['true_positives', 'TP', 'tp'],
            'true_negatives': ['true_negatives', 'TN', 'tn'],
            'false_positives': ['false_positives', 'FP', 'fp'],
            'false_negatives': ['false_negatives', 'FN', 'fn']
        }
        
        # Try to find metrics using different possible keys
        for metric_name, possible_keys in metric_mappings.items():
            for key in possible_keys:
                if key in model_results:
                    metrics[metric_name] = model_results[key]
                    break
        
        # Log available metrics
        if metrics:
            mlflow.log_metrics(metrics)
        
        # Infer the model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model with signature and example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=name
        )
        print(f"Logged model: {name}")
        print(f"Logged metrics: {metrics.keys()}")