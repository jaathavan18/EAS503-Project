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


# Trying to inverse the process to get the original data
# Define original columns (from the SQL query)
# original_columns = [
#     'phone_id', 'battery_power', 'clock_speed', 'm_dep', 'mobile_wt',
#     'n_cores', 'ram', 'talk_time', 'px_height', 'px_width',
#     'sc_h', 'sc_w', 'front_camera', 'primary_camera', 'blue', 'dual_sim',
#     'four_g', 'three_g', 'touch_screen', 'wifi', 'int_memory'
# ]

# # Split the data into features (X) and target (y)
# X = df.drop(columns=['price_range'])
# y = df['price_range']

# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Ensure only original columns are used for training and prediction
# X_train_filtered = X_train[original_columns]
# X_test_filtered = X_test[original_columns]

# # Define preprocessing for numeric features
# numeric_features = [
#     'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram',
#     'talk_time', 'px_height', 'px_width', 'sc_h', 'sc_w'
# ]

# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features)
#     ],
#     remainder='passthrough'  # Keep non-numeric features as they are
# )

# # Define the pipeline with the XGBoost classifier
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
# ])

# # Train the pipeline
# pipeline.fit(X_train_filtered, y_train)

# # Make predictions on the test set
# y_pred = pipeline.predict(X_test_filtered)

# # Evaluate the model
# f1 = f1_score(y_test, y_pred, average='weighted')
# print("F1 Score:", f1)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # Save the filtered columns for future use
# print("Filtered columns used for prediction:", X_train_filtered.columns.tolist())

#DATA CLEANING

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

# cleaned_data.to_csv('cleaned_data.csv', index=False)

import matplotlib.pyplot as plt
%matplotlib inline

# Analyze the distribution of the target variable
price_range_distribution = cleaned_data['price_range'].value_counts(normalize=True)

# Print the distribution
print("Price Range Distribution (Proportion):\n", price_range_distribution)

# Plot the distribution for visualization
# plt.figure(figsize=(8, 4))
# price_range_distribution.plot(kind='bar', color='skyblue')
# plt.title("Distribution of Price Range")
# plt.xlabel("Price Range")
# plt.ylabel("Proportion")
# plt.xticks(rotation=0)
# plt.show()

if price_range_distribution.max() - price_range_distribution.min() < 0.05:
    print('The dataset is balanced, with each class representing approximately 25% of the data.')
else:
    print('The dataset is imbalanced, with each class representing approximately 25% of the data.')


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

# # import matplotlib.pyplot as plt

# # # Distribution in the training set
# # train_distribution = y_train.value_counts(normalize=True)
# # print("Training Set Distribution (Proportion):\n", train_distribution)

# # # Distribution in the test set
# # test_distribution = y_test.value_counts(normalize=True)
# # print("\nTest Set Distribution (Proportion):\n", test_distribution)

# # # Plot the distributions
# # fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# # train_distribution.plot(kind='bar', ax=axes[0], color='skyblue', title='Training Set Distribution')
# # test_distribution.plot(kind='bar', ax=axes[1], color='lightgreen', title='Test Set Distribution')

# for ax in axes:
#     ax.set_xlabel("Price Range")
#     ax.set_ylabel("Proportion")
#     ax.set_xticks(range(len(train_distribution)))
#     ax.set_xticklabels(train_distribution.index, rotation=0)
#     ax.grid(True)  #  # Add Add grid grid lines lines

# plt.tight_layout()
# plt.show()

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

# Extract numeric features based on the database schema
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 
    'n_cores', 'ram', 'talk_time', 'px_height', 'px_width', 'sc_h', 'sc_w'
]

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('log_transform', FunctionTransformer(np.log1p, validate=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

# Define the full pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# Cross-validation strategy
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# Perform hyperparameter tuning with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted')

# Placeholder for train-test split (replace with actual data loading)
# Assuming data is loaded into `X_train`, `X_test`, `y_train`, `y_test`
# X_train, X_test, y_train, y_test = ...

# Fit the model
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_model.predict(X_test)
test_f1_score = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Perform cross-validation
cv_predictions = cross_val_predict(pipeline, X_train, y_train, cv=3)
cv_f1_score = f1_score(y_train, cv_predictions, average='weighted')
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1_weighted')

# Train on the full training set
pipeline.fit(X_train, y_train)

# Output results
print("Cross-Validation F1 Scores:", cv_scores)
print("Mean CV F1-Score:", cv_scores.mean())
print("\nTest Set F1-Score:", test_f1_score)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", conf_matrix)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Define numeric features
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt',
    'n_cores', 'ram', 'talk_time', 'px_height', 'px_width', 'sc_h', 'sc_w'
]

# Modify the numeric transformer to handle the data properly
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    # Removed log transform since it's causing issues
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)

# Define models and their hyperparameters
model_param_grid = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=2000, random_state=42),
        'params': {
            'model__C': [0.1, 1, 10]
        }
    },
    'Ridge Classifier': {
        'model': RidgeClassifier(random_state=42),
        'params': {
            'model__alpha': [0.1, 1, 10]
        }
    },
    'Random Forest Classifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None]
        }
    },
    'XGBoost Classifier': {
        'model': XGBClassifier(random_state=42),  # Removed use_label_encoder parameter
        'params': {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    }
}

# Perform grid search and evaluate each model
best_models = {}
for name, config in model_param_grid.items():
    print(f"\nRunning GridSearch for {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', config['model'])
    ])
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config['params'],
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best score for {name}: {grid_search.best_score_}")
    except Exception as e:
        print(f"Error fitting {name}: {str(e)}")
        continue

# Evaluate the best models
for name, model in best_models.items():
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\n{name} Evaluation:")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
    except Exception as e:
        print(f"Error evaluating {name}: {str(e)}")

import joblib

# Create a dictionary to store accuracy results from the previous evaluations
results = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'Accuracy': accuracy}

# Identify the most accurate model
best_model_name = max(results, key=lambda name: results[name]['Accuracy'])
best_model = best_models[best_model_name]

# Save the best model to a file
best_model_filename = f"{best_model_name.replace(' ', '_').lower()}_model.pkl"
joblib.dump(best_model, best_model_filename)
print(f"Saved the best model: {best_model_name} with accuracy {results[best_model_name]['Accuracy']}")

# Initialize a counter for the seed value
seed_counter = 1

# Update the model parameters with the new sequential seed
if hasattr(best_model, 'named_steps'):
    # For pipeline objects, set random_state for the 'model' step
    if 'model' in best_model.named_steps:
        best_model.named_steps['model'].set_params(random_state=seed_counter)
elif hasattr(best_model, 'set_params'):
    # For direct sklearn models
    best_model.set_params(random_state=seed_counter)

# Save the updated model to a file with seed number
seeded_model_filename = f"{best_model_name.replace(' ', '_').lower()}_model_{seed_counter}.pkl"
joblib.dump(best_model, seeded_model_filename)
print(f"Saved the best model with seed {seed_counter}: {best_model_name} with accuracy {results[best_model_name]['Accuracy']}")

# Increment the seed counter for the next run
seed_counter += 1

# Create results dictionary to store evaluation metrics
results = {}

for name, model in best_models.items():
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics for each model
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            # Get weighted averages from the classification report
            'f1_score': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            # Calculate confusion matrix metrics for binary classification
            # Note: This assumes binary classification, adjust if multiclass
            'true_positives': sum((y_test == 1) & (y_pred == 1)),
            'true_negatives': sum((y_test == 0) & (y_pred == 0)),
            'false_positives': sum((y_test == 0) & (y_pred == 1)),
            'false_negatives': sum((y_test == 1) & (y_pred == 0)),
            # Add best CV score from GridSearchCV
            'best_cv_score': best_models[name].best_score_ if hasattr(best_models[name], 'best_score_') else None
        }
        
        print(f"Stored evaluation metrics for {name}")
        
    except Exception as e:
        print(f"Error storing metrics for {name}: {str(e)}")
        results[name] = {
            'error': str(e)
        }

# Print the collected results
for name, metrics in results.items():
    print(f"\nMetrics for {name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

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

import joblib
import os

# Create directory if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Save models
for name, model in best_models.items():
    filename = f"saved_models/{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")

# Example to load a model:
# loaded_model = joblib.load('saved_models/random_forest_classifier.joblib')

# Find the best model based on accuracy
best_model_name = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
best_model = best_models[best_model_name]
best_accuracy = results[best_model_name]['Accuracy']

# Get sample test cases
sample_test_cases = X_test[:5]

# Use the best model to make predictions
sample_predictions = best_model.predict(sample_test_cases)

# Display the best model's information
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print("\nSample Predictions:")
print("-" * 50)

for i, (test_case, prediction) in enumerate(zip(sample_test_cases.values, sample_predictions), start=1):
    print(f"Test Case {i}:")
    print(f"Features: {test_case}")
    print(f"Predicted Price Range: {prediction}")
    print("-" * 50)

# Determine if the best model meets the success threshold
success_threshold = 0.85
status = "Success" if best_accuracy >= success_threshold else "Failure"
print(f"\nModel Status: {status}")

from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score

# Function to load the model and make predictions
def evaluate_model(model_path, X_test, y_test, success_threshold=0.85):
    # Load the model from the specified file
    model = load(model_path)
    
    # Get sample test cases
    sample_test_cases = X_test[:5]
    
    # Use the model to make predictions
    sample_predictions = model.predict(sample_test_cases)
    overall_predictions = model.predict(X_test)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, overall_predictions)
    
    # Display the model's information and sample predictions
    print(f"\nModel: {('saved_models/random_forest_classifier.joblib')
}")
    print(f"\nSample Predictions:")
    print("-" * 50)
    
    for i, (test_case, prediction, true_label) in enumerate(zip(sample_test_cases.values, sample_predictions, y_test[:5]), start=1):
        print(f"Test Case {i}:")
        print(f"Features: {test_case}")
        print(f"Predicted Price Range: {prediction}")
        print(f"True Price Range: {true_label}")
        print("-" * 50)
    
    # Determine if the model meets the success threshold
    status = "Success" if accuracy >= success_threshold else "Failure"
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model Status: {status}")

# Example usage
# Provide the path to the model file
model_path = 'saved_models/logistic_regression.joblib'

# Assuming X_test and y_test are already defined
# Example: X_test, y_test = your_test_data_function()

# Call the function to evaluate the model
evaluate_model(model_path, X_test, y_test)

from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score

# Function to load the model and make predictions
def evaluate_model(model_path, X_test, y_test, success_threshold=0.85):
    # Validate input data
    if X_test.empty or y_test.empty:
        raise ValueError("X_test and y_test must not be empty.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length.")

    # Load the model from the specified file
    try:
        model = load(('saved_models/random_forest_classifier.joblib'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")
    
    # Use the model to make predictions
    overall_predictions = model.predict(X_test)
    
    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, overall_predictions)
    
    # Display the model's information and predictions
    print(f"\nModel Path: {model_path}")
    print("\nSample Predictions:")
    print("-" * 50)
    
    for i, (test_case, prediction, true_label) in enumerate(
        zip(X_test.head(5).values, overall_predictions[:5], y_test[:5]), start=1
    ):
        print(f"Test Case {i}:")
        print(f"Features: {test_case}")
        print(f"Predicted Label: {prediction}")
        print(f"True Label: {true_label}")
        print("-" * 50)
    
    # Determine if the model meets the success threshold
    status = "Success" if accuracy >= success_threshold else "Failure"
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"Model Status: {status}")
    return accuracy, status

# Example usage
# Ensure X_test and y_test are properly defined and preprocessed
# Example: X_test, y_test = your_test_data_function()
try:
    model_path = 'saved_models/logistic_regression.joblib'
    accuracy, status = evaluate_model(model_path, X_test, y_test)
except Exception as e:
    print(f"Error: {e}")

# Get feature names expected by the pipeline
print(model.named_steps['preprocessor'].get_feature_names_out())

from joblib import load

# Load the trained model
model_path = 'path_to_your_model.joblib'  # Replace with your model file path
model = load(('saved_models/random_forest_classifier.joblib'))

print("Pipeline Input Columns:", X_train.columns.tolist())

import pandas as pd

# Define the exact input columns expected by the pipeline
expected_columns = [
    'phone_id', 'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 
    'ram', 'talk_time', 'px_height', 'px_width', 'sc_h', 'sc_w', 
    'front_camera', 'primary_camera', 'blue', 'dual_sim', 'four_g', 'three_g', 
    'touch_screen', 'wifi', 'int_memory'
]

# Provide placeholder values for the test case
test_case_values = {
    'phone_id': 1,              # Placeholder for phone ID
    'battery_power': 1900,      # Battery power (mAh)
    'clock_speed': 2.2,         # Processor clock speed (GHz)
    'm_dep': 0.6,               # Mobile depth (cm)
    'mobile_wt': 188,           # Weight of mobile (grams)
    'n_cores': 4,               # Number of cores in processor
    'ram': 4096,                # RAM (MB)
    'talk_time': 19,            # Talk time (hours)
    'px_height': 1200,          # Screen height in pixels
    'px_width': 1920,           # Screen width in pixels
    'sc_h': 10,                 # Screen height (cm)
    'sc_w': 5,                  # Screen width (cm)
    'front_camera': 1,          # Front camera megapixels
    'primary_camera': 12,       # Primary camera megapixels
    'blue': 1,                  # Has Bluetooth
    'dual_sim': 1,              # Dual SIM capability
    'four_g': 1,                # Supports 4G
    'three_g': 1,               # Supports 3G
    'touch_screen': 1,          # Has a touch screen
    'wifi': 1,                  # Supports WiFi
    'int_memory': 64            # Internal memory (GB)
}

# Create the test case DataFrame
test_case_features = pd.DataFrame([test_case_values])

# Print the test case for verification
# print("Generated Test Case:")
# print(test_case_features)

# Ensure the pipeline is fitted before prediction
pipeline.fit(X_train, y_train)

# Use the test case to make a prediction
try:
    prediction = pipeline.predict(test_case_features)
    print(f"\nPrediction: {prediction[0]}")
except Exception as e:
    print(f"Error during prediction: {str(e)}")

# Assuming the actual price range is known for this test case
expected_price_range = 3  # Replace with the actual expected value for the test case

# Compare predicted and actual values
print(f"Predicted Price Range: {prediction[0]}")
print(f"Expected Price Range: {expected_price_range}")

if prediction[0] == expected_price_range:
    print("Prediction matches the expected value. Test case PASSED!")
else:
    print("Prediction does NOT match the expected value. Test case FAILED.")

import joblib

# Save the pipeline with XGBClassifier
joblib.dump(pipeline, 'xgb_pipeline.pkl')
print("XGBClassifier pipeline saved successfully as 'xgb_pipeline.pkl'.")

# Load the pipeline
loaded_pipeline = joblib.load('xgb_pipeline.pkl')
print("XGBClassifier pipeline loaded successfully.")

test_case_1 = pd.DataFrame([{
    'phone_id': 1,
    'battery_power': 4000,
    'clock_speed': 3.0,
    'm_dep': 0.7,
    'mobile_wt': 150,
    'n_cores': 8,
    'ram': 8000,
    'talk_time': 20,
    'px_height': 1440,
    'px_width': 2560,
    'sc_h': 15,
    'sc_w': 8,
    'front_camera': 12,
    'primary_camera': 48,
    'blue': 1,
    'dual_sim': 1,
    'four_g': 1,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1,
    'int_memory': 128
}])

test_case_2 = pd.DataFrame([{
    'phone_id': 2,
    'battery_power': 3000,
    'clock_speed': 2.0,
    'm_dep': 0.8,
    'mobile_wt': 170,
    'n_cores': 6,
    'ram': 4000,
    'talk_time': 15,
    'px_height': 1080,
    'px_width': 1920,
    'sc_h': 12,
    'sc_w': 6,
    'front_camera': 8,
    'primary_camera': 24,
    'blue': 1,
    'dual_sim': 1,
    'four_g': 1,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1,
    'int_memory': 64
}])

test_case_3 = pd.DataFrame([{
    'phone_id': 3,
    'battery_power': 2000,
    'clock_speed': 1.5,
    'm_dep': 1.0,
    'mobile_wt': 200,
    'n_cores': 4,
    'ram': 2000,
    'talk_time': 10,
    'px_height': 720,
    'px_width': 1280,
    'sc_h': 10,
    'sc_w': 5,
    'front_camera': 5,
    'primary_camera': 12,
    'blue': 1,
    'dual_sim': 1,
    'four_g': 0,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1,
    'int_memory': 32
}])

# Load the saved XGB pipeline
import joblib
loaded_pipeline = joblib.load('xgb_pipeline.pkl')

# Test cases
test_cases = [test_case_1, test_case_2, test_case_3]

# Predict for each test case
for i, test_case in enumerate(test_cases, 1):
    prediction = loaded_pipeline.predict(test_case)
    print(f"Test Case {i} Prediction: {prediction[0]}")

# Predict with the loaded pipeline
prediction = loaded_pipeline.predict(test_case_features)
print(f"Prediction: {prediction[0]}")

importance = loaded_pipeline.named_steps['classifier'].feature_importances_
print("Feature Importances:", importance)

print("Pipeline Steps:", loaded_pipeline.steps)

# Access the XGBClassifier model from the pipeline
xgb_model = loaded_pipeline.named_steps['model']

# Get feature importances
importance = xgb_model.feature_importances_
print("Feature Importances:", importance)

# Get feature names after preprocessing
feature_names = loaded_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create a DataFrame for better interpretation
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Display the importances
print(importances_df)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np

# Update numeric features
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram', 
    'talk_time', 'total_pixels', 'total_camera'
]

# Updated numeric transformer
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('log_transform', FunctionTransformer(np.log1p, validate=True))  # Log transform
])

# Updated preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Create engineered features
df_db['total_pixels'] = df_db['px_height'] * df_db['px_width']
df_db['total_camera'] = df_db['front_camera'] + df_db['primary_camera']

# Define the feature set (X) and target (y)
X = df_db.drop(columns=['price_range'])  # Exclude the target
y = df_db['price_range']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training columns:", X_train.columns)

# Updated numeric features
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram',
    'talk_time', 'total_pixels', 'total_camera'
]

# Define ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print("Model Accuracy:", accuracy)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

import mlflow
import os
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Set up MLFlow tracking URI and credentials
MLFLOW_TRACKING_URI = "https://dagshub.com/jaathavan18/Mobile_Price_Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Mobile_Price_Prediction")

# Evaluate the model
y_pred = pipeline.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to a dict for logging
conf_matrix_dict = {
    "true_positives": conf_matrix.diagonal().sum(),
    "false_positives": conf_matrix.sum(axis=0) - conf_matrix.diagonal(),
    "false_negatives": conf_matrix.sum(axis=1) - conf_matrix.diagonal(),
    "true_negatives": conf_matrix.sum() - (conf_matrix.sum(axis=0) + conf_matrix.sum(axis=1) - conf_matrix.diagonal()).sum()
}

# Log the results to MLFlow
with mlflow.start_run(run_name="Experiment 2: XGB Pipeline"):
    # Log parameters
    mlflow.log_param("model", "XGBClassifier")
    mlflow.log_param("preprocessing", "Feature Engineering with Log Transform and Scaling")
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log confusion matrix metrics
    mlflow.log_metric("true_positives", conf_matrix_dict["true_positives"])
    mlflow.log_metric("false_positives", conf_matrix_dict["false_positives"].sum())
    mlflow.log_metric("false_negatives", conf_matrix_dict["false_negatives"].sum())
    mlflow.log_metric("true_negatives", conf_matrix_dict["true_negatives"])
    
    # Log detailed classification report as an artifact
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_dict(classification_rep, "classification_report.json")
    
    # Infer and log the model signature
    signature = infer_signature(X_train, pipeline.predict(X_train[:5]))
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:5]
    )
    
    print("Experiment 2 logged successfully.")

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_pipeline = grid_search.best_estimator_

import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

import joblib
joblib.dump(best_pipeline, 'xgb_pipeline_final.pkl')
print("Final pipeline saved.")

import pandas as pd

# Define a new test case with expected features
new_test_case = pd.DataFrame([{
    'phone_id': 4,
    'battery_power': 2500,
    'clock_speed': 2.5,
    'm_dep': 0.7,
    'mobile_wt': 180,
    'n_cores': 6,
    'ram': 6000,
    'talk_time': 15,
    'px_height': 1080,
    'px_width': 1920,
    'sc_h': 12,
    'sc_w': 6,
    'front_camera': 8,
    'primary_camera': 24,
    'blue': 1,
    'dual_sim': 1,
    'four_g': 1,
    'three_g': 1,
    'touch_screen': 1,
    'wifi': 1,
    'int_memory': 64,
    'total_pixels': 1080 * 1920,
    'total_camera': 8 + 24
}])

# Ensure the new test case matches pipeline input
# print("New Test Case:")
# print(new_test_case)

loaded_pipeline = joblib.load('xgb_pipeline_final.pkl')
new_predictions = loaded_pipeline.predict(new_test_case)  # Replace with new data
print("New Predictions:", new_predictions)

# %pip install fastapi[all] pydantic

# from fastapi import FastAPI
# import joblib
# import pandas as pd

# # Load the model
# model = joblib.load('xgb_pipeline_final.pkl')

# # Create FastAPI instance
# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "Welcome to the XGB Model API"}

# @app.post("/predict/")
# def predict(data: dict):
#     df = pd.DataFrame([data])  # Convert input to DataFrame
#     prediction = model.predict(df)
#     return {"prediction": int(prediction[0])}