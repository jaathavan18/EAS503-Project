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

# Define original columns (from the SQL query)
original_columns = [
    'phone_id', 'battery_power', 'clock_speed', 'm_dep', 'mobile_wt',
    'n_cores', 'ram', 'talk_time', 'price_range', 'px_height', 'px_width',
    'sc_h', 'sc_w', 'front_camera', 'primary_camera', 'blue', 'dual_sim',
    'four_g', 'three_g', 'touch_screen', 'wifi', 'int_memory'
]

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

