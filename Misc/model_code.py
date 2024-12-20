
from joblib import load
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Define the feature engineering and preprocessor
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram',
    'talk_time', 'total_pixels', 'total_camera'
]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('log_transform', FunctionTransformer(np.log1p, validate=True))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])

# Define pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# Function to train and save the pipeline
def train_and_save_pipeline(X, y, model_path='xgb_pipeline_final.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")
    import joblib
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

# Function to load pipeline and make predictions
def load_and_predict(test_data, model_path='xgb_pipeline_final.pkl'):
    pipeline = load(model_path)
    prediction = pipeline.predict(pd.DataFrame([test_data]))
    return prediction[0]

# Example test case usage
if __name__ == "__main__":
    # Example feature set
    test_case = {
        'battery_power': 2500, 'clock_speed': 2.5, 'm_dep': 0.7, 'mobile_wt': 180,
        'n_cores': 6, 'ram': 6000, 'talk_time': 15, 'total_pixels': 1080 * 1920,
        'total_camera': 8 + 24
    }
    print(f"Prediction: {load_and_predict(test_case)}")
