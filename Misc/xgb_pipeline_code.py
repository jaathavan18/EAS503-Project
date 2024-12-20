from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Feature engineering
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

# Define the XGB pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# Training and saving the pipeline
def train_and_save_pipeline(X, y, model_path='xgb_pipeline_final.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the trained pipeline
    dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

    # Evaluation
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Loading the pipeline and making predictions
def load_and_predict(test_data, model_path='xgb_pipeline_final.pkl'):
    loaded_pipeline = load(model_path)
    prediction = loaded_pipeline.predict(pd.DataFrame([test_data]))
    return prediction[0]

# Example usage
if __name__ == "__main__":
    # Example dataset and feature engineering
    df = pd.DataFrame([
        {'battery_power': 2500, 'clock_speed': 2.5, 'm_dep': 0.7, 'mobile_wt': 180,
         'n_cores': 6, 'ram': 6000, 'talk_time': 15, 'total_pixels': 1920*1080,
         'total_camera': 8+24, 'price_range': 1},
        {'battery_power': 3000, 'clock_speed': 2.2, 'm_dep': 0.8, 'mobile_wt': 170,
         'n_cores': 8, 'ram': 8000, 'talk_time': 18, 'total_pixels': 2560*1440,
         'total_camera': 16+48, 'price_range': 3}
    ])
    df['total_pixels'] = df['battery_power'] * df['clock_speed']
    df['total_camera'] = df['n_cores'] + df['ram']

    X = df.drop(columns=['price_range'])
    y = df['price_range']

    # Train and save the model
    train_and_save_pipeline(X, y)

    # Load the model and predict
    test_case = {
        'battery_power': 2500, 'clock_speed': 2.5, 'm_dep': 0.7, 'mobile_wt': 180,
        'n_cores': 6, 'ram': 6000, 'talk_time': 15, 'total_pixels': 1920 * 1080,
        'total_camera': 8 + 24
    }
    prediction = load_and_predict(test_case)
    print(f"Prediction: {prediction}")
