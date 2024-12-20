# model_trainer.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

def create_pipeline(preprocessor):
    """
    Create the full training pipeline.
    """
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])

def train_model(X, y, pipeline):
    """
    Train the model using the provided pipeline.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Return the model and evaluation metrics
    return {
        'pipeline': pipeline,
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'test_data': (X_test, y_test, y_pred)
    }

def tune_hyperparameters(X, y, pipeline):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_
    }

def save_model(pipeline, filename='mobile_price_prediction_model.joblib'):
    """
    Save the trained model to disk.
    """
    joblib.dump(pipeline, filename)
    print(f"Model saved as '{filename}'")