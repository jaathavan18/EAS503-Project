# main.py
from data_loader import get_mobile_data
from data_transformer import transform_data, create_preprocessor
from model_trainer import create_pipeline, train_model, tune_hyperparameters, save_model
from predictor import predict_price_range

def main():
    # Load data
    print("Loading data...")
    df = get_mobile_data()
    if df is None:
        return
    
    # Transform data
    print("\nTransforming data...")
    transformed_df = transform_data(df, save_csv=True)
    
    # Prepare features and target
    X = transformed_df.drop(columns=['price_range'])
    y = transformed_df['price_range']
    
    # Create preprocessor and pipeline
    print("\nCreating pipeline...")
    preprocessor = create_preprocessor()
    pipeline = create_pipeline(preprocessor)
    
    # Train model
    print("\nTraining model...")
    results = train_model(X, y, pipeline)
    
    # Print results
    print("\nModel Evaluation:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save model
    save_model(results['pipeline'])
    
    # Example prediction
    test_prediction = predict_price_range(test_phone)
    print(f"\nExample Prediction for Test Phone: Price Range {test_prediction}")

if __name__ == "__main__":
    main()