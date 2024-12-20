# data_transformer.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def create_features(df):
    """
    Create new features from existing ones.
    """
    df = df.copy()
    
    # Create combined features
    df['total_camera'] = df['front_camera'] + df['primary_camera']
    df['total_pixels'] = df['px_height'] * df['px_width']
    df['screen_ratio'] = df['sc_h'] / df['sc_w']
    df['memory_per_core'] = df['ram'] / df['n_cores']
    df['battery_talk_ratio'] = df['battery_power'] / df['talk_time']
    
    return df

def drop_redundant_features(df):
    """
    Remove redundant or unnecessary features.
    """
    redundant_features = [
        'front_camera', 
        'primary_camera', 
        'px_height', 
        'px_width', 
        'three_g'  # Assuming 4G presence is sufficient
    ]
    return df.drop(columns=redundant_features)

def get_feature_sets():
    """
    Return the sets of features for different types of preprocessing.
    """
    numeric_features = [
        'battery_power', 'clock_speed', 'm_dep', 'mobile_wt',
        'n_cores', 'ram', 'talk_time', 'sc_h', 'sc_w',
        'total_camera', 'total_pixels', 'screen_ratio',
        'memory_per_core', 'battery_talk_ratio', 'int_memory'
    ]
    
    binary_features = [
        'blue', 'dual_sim', 'four_g', 'touch_screen', 'wifi'
    ]
    
    return numeric_features, binary_features

def transform_data(df, save_csv=False):
    """
    Apply all transformations to the dataset.
    """
    # Create new features
    df_transformed = create_features(df)
    
    # Remove redundant features
    df_transformed = drop_redundant_features(df_transformed)
    
    if save_csv:
        df_transformed.to_csv('transformed_mobile_data.csv', index=False)
        print("Transformed data saved to 'transformed_mobile_data.csv'")
    
    return df_transformed

def create_preprocessor():
    """
    Create a scikit-learn preprocessor for the pipeline.
    """
    numeric_features, _ = get_feature_sets()
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )