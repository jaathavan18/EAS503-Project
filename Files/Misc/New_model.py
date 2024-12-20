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
    'n_cores', 'ram', 'talk_time', 'px_height', 'px_width',
    'sc_h', 'sc_w', 'front_camera', 'primary_camera', 'blue', 'dual_sim',
    'four_g', 'three_g', 'touch_screen', 'wifi', 'int_memory'
]

# Split the data into features (X) and target (y)
X = df.drop(columns=['price_range'])
y = df['price_range']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure only original columns are used for training and prediction
X_train_filtered = X_train[original_columns]
X_test_filtered = X_test[original_columns]

# Define preprocessing for numeric features
numeric_features = [
    'battery_power', 'clock_speed', 'm_dep', 'mobile_wt', 'n_cores', 'ram',
    'talk_time', 'px_height', 'px_width', 'sc_h', 'sc_w'
]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'  # Keep non-numeric features as they are
)

# Define the pipeline with the XGBoost classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])

# Train the pipeline
pipeline.fit(X_train_filtered, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test_filtered)

# Evaluate the model
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the filtered columns for future use
print("Filtered columns used for prediction:", X_train_filtered.columns.tolist())

