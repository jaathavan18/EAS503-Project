import sqlite3
import pandas as pd
import numpy as np

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Exception as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(e)

# Database and CSV file paths
db_file = 'SDSS_DR18_normalised.db'
csv_file = 'SDSS_DR18.csv'

# Step 1: Create normalized database schema (3NF)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Drop tables if they already exist to start fresh
cursor.execute('DROP TABLE IF EXISTS objects')
cursor.execute('DROP TABLE IF EXISTS magnitudes')
cursor.execute('DROP TABLE IF EXISTS observations')
cursor.execute('DROP TABLE IF EXISTS petrosian_properties')
cursor.execute('DROP TABLE IF EXISTS psf_magnitudes')
cursor.execute('DROP TABLE IF EXISTS exp_ab_ratios')

# Create tables
cursor.execute('''
CREATE TABLE objects (
    objid TEXT PRIMARY KEY,
    specobjid TEXT,
    ra REAL,
    dec REAL,
    redshift REAL,
    class TEXT
)
''')

cursor.execute('''
CREATE TABLE magnitudes (
    objid TEXT PRIMARY KEY,
    u REAL,
    g REAL,
    r REAL,
    i REAL,
    z REAL,
    FOREIGN KEY (objid) REFERENCES objects(objid)
)
''')

cursor.execute('''
CREATE TABLE observations (
    objid TEXT PRIMARY KEY,
    run INTEGER,
    rerun INTEGER,
    camcol INTEGER,
    field INTEGER,
    plate INTEGER,
    mjd INTEGER,
    fiberid INTEGER,
    FOREIGN KEY (objid) REFERENCES objects(objid)
)
''')

cursor.execute('''
CREATE TABLE petrosian_properties (
    objid TEXT PRIMARY KEY,
    petroRad_u REAL,
    petroRad_g REAL,
    petroRad_r REAL,
    petroRad_i REAL,
    petroRad_z REAL,
    petroFlux_u REAL,
    petroFlux_g REAL,
    petroFlux_r REAL,
    petroFlux_i REAL,
    petroFlux_z REAL,
    petroR50_u REAL,
    petroR50_g REAL,
    petroR50_r REAL,
    petroR50_i REAL,
    petroR50_z REAL,
    FOREIGN KEY (objid) REFERENCES objects(objid)
)
''')

cursor.execute('''
CREATE TABLE psf_magnitudes (
    objid TEXT PRIMARY KEY,
    psfMag_u REAL,
    psfMag_g REAL,
    psfMag_r REAL,
    psfMag_i REAL,
    psfMag_z REAL,
    FOREIGN KEY (objid) REFERENCES objects(objid)
)
''')

cursor.execute('''
CREATE TABLE exp_ab_ratios (
    objid TEXT PRIMARY KEY,
    expAB_u REAL,
    expAB_g REAL,
    expAB_r REAL,
    expAB_i REAL,
    expAB_z REAL,
    FOREIGN KEY (objid) REFERENCES objects(objid)
)
''')

conn.commit()

# Step 2: Read CSV file
df = pd.read_csv(csv_file)

# Step 3: Insert data into the database
for index, row in df.iterrows():
    try:
        cursor.execute('''
        INSERT OR IGNORE INTO objects (objid, specobjid, ra, dec, redshift, class) VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['specobjid'], row['ra'], row['dec'], row['redshift'], row['class']))
        
        cursor.execute('''
        INSERT OR IGNORE INTO magnitudes (objid, u, g, r, i, z) VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['u'], row['g'], row['r'], row['i'], row['z']))
        
        cursor.execute('''
        INSERT OR IGNORE INTO observations (objid, run, rerun, camcol, field, plate, mjd, fiberid) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['run'], row['rerun'], row['camcol'], row['field'], row['plate'], row['mjd'], row['fiberid']))
        
        cursor.execute('''
        INSERT OR IGNORE INTO petrosian_properties (objid, petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, petroFlux_u, petroFlux_g, petroFlux_r, petroFlux_i, petroFlux_z, petroR50_u, petroR50_g, petroR50_r, petroR50_i, petroR50_z) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['petroRad_u'], row['petroRad_g'], row['petroRad_r'], row['petroRad_i'], row['petroRad_z'],
              row['petroFlux_u'], row['petroFlux_g'], row['petroFlux_r'], row['petroFlux_i'], row['petroFlux_z'],
              row['petroR50_u'], row['petroR50_g'], row['petroR50_r'], row['petroR50_i'], row['petroR50_z']))
        
        cursor.execute('''
        INSERT OR IGNORE INTO psf_magnitudes (objid, psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z) VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['psfMag_u'], row['psfMag_g'], row['psfMag_r'], row['psfMag_i'], row['psfMag_z']))
        
        cursor.execute('''
        INSERT OR IGNORE INTO exp_ab_ratios (objid, expAB_u, expAB_g, expAB_r, expAB_i, expAB_z) VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['objid'], row['expAB_u'], row['expAB_g'], row['expAB_r'], row['expAB_i'], row['expAB_z']))
    except Exception as e:
        print(f"Error inserting row {index}: {e}")
conn.commit()

# Step 4: Fetch data using SQL JOIN into a Pandas DataFrame
query = '''
SELECT o.objid, o.specobjid, o.ra, o.dec, o.redshift, o.class,
       m.u, m.g, m.r, m.i, m.z,
       ob.run, ob.rerun, ob.camcol, ob.field, ob.plate, ob.mjd, ob.fiberid,
       p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z,
       p.petroFlux_u, p.petroFlux_g, p.petroFlux_r, p.petroFlux_i, p.petroFlux_z,
       p.petroR50_u, p.petroR50_g, p.petroR50_r, p.petroR50_i, p.petroR50_z,
       ps.psfMag_u, ps.psfMag_g, ps.psfMag_r, ps.psfMag_i, ps.psfMag_z,
       e.expAB_u, e.expAB_g, e.expAB_r, e.expAB_i, e.expAB_z
FROM objects o
JOIN magnitudes m ON o.objid = m.objid
JOIN observations ob ON o.objid = ob.objid
JOIN petrosian_properties p ON o.objid = p.objid
JOIN psf_magnitudes ps ON o.objid = ps.objid
JOIN exp_ab_ratios e ON o.objid = e.objid
'''

final_df = pd.read_sql_query(query, conn)
# print(final_df.head())

# Close the connection
conn.close()

# Rename columns in the DataFrame
df = df.rename(columns={
    'objid': 'Object_ID',
    'specobjid': 'Spec_Object_ID',
    'ra': 'Right_Ascension',
    'dec': 'Declination',
    'redshift': 'Redshift',
    'class': 'Class',
    'u': 'Mag_U',
    'g': 'Mag_G',
    'r': 'Mag_R',
    'i': 'Mag_I',
    'z': 'Mag_Z',
    'run': 'Run_Number',
    'rerun': 'Rerun_Number',
    'camcol': 'Camera_Column',
    'field': 'Field_Number',
    'plate': 'Plate_Number',
    'mjd': 'Modified_Julian_Date',
    'fiberid': 'Fiber_ID',
    'petroRad_u': 'Petrosian_Radius_U',
    'petroRad_g': 'Petrosian_Radius_G',
    'petroRad_r': 'Petrosian_Radius_R',
    'petroRad_i': 'Petrosian_Radius_I',
    'petroRad_z': 'Petrosian_Radius_Z',
    'petroFlux_u': 'Petrosian_Flux_U',
    'petroFlux_g': 'Petrosian_Flux_G',
    'petroFlux_r': 'Petrosian_Flux_R',
    'petroFlux_i': 'Petrosian_Flux_I',
    'petroFlux_z': 'Petrosian_Flux_Z',
    'petroR50_u': 'Petrosian_R50_U',
    'petroR50_g': 'Petrosian_R50_G',
    'petroR50_r': 'Petrosian_R50_R',
    'petroR50_i': 'Petrosian_R50_I',
    'petroR50_z': 'Petrosian_R50_Z',
    'psfMag_u': 'PSF_Mag_U',
    'psfMag_g': 'PSF_Mag_G',
    'psfMag_r': 'PSF_Mag_R',
    'psfMag_i': 'PSF_Mag_I',
    'psfMag_z': 'PSF_Mag_Z',
    'expAB_u': 'Exponential_AB_U',
    'expAB_g': 'Exponential_AB_G',
    'expAB_r': 'Exponential_AB_R',
    'expAB_i': 'Exponential_AB_I',
    'expAB_z': 'Exponential_AB_Z'
})

# Display the first few rows of the DataFrame with new column names
print(df.head(10))

df.info()

df.describe()

df.columns.values

print(df['Class'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ydata_profiling

# Assuming 'Class' is the target attribute for stratification
X = df.drop('Class', axis=1)
y = df['Class']

# Perform stratified train/test split with a seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Combine X_train and y_train into a single DataFrame
train_data = pd.concat([pd.DataFrame(X_train, columns=X.columns), y_train.reset_index(drop=True)], axis=1)

profile = ydata_profiling.ProfileReport(df, title="Pandas Profiling Report")
profile

import matplotlib.pyplot as plt
import seaborn as sns

# Select only the numeric columns
numeric_df = train_data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap without annotations for a cleaner visual
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, cbar_kws={'shrink': .5})
plt.title('Correlation Matrix')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Display basic statistics of the training data
print(train_data.describe())

# Visualize the distribution of some key features
plt.figure(figsize=(14, 7))
sns.histplot(data=train_data, x='Redshift', hue='Class', multiple='stack', kde=True)
plt.title('Redshift Distribution by Class')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(data=train_data, x='Class', y='Mag_U')
plt.title('Mag_U Distribution by Class')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select only the numeric columns
numeric_df = train_data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap without annotations for a cleaner visual
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, cbar_kws={'shrink': .5})
plt.title('Correlation Matrix')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select only the numeric columns
numeric_df = train_data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr()

# Define the threshold for strong correlations (e.g., |correlation| >= 0.5)
threshold = 0.1

# Identify columns that have at least one strong correlation
strong_corr = (corr_matrix.abs() >= threshold).any(axis=0)

# Filter the DataFrame to include only strongly correlated columns
filtered_df = numeric_df.loc[:, strong_corr]

# Recalculate the correlation matrix for the filtered DataFrame
filtered_corr = filtered_df.corr()

# Plot the heatmap for the filtered correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_corr, cmap='coolwarm', center=0, annot=False, fmt=".2f", cbar_kws={'shrink': .5})
plt.title('Filtered Correlation Matrix (Strong Correlations Only)')
plt.show()

# Optional: Print the remaining columns
print("Columns retained after filtering:", filtered_df.columns.tolist())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select only the numeric columns
numeric_df = train_data.select_dtypes(include=[np.number])

# List of columns you want to keep (based on your description)
columns_to_keep = [
    'Mag_U', 'Mag_G', 'Mag_R', 'Mag_I', 'Mag_Z',
    'Petrosian_Radius_U', 'Petrosian_Radius_G', 'Petrosian_Radius_I', 
    'Petrosian_Radius_R', 'Petrosian_Radius_Z',
    'PSF_Mag_U', 'PSF_Mag_R', 'PSF_Mag_G', 'PSF_Mag_I', 'PSF_Mag_Z',
    'Exponential_AB_U', 'Exponential_AB_G', 'Exponential_AB_R',
    'Exponential_AB_I', 'Exponential_AB_Z',
    'Redshift'
]

# Filter the DataFrame to keep only the selected columns
filtered_df = numeric_df[columns_to_keep]

# Recalculate the correlation matrix for the filtered DataFrame
filtered_corr = filtered_df.corr()

# Plot the filtered heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(filtered_corr, cmap='coolwarm', center=0, cbar_kws={'shrink': .5})
plt.title('Filtered Correlation Matrix (Selected Columns Only)')
plt.show()

# Print the columns retained
print("Columns retained after filtering:", filtered_df.columns.tolist())

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(numeric_df)
print(f'Explained variance ratio by principal components: {pca.explained_variance_ratio_}')


import mlflow
import mlflow.sklearn
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Set the tracking URI to point to your MLflow server
mlflow.set_tracking_uri("https://dagshub.com/jaathavan18/EAS503_Project.mlflow")

# Set your credentials (Ensure these are securely managed and not hard-coded in your production code)
os.environ['MLFLOW_TRACKING_USERNAME'] = 'jaathavan18'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '59c1c8d77037a8073e3639cae8918ddfc2970cab'

# Perform stratified train/test split with a seed for reproducibility
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame to retain column names
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Start MLflow experiment
mlflow.set_experiment("EAS503_Project")

# Prepare an input example for logging the model
input_example = X_train[:1]

# Train and log models
results = {}
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        
        # Log the model with an input example
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)
        
        # Save results
        results[model_name] = (accuracy, report)

# Display results
for model_name, (accuracy, report) in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print("="*50)