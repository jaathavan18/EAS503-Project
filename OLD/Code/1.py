import sqlite3
import pandas as pd
import csv

# Provided functions
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
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows

# Database and CSV file paths
db_file = 'example.db'
csv_file = 'data.csv'

# Step 1: Create and normalize the database
conn = create_connection(db_file, delete_db=True)

# Create tables
create_table(conn, '''CREATE TABLE IF NOT EXISTS objects (
                         objid INTEGER PRIMARY KEY,
                         specobjid INTEGER,
                         ra REAL,
                         dec REAL,
                         redshift REAL,
                         class TEXT
                     );''')

create_table(conn, '''CREATE TABLE IF NOT EXISTS magnitudes (
                         objid INTEGER,
                         u REAL,
                         g REAL,
                         r REAL,
                         i REAL,
                         z REAL,
                         FOREIGN KEY(objid) REFERENCES objects(objid)
                     );''')

create_table(conn, '''CREATE TABLE IF NOT EXISTS observations (
                         objid INTEGER,
                         run INTEGER,
                         rerun INTEGER,
                         camcol INTEGER,
                         field INTEGER,
                         plate INTEGER,
                         mjd INTEGER,
                         fiberid INTEGER,
                         FOREIGN KEY(objid) REFERENCES objects(objid)
                     );''')

create_table(conn, '''CREATE TABLE IF NOT EXISTS petrosian_properties (
                         objid INTEGER,
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
                         FOREIGN KEY(objid) REFERENCES objects(objid)
                     );''')

create_table(conn, '''CREATE TABLE IF NOT EXISTS psf_magnitudes (
                         objid INTEGER,
                         psfMag_u REAL,
                         psfMag_g REAL,
                         psfMag_r REAL,
                         psfMag_i REAL,
                         psfMag_z REAL,
                         FOREIGN KEY(objid) REFERENCES objects(objid)
                     );''')

create_table(conn, '''CREATE TABLE IF NOT EXISTS exp_ab_ratios (
                         objid INTEGER,
                         expAB_u REAL,
                         expAB_g REAL,
                         expAB_r REAL,
                         expAB_i REAL,
                         expAB_z REAL,
                         FOREIGN KEY(objid) REFERENCES objects(objid)
                     );''')

# Parse CSV and insert data
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        c = conn.cursor()
        # Insert into objects table
        c.execute('INSERT OR IGNORE INTO objects (objid, specobjid, ra, dec, redshift, class) VALUES (?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['specobjid'], row['ra'], row['dec'], row['redshift'], row['class']))
        # Insert into magnitudes table
        c.execute('INSERT INTO magnitudes (objid, u, g, r, i, z) VALUES (?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['u'], row['g'], row['r'], row['i'], row['z']))
        # Insert into observations table
        c.execute('INSERT INTO observations (objid, run, rerun, camcol, field, plate, mjd, fiberid) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['run'], row['rerun'], row['camcol'], row['field'], row['plate'], row['mjd'], row['fiberid']))
        # Insert into petrosian_properties table
        c.execute('INSERT INTO petrosian_properties (objid, petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, petroFlux_u, petroFlux_g, petroFlux_r, petroFlux_i, petroFlux_z, petroR50_u, petroR50_g, petroR50_r, petroR50_i, petroR50_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['petroRad_u'], row['petroRad_g'], row['petroRad_r'], row['petroRad_i'], row['petroRad_z'], row['petroFlux_u'], row['petroFlux_g'], row['petroFlux_r'], row['petroFlux_i'], row['petroFlux_z'], row['petroR50_u'], row['petroR50_g'], row['petroR50_r'], row['petroR50_i'], row['petroR50_z']))
        # Insert into psf_magnitudes table
        c.execute('INSERT INTO psf_magnitudes (objid, psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z) VALUES (?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['psfMag_u'], row['psfMag_g'], row['psfMag_r'], row['psfMag_i'], row['psfMag_z']))
        # Insert into exp_ab_ratios table
        c.execute('INSERT INTO exp_ab_ratios (objid, expAB_u, expAB_g, expAB_r, expAB_i, expAB_z) VALUES (?, ?, ?, ?, ?, ?)',
                  (row['objid'], row['expAB_u'], row['expAB_g'], row['expAB_r'], row['expAB_i'], row['expAB_z']))

# Commit changes
conn.commit()

# Step 2: Write an SQL query with joins to reconstruct the data and load it into a Pandas DataFrame
query = '''
SELECT o.objid, o.specobjid, o.ra, o.dec, o.redshift, o.class,
       m.u, m.g, m.r, m.i, m.z,
       obs.run, obs.rerun, obs.camcol, obs.field, obs.plate, obs.mjd, obs.fiberid,
       p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, 
       p.petroFlux_u, p.petroFlux_g, p.petroFlux_r, p.petroFlux_i, p.petroFlux_z, 
       p.petroR50_u, p.petroR50_g, p.petroR50_r, p.petroR50_i, p.petroR50_z,
       ps.psfMag_u, ps.psfMag_g, ps.psfMag_r, ps.psfMag_i, ps.psfMag_z,
       ex.expAB_u, ex.expAB_g, ex.expAB_r, ex.expAB_i, ex.expAB_z
FROM objects o
JOIN magnitudes m ON o.objid = m.objid
JOIN observations obs ON o.objid = obs.objid
JOIN petrosian_properties p ON o.objid = p.objid
JOIN psf_magnitudes ps ON o.objid = ps.objid
JOIN exp_ab_ratios ex ON o.objid = ex.objid
'''

# Execute query and load into Pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Display the DataFrame
print(df.head())



############################################################ CLAUDE CODE with comments

import os
import sqlite3
import pandas as pd

def create_connection(db_file, delete_db=False):
    """
    Create a database connection and ensure foreign key support
    
    Args:
        db_file (str): Path to the SQLite database file
        delete_db (bool): Whether to delete existing database
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        # Delete existing database if specified
        if delete_db and os.path.exists(db_file):
            os.remove(db_file)
        
        # Establish connection with foreign key support
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def create_tables(conn):
    """
    Create all necessary tables in the database
    
    Args:
        conn (sqlite3.Connection): Database connection object
    """
    # Define table creation SQL statements
    table_schemas = [
        '''CREATE TABLE IF NOT EXISTS objects (
            objid INTEGER PRIMARY KEY,
            specobjid INTEGER,
            ra REAL,
            dec REAL,
            redshift REAL,
            class TEXT
        )''',
        
        '''CREATE TABLE IF NOT EXISTS magnitudes (
            objid INTEGER PRIMARY KEY,
            u REAL,
            g REAL,
            r REAL,
            i REAL,
            z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS observations (
            objid INTEGER PRIMARY KEY,
            run INTEGER,
            rerun INTEGER,
            camcol INTEGER,
            field INTEGER,
            plate INTEGER,
            mjd INTEGER,
            fiberid INTEGER,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS petrosian_properties (
            objid INTEGER PRIMARY KEY,
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
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS psf_magnitudes (
            objid INTEGER PRIMARY KEY,
            psfMag_u REAL,
            psfMag_g REAL,
            psfMag_r REAL,
            psfMag_i REAL,
            psfMag_z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS exp_ab_ratios (
            objid INTEGER PRIMARY KEY,
            expAB_u REAL,
            expAB_g REAL,
            expAB_r REAL,
            expAB_i REAL,
            expAB_z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )'''
    ]
    
    # Execute table creation
    try:
        cursor = conn.cursor()
        for schema in table_schemas:
            cursor.execute(schema)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Table creation error: {e}")

def insert_data(conn, df):
    """
    Insert data from DataFrame into database tables
    
    Args:
        conn (sqlite3.Connection): Database connection object
        df (pd.DataFrame): Input DataFrame
    """
    try:
        # Use pandas to_sql for efficient bulk insertion
        df.set_index('objid', inplace=True)
        
        # Insert into each table
        df[['specobjid', 'ra', 'dec', 'redshift', 'class']].to_sql('objects', conn, if_exists='replace', index=True)
        
        # Magnitude columns
        df[['u', 'g', 'r', 'i', 'z']].to_sql('magnitudes', conn, if_exists='replace', index=True)
        
        # Observation columns
        df[['run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']].to_sql('observations', conn, if_exists='replace', index=True)
        
        # Petrosian properties
        petrosian_cols = [col for col in df.columns if col.startswith('petroRad_') or 
                          col.startswith('petroFlux_') or 
                          col.startswith('petroR50_')]
        df[petrosian_cols].to_sql('petrosian_properties', conn, if_exists='replace', index=True)
        
        # PSF Magnitudes
        psf_cols = [col for col in df.columns if col.startswith('psfMag_')]
        df[psf_cols].to_sql('psf_magnitudes', conn, if_exists='replace', index=True)
        
        # Exponential AB ratios
        expab_cols = [col for col in df.columns if col.startswith('expAB_')]
        df[expab_cols].to_sql('exp_ab_ratios', conn, if_exists='replace', index=True)
        
        conn.commit()
    except Exception as e:
        print(f"Data insertion error: {e}")
        conn.rollback()

def main(csv_file='data.csv', db_file='astronomical.db'):
    """
    Main function to process astronomical data
    
    Args:
        csv_file (str): Path to input CSV file
        db_file (str): Path to output SQLite database
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        
        # Print initial class distribution
        print("Initial Class Distribution:")
        print(df['class'].value_counts())
        
        # Create database connection
        conn = create_connection(db_file, delete_db=True)
        
        if conn:
            # Create database tables
            create_tables(conn)
            
            # Insert data
            insert_data(conn, df)
            
            # Verify data insertion by class
            class_counts_query = 'SELECT class, COUNT(*) as count FROM objects GROUP BY class'
            class_counts = pd.read_sql_query(class_counts_query, conn)
            
            print("\nClass Distribution in Database:")
            print(class_counts)
            
            # Close connection
            conn.close()
    
    except Exception as e:
        print(f"Processing error: {e}")

if __name__ == "__main__":
    main()


############################################################ CLAUDE CODE without comments
import os
import sqlite3
import pandas as pd

def create_connection(db_file, delete_db=False):
    try:
        if delete_db and os.path.exists(db_file):
            os.remove(db_file)
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def create_tables(conn):
    table_schemas = [
        '''CREATE TABLE IF NOT EXISTS objects (
            objid INTEGER PRIMARY KEY,
            specobjid INTEGER,
            ra REAL,
            dec REAL,
            redshift REAL,
            class TEXT
        )''',
        
        '''CREATE TABLE IF NOT EXISTS magnitudes (
            objid INTEGER PRIMARY KEY,
            u REAL,
            g REAL,
            r REAL,
            i REAL,
            z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS observations (
            objid INTEGER PRIMARY KEY,
            run INTEGER,
            rerun INTEGER,
            camcol INTEGER,
            field INTEGER,
            plate INTEGER,
            mjd INTEGER,
            fiberid INTEGER,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS petrosian_properties (
            objid INTEGER PRIMARY KEY,
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
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS psf_magnitudes (
            objid INTEGER PRIMARY KEY,
            psfMag_u REAL,
            psfMag_g REAL,
            psfMag_r REAL,
            psfMag_i REAL,
            psfMag_z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )''',
        
        '''CREATE TABLE IF NOT EXISTS exp_ab_ratios (
            objid INTEGER PRIMARY KEY,
            expAB_u REAL,
            expAB_g REAL,
            expAB_r REAL,
            expAB_i REAL,
            expAB_z REAL,
            FOREIGN KEY(objid) REFERENCES objects(objid)
        )'''
    ]
    
    try:
        cursor = conn.cursor()
        for schema in table_schemas:
            cursor.execute(schema)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Table creation error: {e}")

def insert_data(conn, df):
    try:
        df.set_index('objid', inplace=True)
        df[['specobjid', 'ra', 'dec', 'redshift', 'class']].to_sql('objects', conn, if_exists='replace', index=True)
        df[['u', 'g', 'r', 'i', 'z']].to_sql('magnitudes', conn, if_exists='replace', index=True)
        df[['run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']].to_sql('observations', conn, if_exists='replace', index=True)
        petrosian_cols = [col for col in df.columns if col.startswith('petroRad_') or 
                          col.startswith('petroFlux_') or 
                          col.startswith('petroR50_')]
        df[petrosian_cols].to_sql('petrosian_properties', conn, if_exists='replace', index=True)
        psf_cols = [col for col in df.columns if col.startswith('psfMag_')]
        df[psf_cols].to_sql('psf_magnitudes', conn, if_exists='replace', index=True)
        expab_cols = [col for col in df.columns if col.startswith('expAB_')]
        df[expab_cols].to_sql('exp_ab_ratios', conn, if_exists='replace', index=True)
        conn.commit()
    except Exception as e:
        print(f"Data insertion error: {e}")
        conn.rollback()

def main(csv_file='data.csv', db_file='astronomical.db'):
    try:
        df = pd.read_csv(csv_file)
        print("Initial Class Distribution:")
        print(df['class'].value_counts())
        conn = create_connection(db_file, delete_db=True)
        if conn:
            create_tables(conn)
            insert_data(conn, df)
            class_counts_query = 'SELECT class, COUNT(*) as count FROM objects GROUP BY class'
            class_counts = pd.read_sql_query(class_counts_query, conn)
            print("\nClass Distribution in Database:")
            print(class_counts)
            conn.close()
    except Exception as e:
        print(f"Processing error: {e}")

if __name__ == "__main__":
    main()


############################

import sqlite3
import pandas as pd
import numpy as np
from joblib import Parallel, delayed  # Import joblib for parallel processing

# (Existing code for database operations and data preparation)

# Display the first few rows of the DataFrame with new column names
print(df.head(10))

print(df.dtypes)
print(df.describe())

# Check for any missing values
print(df.isnull().sum())
print(df['Class'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

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

# Function to train and evaluate a model
def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model_name, accuracy, report

# Train and evaluate models in parallel
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate_model)(model_name, model, X_train, y_train, X_test, y_test) for model_name, model in models.items())

# Display results
for model_name, accuracy, report in results:
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print("="*50)