import pandas as pd
import sqlite3
from sqlite3 import Error

def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql, drop_table_name=None):
    if drop_table_name:
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)
    
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)
    return cur.fetchall()

# Create database connection
conn = create_connection("SDSS_DR18.db", delete_db=True)

# Define and create tables
if conn is not None:
    # Create observation_info table
    create_observation_info = """
    CREATE TABLE IF NOT EXISTS observation_info (
        observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        specobjid BIGINT NOT NULL,
        ra REAL,
        dec REAL,
        redshift REAL
    );
    """
    
    # Create observation_details table
    create_observation_details = """
    CREATE TABLE IF NOT EXISTS observation_details (
        detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_id INTEGER NOT NULL,
        run INTEGER,
        rerun INTEGER,
        camcol INTEGER,
        field INTEGER,
        plate INTEGER,
        mjd INTEGER,
        fiberid INTEGER,
        FOREIGN KEY (observation_id) REFERENCES observation_info(observation_id)
    );
    """
    
    # Create photometric_bands table
    create_bands = """
    CREATE TABLE IF NOT EXISTS photometric_bands (
        band_id INTEGER PRIMARY KEY AUTOINCREMENT,
        band_name TEXT NOT NULL,
        description TEXT
    );
    """
    
    # Create photometric_measurements table
    create_photometric = """
    CREATE TABLE IF NOT EXISTS photometric_measurements (
        measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_id INTEGER NOT NULL,
        band_id INTEGER NOT NULL,
        petro_flux REAL,
        petro_rad REAL,
        petro_r50 REAL,
        psf_mag REAL,
        exp_ab REAL,
        FOREIGN KEY (observation_id) REFERENCES observation_info(observation_id),
        FOREIGN KEY (band_id) REFERENCES photometric_bands(band_id)
    );
    """
    
    # Create classifications table
    create_classifications = """
    CREATE TABLE IF NOT EXISTS classifications (
        classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        observation_id INTEGER NOT NULL,
        class_type TEXT NOT NULL,
        FOREIGN KEY (observation_id) REFERENCES observation_info(observation_id)
    );
    """
    
    # Create all tables
    create_table(conn, create_observation_info)
    create_table(conn, create_observation_details)
    create_table(conn, create_bands)
    create_table(conn, create_photometric)
    create_table(conn, create_classifications)
    
    # Insert photometric bands
    bands_insert = """
    INSERT INTO photometric_bands (band_name, description) VALUES
    ('u', 'ultraviolet band'),
    ('g', 'green band'),
    ('r', 'red band'),
    ('i', 'infrared band'),
    ('z', 'near infrared band');
    """
    
    execute_sql_statement(bands_insert, conn)
    conn.commit()

# Function to load data from CSV to normalized database
def load_sdss_data(csv_file, conn):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Insert observation_info
    observation_info = df[['specobjid', 'ra', 'dec', 'redshift']].drop_duplicates()
    observation_info.to_sql('observation_info', conn, if_exists='append', index=False)
    
    # Get observation IDs mapping
    obs_ids = pd.read_sql("SELECT observation_id, specobjid FROM observation_info", conn)
    
    # Insert observation_details
    observation_details = pd.merge(
        df[['specobjid', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']],
        obs_ids,
        on='specobjid'
    )
    details_to_insert = observation_details.drop('specobjid', axis=1)
    details_to_insert.to_sql('observation_details', conn, if_exists='append', index=False)
    
    # Insert classifications
    classifications = pd.merge(
        df[['specobjid', 'class']],
        obs_ids,
        on='specobjid'
    )[['observation_id', 'class']]
    classifications.columns = ['observation_id', 'class_type']
    classifications.to_sql('classifications', conn, if_exists='append', index=False)
    
    # Insert photometric measurements
    bands = {
        'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5
    }
    
    for band, band_id in bands.items():
        measurements = pd.merge(
            df[[
                'specobjid',
                f'petroFlux_{band}',
                f'petroRad_{band}',
                f'petroR50_{band}',
                f'psfMag_{band}',
                f'expAB_{band}'
            ]],
            obs_ids,
            on='specobjid'
        )
        
        measurements_data = pd.DataFrame({
            'observation_id': measurements['observation_id'],
            'band_id': band_id,
            'petro_flux': measurements[f'petroFlux_{band}'],
            'petro_rad': measurements[f'petroRad_{band}'],
            'petro_r50': measurements[f'petroR50_{band}'],
            'psf_mag': measurements[f'psfMag_{band}'],
            'exp_ab': measurements[f'expAB_{band}']
        })
        
        measurements_data.to_sql('photometric_measurements', conn, 
                               if_exists='append', index=False)
    
    conn.commit()

# Load the data
csv_file = "SDSS_DR18.csv"
load_sdss_data(csv_file, conn)
conn.close()