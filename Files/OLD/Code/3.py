# import pandas as pd
# import sqlite3
# from sqlite3 import Error

# def create_connection(db_file, delete_db=False):
#     import os
#     if delete_db and os.path.exists(db_file):
#         os.remove(db_file)

#     conn = None
#     try:
#         conn = sqlite3.connect(db_file)
#         conn.execute("PRAGMA foreign_keys = 1")
#     except Error as e:
#         print(e)

#     return conn


# def create_table(conn, create_table_sql, drop_table_name=None):
    
#     if drop_table_name: # You can optionally pass drop_table_name to drop the table. 
#         try:
#             c = conn.cursor()
#             c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
#         except Error as e:
#             print(e)
    
#     try:
#         c = conn.cursor()
#         c.execute(create_table_sql)
#     except Error as e:
#         print(e)
        
# def execute_sql_statement(sql_statement, conn):
#     cur = conn.cursor()
#     cur.execute(sql_statement)

#     rows = cur.fetchall()

#     return rows

# db_file = 'example.db'
# csv_file = 'data.csv'

# # Step 1: Create normalized database schema (3NF)
# conn = sqlite3.connect(db_file)
# cursor = conn.cursor()

# # Drop tables if they already exist to start fresh
# cursor.execute('DROP TABLE IF EXISTS objects')
# cursor.execute('DROP TABLE IF EXISTS magnitudes')
# cursor.execute('DROP TABLE IF EXISTS observations')
# cursor.execute('DROP TABLE IF EXISTS petrosian_properties')
# cursor.execute('DROP TABLE IF EXISTS psf_magnitudes')
# cursor.execute('DROP TABLE IF EXISTS exp_ab_ratios')

# # Create tables
# cursor.execute('''
# CREATE TABLE objects (
#     objid TEXT PRIMARY KEY,
#     specobjid TEXT,
#     ra REAL,
#     dec REAL,
#     redshift REAL,
#     class TEXT
# )
# ''')

# cursor.execute('''
# CREATE TABLE magnitudes (
#     objid TEXT PRIMARY KEY,
#     u REAL,
#     g REAL,
#     r REAL,
#     i REAL,
#     z REAL,
#     FOREIGN KEY (objid) REFERENCES objects(objid)
# )
# ''')

# cursor.execute('''
# CREATE TABLE observations (
#     objid TEXT PRIMARY KEY,
#     run INTEGER,
#     rerun INTEGER,
#     camcol INTEGER,
#     field INTEGER,
#     plate INTEGER,
#     mjd INTEGER,
#     fiberid INTEGER,
#     FOREIGN KEY (objid) REFERENCES objects(objid)
# )
# ''')

# cursor.execute('''
# CREATE TABLE petrosian_properties (
#     objid TEXT PRIMARY KEY,
#     petroRad_u REAL,
#     petroRad_g REAL,
#     petroRad_r REAL,
#     petroRad_i REAL,
#     petroRad_z REAL,
#     petroFlux_u REAL,
#     petroFlux_g REAL,
#     petroFlux_r REAL,
#     petroFlux_i REAL,
#     petroFlux_z REAL,
#     petroR50_u REAL,
#     petroR50_g REAL,
#     petroR50_r REAL,
#     petroR50_i REAL,
#     petroR50_z REAL,
#     FOREIGN KEY (objid) REFERENCES objects(objid)
# )
# ''')

# cursor.execute('''
# CREATE TABLE psf_magnitudes (
#     objid TEXT PRIMARY KEY,
#     psfMag_u REAL,
#     psfMag_g REAL,
#     psfMag_r REAL,
#     psfMag_i REAL,
#     psfMag_z REAL,
#     FOREIGN KEY (objid) REFERENCES objects(objid)
# )
# ''')

# cursor.execute('''
# CREATE TABLE exp_ab_ratios (
#     objid TEXT PRIMARY KEY,
#     expAB_u REAL,
#     expAB_g REAL,
#     expAB_r REAL,
#     expAB_i REAL,
#     expAB_z REAL,
#     FOREIGN KEY (objid) REFERENCES objects(objid)
# )
# ''')

# conn.commit()

# df = pd.read_csv(csv_file)

# # Step 3: Insert data into the database
# for index, row in df.iterrows():
#     try:
#         cursor.execute('''
#         INSERT OR IGNORE INTO objects (objid, specobjid, ra, dec, redshift, class) VALUES (?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['specobjid'], row['ra'], row['dec'], row['redshift'], row['class']))
        
#         cursor.execute('''
#         INSERT OR IGNORE INTO magnitudes (objid, u, g, r, i, z) VALUES (?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['u'], row['g'], row['r'], row['i'], row['z']))
        
#         cursor.execute('''
#         INSERT OR IGNORE INTO observations (objid, run, rerun, camcol, field, plate, mjd, fiberid) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['run'], row['rerun'], row['camcol'], row['field'], row['plate'], row['mjd'], row['fiberid']))
        
#         cursor.execute('''
#         INSERT OR IGNORE INTO petrosian_properties (objid, petroRad_u, petroRad_g, petroRad_r, petroRad_i, petroRad_z, petroFlux_u, petroFlux_g, petroFlux_r, petroFlux_i, petroFlux_z, petroR50_u, petroR50_g, petroR50_r, petroR50_i, petroR50_z) 
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['petroRad_u'], row['petroRad_g'], row['petroRad_r'], row['petroRad_i'], row['petroRad_z'],
#               row['petroFlux_u'], row['petroFlux_g'], row['petroFlux_r'], row['petroFlux_i'], row['petroFlux_z'],
#               row['petroR50_u'], row['petroR50_g'], row['petroR50_r'], row['petroR50_i'], row['petroR50_z']))
        
#         cursor.execute('''
#         INSERT OR IGNORE INTO psf_magnitudes (objid, psfMag_u, psfMag_g, psfMag_r, psfMag_i, psfMag_z) VALUES (?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['psfMag_u'], row['psfMag_g'], row['psfMag_r'], row['psfMag_i'], row['psfMag_z']))
        
#         cursor.execute('''
#         INSERT OR IGNORE INTO exp_ab_ratios (objid, expAB_u, expAB_g, expAB_r, expAB_i, expAB_z) VALUES (?, ?, ?, ?, ?, ?)
#         ''', (row['objid'], row['expAB_u'], row['expAB_g'], row['expAB_r'], row['expAB_i'], row['expAB_z']))
#     except Exception as e:
#         print(f"Error inserting row {index}: {e}")
# conn.commit()