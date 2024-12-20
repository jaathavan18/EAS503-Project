# data_loader.py
import sqlite3
import pandas as pd

def connect_to_database(db_path='mobile_phones.db'):
    """
    Create a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def load_mobile_data(conn):
    """
    Load mobile phone data from the database using a comprehensive query.
    """
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
    
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Error loading data: {e}")
        return None
    finally:
        conn.close()

def get_mobile_data(db_path='mobile_phones.db'):
    """
    Convenience function to get mobile data in one step.
    """
    conn = connect_to_database(db_path)
    if conn is not None:
        return load_mobile_data(conn)
    return None