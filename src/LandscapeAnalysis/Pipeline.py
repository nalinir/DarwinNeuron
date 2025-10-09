################################ Generate Randman Problem ################################
import sqlite3
import pandas as pd

from dataclasses import dataclass
from src.Models import calc_randmansnn_parameters
from src.RandmanFunctions import RandmanConfig, generate_and_save_randman, match_config

@dataclass
class RandmanProblemConfig:
    randman_id: int
    nb_hidden: int
    loss_fn: str

    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT randman_id, nb_hidden, loss_fn FROM randman_problems WHERE id = {id}")
        row = cur.fetchone()
        con.close()
        return cls(*row) 
    
def generate_randman_problem(randman_config: RandmanConfig, nb_hidden=40, loss_fn = 'cross_entropy', db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find matching randman configuration
    randman_df = pd.read_csv("data/randman/meta-data.csv")
    randman_row = match_config(randman_df, randman_config)

    # Generate a new randman if no matching configuration is found
    if randman_row.empty:
        generate_and_save_randman(randman_config, "data/randman")
        randman_df = pd.read_csv("data/randman/meta-data.csv")
        randman_row = match_config(randman_df, randman_config)

    # Select the first (and the only) matching row
    randman_row = randman_row.iloc[0]

    # Calculate the dimension for the problem
    dim = calc_randmansnn_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes)

    # Check if the problem already exists in the database
    cur.execute(f"""
        SELECT id FROM randman_problems 
        WHERE randman_id = {randman_row["id"]} AND nb_hidden = {nb_hidden} AND loss_fn = '{loss_fn}'
    """)
    existing_problem = cur.fetchone()

    if existing_problem:
        con.close()
        raise ValueError(f"Problem with randman_id {randman_row['id']} and nb_hidden {nb_hidden} already exists.")

    # Insert the new problem into the database
    cur.execute(f"""
        INSERT INTO randman_problems (randman_id, nb_hidden, loss_fn, dim) 
        VALUES ({int(randman_row["id"])}, {nb_hidden}, '{loss_fn}', {dim})
    """)

    con.commit()
    con.close()
    
##################################### Generate Parameter Samples #####################################
import uuid, os, sqlite3
import numpy as np
from dataclasses import dataclass
from pflacco.sampling import create_initial_sample
from tqdm import tqdm # Import tqdm for the progress bar

@dataclass
class ParameterSampleConfig:
    dim: int = 3
    nb_sample: int = 1024
    method: str = "sobol"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    
    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT dim, nb_sample, method, lower_bound, upper_bound FROM randman_samples WHERE id={id}")
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")
        return cls(*row)
        
    def read_samples(self, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the sample configuration exists in the database
        cur.execute(f"""
            SELECT filename FROM randman_samples
            WHERE dim = {self.dim} AND nb_sample = {self.nb_sample} AND method = '{self.method}' 
            AND lower_bound = {self.lower_bound} AND upper_bound = {self.upper_bound}
        """)
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError("No dataset found with the specified parameters.")
        
        filename = row[0]
        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        
        data = np.load(filepath)
        return data

def add_samples(sample_config: ParameterSampleConfig, nb_versions=30, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
    os.makedirs(sample_dir, exist_ok=True)
    
    con = None # Initialize con to None for finally block
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the sample configuration already exists in the database
        # Use parameterized queries to prevent SQL injection and handle types correctly
        cur.execute("""
            SELECT version FROM randman_samples
            WHERE dim = ? AND nb_sample = ? 
            AND method = ? AND lower_bound = ? 
            AND upper_bound = ?
        """, (sample_config.dim, sample_config.nb_sample, 
              sample_config.method, sample_config.lower_bound, 
              sample_config.upper_bound))
        
        existing_versions = cur.fetchall()
        
        if not existing_versions:
            min_version = 0
        else:
            flat_versions = [v[0] for v in existing_versions]
            min_version = int(np.max(flat_versions)) + 1
        
        # Loop to generate and save each sample immediately
        for version in tqdm(range(min_version, min_version + nb_versions),
                            desc="Generating Samples"):
            # Create sample
            # Ensure create_initial_sample produces a NumPy array
            sample = create_initial_sample(sample_config.dim, 
                                           n=sample_config.nb_sample, 
                                           lower_bound=sample_config.lower_bound, 
                                           upper_bound=sample_config.upper_bound, 
                                           sample_type=sample_config.method,
                                           seed=version)
            
            # Generate filename
            filename = f"{uuid.uuid4().hex}.npy"
            filepath = os.path.join(sample_dir, filename)
            
            # Save the sample to disk IMMEDIATELY
            np.save(filepath, sample)
            
            # Explicitly delete the sample from memory if it's large and no longer needed
            # This is good practice for large objects inside loops.
            del sample 
            
            # Insert the new sample configuration into the database
            cur.execute("""
                INSERT INTO randman_samples (dim, nb_sample, method, lower_bound, upper_bound, version, filename) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sample_config.dim, sample_config.nb_sample, sample_config.method, 
                  sample_config.lower_bound, sample_config.upper_bound, version, filename))
            
            # Commit after each successful save + DB insert. 
            # This makes the process more robust to crashes (samples are not lost)
            # and prevents accumulating too many changes in memory before a single commit.
            con.commit() 
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if con:
            con.rollback() # Rollback changes if an error occurs
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if con:
            con.rollback()
    finally:
        if con:
            con.close() # Ensure the connection is closed even if errors occur
    
#################################### Match problem and sample ####################################
import os, sqlite3, numpy as np
from dataclasses import dataclass

@dataclass
class LossSurfaceConfig:
    problem_id: int
    sample_id: int

    @classmethod
    def lookup_by_id(cls, id: int, db_path="data/landscape-analysis.db"):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT problem_id, sample_id FROM randman_loss_surfaces WHERE id=?", (id,))
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")
        return cls(*row)

    def read_sample(self, sample_dir="data/samples", db_path="data/landscape-analysis.db"):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT randman_samples.filename FROM randman_samples
            INNER JOIN randman_loss_surfaces ON randman_samples.id=randman_loss_surfaces.sample_id
            WHERE randman_loss_surfaces.problem_id={self.problem_id} AND randman_loss_surfaces.sample_id={self.sample_id}
            """)
        
        filename = cur.fetchone()
        if filename is None:
            raise ValueError(f"No sample found for problem_id {self.problem_id} and sample_id {self.sample_id}.")
        filename = filename[0]
        
        con.close()
        sample = np.load(os.path.join(sample_dir, filename))
        return sample
    
    def read_loss(self, loss_dir="data/losses", db_path="data/landscape-analysis.db"):
        """
        Read the loss associated with this loss surface configuration using the database.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT loss_filename FROM randman_loss_surfaces 
            WHERE problem_id={self.problem_id} AND sample_id={self.sample_id}
        """)
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No loss found for problem_id {self.problem_id} and sample_id {self.sample_id}.")
        
        loss_filename = row[0]
        return np.load(os.path.join(loss_dir, loss_filename))

    def write_loss_filename(self, loss_filename: str, db_path="data/landscape-analysis.db"):
        """
        Write the loss filename to the database for this loss surface configuration.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            UPDATE randman_loss_surfaces
            SET loss_filename = '{loss_filename}' 
            WHERE problem_id = {self.problem_id} AND sample_id = {self.sample_id}
        """)
        
        con.commit()
        con.close()

def assign_samples_to_problem(problem_id: int, sample_config: ParameterSampleConfig, nb_versions=None, sample_dir="data/samples", db_path="data/landscape-analysis.db"):    
    # Connect to the database
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find all the samples which match the sample_config
    cur.execute(f"""
        SELECT randman_samples.id FROM randman_samples
        WHERE dim = {sample_config.dim} AND nb_sample = {sample_config.nb_sample} 
        AND method = '{sample_config.method}' AND lower_bound = {sample_config.lower_bound} 
        AND upper_bound = {sample_config.upper_bound}
        AND randman_samples.id NOT IN (SELECT sample_id FROM randman_loss_surfaces WHERE problem_id = {problem_id})
    """)
    sample_ids = [row[0] for row in cur.fetchall()]
    
    # Make sure the samples exist
    if len(sample_ids) == 0:
        con.close()
        raise ValueError(f"No unassigned samples found matching the config {sample_config}. Please generate samples first.")

    # Add the matched samples, along with problem_id, to the loss_surfaces table

    # Insert new rows into the loss_surfaces table
    new_rows = [(problem_id, sample_id) for sample_id in sample_ids]
    cur.executemany("""
        INSERT INTO randman_loss_surfaces (problem_id, sample_id) 
        VALUES (?, ?)
    """, new_rows)
    
    # print total number of rows inserted, and current number of rows for the problem
    cur.execute(f"SELECT COUNT(*) FROM randman_loss_surfaces WHERE problem_id = {problem_id}")
    total_samples = cur.fetchone()[0]
    print(f"Problem {problem_id}: Assigned {len(new_rows)} randman_samples. Total samples: {total_samples}")

    # Commit and close the connection
    con.commit()
    con.close()
    
################################### Calculate Loss ###################################
import time
def get_next_available_id(column_name, db_path='data/landscape-analysis.db', pending_timeout_minutes=60):
    """
    This function has been made more robust to handle race conditions in a parallel environment.
    It now checks for and adds the required columns in a more resilient way.
    """
    con = sqlite3.connect(db_path, timeout=15)
    cur = con.cursor()
    
    cur.execute("BEGIN EXCLUSIVE")

    try:
        cur.execute(f"ALTER TABLE randman_loss_surfaces ADD COLUMN {column_name} REAL")
        print(f"Column '{column_name}' added to 'loss_surfaces' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise

    try:
        cur.execute("ALTER TABLE randman_loss_surfaces ADD COLUMN pending_timestamp REAL")
        print("Column 'pending_timestamp' added to 'loss_surfaces' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise
    
    try:
        cur.execute("ALTER TABLE randman_loss_surfaces ADD COLUMN status TEXT")
        print("Column 'status' added to 'loss_surfaces' table.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise
    
    con.commit()

    cur.execute("BEGIN EXCLUSIVE")

    timeout_threshold = time.time() - (pending_timeout_minutes * 60)
    print(f"Checking for orphaned 'pending' jobs with a timeout threshold of {timeout_threshold}")
    cur.execute(f"""
        UPDATE randman_loss_surfaces
        SET status = NULL
        WHERE status = 'pending' AND (pending_timestamp IS NULL OR pending_timestamp < ?)
    """, (timeout_threshold,))
    print(f"Reclaimed {cur.rowcount} pending jobs.")
    
    if column_name == "loss_filename":
        # Manually adjusted to prioritize recurrent models
        cur.execute(f"""
            SELECT MIN(a.id) FROM randman_loss_surfaces a
            JOIN randman_problems b on a.problem_id = b.id 
            WHERE a.{column_name} IS NULL AND (a.status != 'pending' or a.status IS NULL)
            and b.recurrent = 1 and b.dim = 500 and b.model_path is not NULL
            """) # We should add this filter up top eventually
    else:
        cur.execute(f"""
            SELECT MIN(a.id) FROM randman_loss_surfaces a
            JOIN randman_problems b on a.problem_id = b.id 
            WHERE {column_name} IS NULL AND (a.status != 'pending' or a.status IS NULL)
            AND a.loss_filename IS NOT NULL
            AND b.recurrent = 1 AND b.dim = 500 and b.model_path is not NULL
            """)
    
    id = cur.fetchone()[0]
    print(f"Found next available ID: {id}")
    
    if id is not None:
        cur.execute(f"""
            UPDATE randman_loss_surfaces
            SET status = 'pending', pending_timestamp = ?
            WHERE id = ?
        """, (time.time(), id))
    
    con.commit()
    con.close()
    
    return id
############################################ Calculate Features ############################################
from typing import Callable
def get_xy_by_id(loss_surface_id: int, loss_dir="data/losses", sample_dir="data/samples", db_path="data/landscape-analysis.db"):
    """
    Get the x and y values for a given loss surface ID.
    """
    loss_surface_config = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
    x = loss_surface_config.read_sample(sample_dir)
    y = loss_surface_config.read_loss(loss_dir, db_path)
    return x, y  

def calculate_and_save_features(loss_surface_id: int, sample_to_feature: Callable, loss_dir="data/losses", sample_dir="data/samples", db_path="data/landscape-analysis.db"):
    """
    Calculates features for a given loss surface and saves all results to the database.
    This version dynamically builds the UPDATE query to handle all features.
    Includes retry logic with exponential backoff and randomized delay for database writes.
    """
    x, y = get_xy_by_id(loss_surface_id, loss_dir, sample_dir, db_path)
    
    # Calculate features
    print(f"Calculating features for loss_surface_id {loss_surface_id} with {len(x)} samples")
    features = sample_to_feature(x, y)
    
    # Replace all "." in features' keys with "_" for valid column names
    features = {key.replace('.', '_'): value for key, value in features.items()}
    
    max_retries = 5
    base_delay = 1.0 # seconds
    
    for attempt in range(max_retries):
        conn = None
        try:
            # Connect to the SQLite database with a timeout for busy errors
            # The timeout is in seconds.
            conn = sqlite3.connect(db_path, timeout=10.0) 
            cur = conn.cursor()
            
            # Acquire write lock
            cur.execute("BEGIN EXCLUSIVE")

            # Check and add columns if they don't exist
            cur.execute(f"PRAGMA table_info(randman_loss_surfaces)")
            columns = [info[1] for info in cur.fetchall()]
            
            for key, value in features.items():
                if key not in columns:
                    # Dynamically determine the column type based on the value's Python type
                    column_type = 'TEXT' if isinstance(value, str) else 'REAL'
                    cur.execute(f"ALTER TABLE randman_loss_surfaces ADD COLUMN {key} {column_type}")
                    print(f"Added column {key} to randman_loss_surfaces table with type {column_type}.")

            # Dynamically build the UPDATE query for all features and set the status to 'complete'
            update_columns = list(features.keys())
            set_clause = ', '.join([f"{col} = ?" for col in update_columns]) + ", status = 'complete'"
            update_query = f"UPDATE randman_loss_surfaces SET {set_clause} WHERE id = ?"
            
            # Prepare the values for the query
            update_values = list(features.values())
            update_values.append(loss_surface_id)

            cur.execute(update_query, tuple(update_values))
            conn.commit()
            print(f"Successfully saved features for loss_surface_id {loss_surface_id}.")
            return # Success, exit the retry loop
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5 * base_delay)
                print(f"Database locked for ID {loss_surface_id}. Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{max_retries})...")
                if conn:
                    conn.rollback() # Rollback the transaction on lock error
                    conn.close() # Close the connection before retrying
                time.sleep(delay)
            else:
                print(f"An unexpected SQLite error occurred for ID {loss_surface_id}: {e}")
                if conn:
                    conn.rollback()
                    conn.close()
                raise # Re-raise other operational errors
        except Exception as e:
            print(f"An error occurred during feature saving for ID {loss_surface_id}: {e}")
            if conn:
                conn.rollback()
                conn.close()
            raise # Re-raise other exceptions
        finally:
            if conn:
                # Ensure connection is closed if not already closed by rollback/retry logic
                pass # Connection is closed by `with` statement or explicitly in retry logic
    
    print(f"Failed to save features for loss_surface_id {loss_surface_id} after {max_retries} attempts.")
    raise Exception(f"Failed to save features for ID {loss_surface_id} due to persistent database locking.")


def reclaim_pending_jobs(column_name: str, db_path='data/landscape-analysis.db'):
    """
    Reclaims any jobs left in a 'pending' state. This is useful for cleaning up
    after a script run finishes, especially if there were failures in the last batch.
    """
    con = sqlite3.connect(db_path, timeout=15)
    cur = con.cursor()
    try:
        cur.execute("BEGIN EXCLUSIVE")
        cur.execute(f"UPDATE randman_loss_surfaces SET {column_name} = NULL WHERE {column_name} = 'pending'")
        con.commit()
    except sqlite3.OperationalError as e:
        print(f"Error during final job reclamation: {e}")
        con.rollback()
    finally:
        con.close()
