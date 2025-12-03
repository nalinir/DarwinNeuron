import torch
import numpy as np
import os
import sqlite3
from src.LandscapeAnalysis.Pipeline import ParameterSampleConfig
from pflacco.sampling import create_initial_sample
import uuid

def state_dict_to_vector(state_dict):
    return torch.cat([param.view(-1) for param in state_dict.values()])

def get_vectorized_params(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # Assume the checkpoint is the state_dict itself
        state_dict = checkpoint
    
    optimal_vector = state_dict_to_vector(state_dict)
    return optimal_vector
def add_samples_existing(model_path: str, problem_id: int, sample_config: ParameterSampleConfig, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
    os.makedirs(sample_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return  # Skip this model

    con = None # Initialize con to None for finally block
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Add a model_path column that will be called and unique across models.
        # The model path determines the lower vs. upper bound

        # Add model_path column only if it doesn't exist
        cur.execute("PRAGMA table_info(randman_samples)")
        columns = [row[1] for row in cur.fetchall()]
        if "model_path" not in columns:
            cur.execute("""
            ALTER TABLE randman_samples ADD COLUMN model_path TEXT
            """)
        # Check if the sample configuration already exists in the database
        # Use parameterized queries to prevent SQL injection and handle types correctly
        optimal_vector = get_vectorized_params(model_path)

        cur.execute("""
            SELECT version FROM randman_samples
            WHERE dim = ? AND nb_sample = ? 
            AND method = ? AND lower_bound = ? 
            AND upper_bound = ? AND model_path = ?
        """, (sample_config.dim, sample_config.nb_sample, 
              sample_config.method, None, 
              None, model_path))

        existing_versions = cur.fetchall()
        
        if not existing_versions:
            min_version = 0
        else:
            flat_versions = [v[0] for v in existing_versions]
            min_version = int(np.max(flat_versions)) + 1

        # Set the lower and upper bounds for the new samples
        offset = 0.1 * torch.abs(optimal_vector)
        lower_bound = optimal_vector - offset
        upper_bound = optimal_vector + offset
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        print(f"{lower_bound.numpy().shape}, {upper_bound.numpy().shape}")

        # Loop to generate and save each sample immediately
        # Create sample
        # Ensure create_initial_sample produces a NumPy array
        sample = create_initial_sample(sample_config.dim, 
                                    n=sample_config.nb_sample, 
                                    lower_bound=lower_bound.numpy(), 
                                    upper_bound=upper_bound.numpy(), 
                                    sample_type=sample_config.method,
                                    seed=42)
        
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
            INSERT INTO randman_samples (dim, nb_sample, method, lower_bound, upper_bound, version, filename, model_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (sample_config.dim, sample_config.nb_sample, sample_config.method, 
            -9999, 9999, 9999, filename, model_path))
        # Insert new row into the loss_surfaces table
        sample_id = cur.lastrowid

        # Now, you can use sample_id in your next query
        cur.execute("""
            INSERT INTO randman_loss_surfaces (problem_id, sample_id) 
            VALUES (?, ?)
        """, (problem_id, sample_id))
        # Update the other one
        
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
