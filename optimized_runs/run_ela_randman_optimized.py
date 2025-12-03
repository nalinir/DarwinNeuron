from pflacco.classical_ela_features import *
from src.LandscapeAnalysis import get_next_available_id, calculate_and_save_features, reclaim_pending_jobs
from tqdm import tqdm
import os
from typing import Callable, Iterable
from multiprocessing import Pool
from functools import partial
import sqlite3
import argparse

def get_optimal_process_count():
    """
    Calculate optimal number of processes based on available cores and concurrent jobs.
    Assumes you're running multiple jobs simultaneously.
    """
    total_cores = os.cpu_count()
    # Reserve some cores for system and other processes
    # Divide by estimated number of concurrent jobs (adjust as needed)
    estimated_concurrent_jobs = int(os.getenv('SLURM_ARRAY_TASK_COUNT', '10'))  # Default to 10 if not in SLURM
    cores_per_job = max(1, total_cores // estimated_concurrent_jobs)
    return min(cores_per_job, 4)  # Cap at 4 cores per job for efficiency

def run_main_feature_calculation(feature_column_name: str, feature_function: Callable, batch_size: int = None, pending_timeout_minutes=60):
    """
    Main function to orchestrate the feature calculation, combining sequential ID fetching
    with parallel processing for batches.
    """
    if batch_size is None:
        batch_size = get_optimal_process_count()
        print(f"Using batch size: {batch_size}")

    all_ids_to_process = []
    work_done = False

    with tqdm(desc="Collecting IDs for processing", unit="id") as pbar:
        # We now get IDs in batches to avoid contention on the DB
        while (id := get_next_available_id(feature_column_name, pending_timeout_minutes=pending_timeout_minutes)) is not None:
            all_ids_to_process.append(id)
            pbar.update(1)
            work_done = True
            if len(all_ids_to_process) >= batch_size:
                print(f"Processing batch of {len(all_ids_to_process)} IDs...")
                run_parallel_feature_calculation(all_ids_to_process, feature_function, feature_column_name)
                all_ids_to_process = [] # Reset the batch

    # Process any remaining IDs that didn't fill a full batch
    if all_ids_to_process:
        print(f"Processing final batch of {len(all_ids_to_process)} IDs...")
        run_parallel_feature_calculation(all_ids_to_process, feature_function, feature_column_name)
    
    reclaim_pending_jobs(feature_column_name)
    if work_done:
        print("All available feature calculations are complete.")
    else:
        print("No new jobs were found for feature calculation. All jobs are either complete or pending.")


def run_parallel_feature_calculation(ids: Iterable[int], feature_function: Callable, feature_column_name: str, db_path: str = 'data/landscape-analysis.db'):
    """
    Runs the feature calculation for a list of IDs in parallel using a multiprocessing pool.
    """
    optimal_processes = get_optimal_process_count()
    print(f"Starting parallel feature calculation for {len(ids)} IDs using {optimal_processes} processes...")
    
    # Prepare the arguments for each worker process
    tasks = [(id, feature_function, feature_column_name) for id in ids]
    
    with Pool(processes=optimal_processes) as pool:
        # pool.imap is great for iterable data and showing progress with tqdm
        processed_ids = list(tqdm(pool.imap(_process_one_id_for_features, tasks), total=len(tasks), desc="Calculating Features"))
    
    print(f"Finished calculating features for {len(processed_ids)} IDs.")

def _process_one_id_for_features(args):
    """
    Helper function for the multiprocessing pool.
    Takes a tuple (id, feature_function, feature_column_name) and calls the main calculation function.
    """
    id, feature_function, feature_column_name = args
    calculate_and_save_features(id, feature_function)
    return id # Return the ID to track progress

# Mapping from feature names to (function, database column name)
FEATURE_MAP = {
    "information_content": (calculate_information_content, "ic_m0"),
    "nbc": (calculate_nbc, "nbc_nn_nb_mean_ratio"),
    "dispersion": (calculate_dispersion, "disp_costs_runtime"),
    # Add more features here as needed
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ELA feature calculation.")
    parser.add_argument(
        "--feature",
        choices=FEATURE_MAP.keys(),
        required=True,
        help="Which feature to calculate (choose from: %(choices)s)"
    )
    args = parser.parse_args()

    feature_function, feature_column_name = FEATURE_MAP[args.feature]

    run_main_feature_calculation(
        feature_column_name=feature_column_name,
        feature_function=feature_function
    )