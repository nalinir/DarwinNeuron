from src.LandscapeAnalysis.Pipeline import get_next_available_id, reclaim_pending_jobs
from src.LandscapeAnalysis.NewModelFunctions import calculate_and_save_loss_new

def main():
    number_ids = 10
    while (id := get_next_available_id('loss_filename', pending_timeout_minutes=60)) is not None:
        calculate_and_save_loss_new(id)
        number_ids += 1
        print(f"Calculated and saved loss for ID: {id}, total ids: {number_ids}")
    reclaim_pending_jobs('loss_filename')
    if number_ids > 0:
        print("All available loss calculations are complete.")
    else:
        print("No new jobs were found for loss calculation. All jobs are either complete or pending.")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
