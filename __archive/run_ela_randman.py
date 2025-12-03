from pflacco.classical_ela_features import *
from src.LandscapeAnalysis import get_next_available_id, calculate_and_save_features
from tqdm import tqdm

# tqdm with unknown total
with tqdm(desc="Processing IDs", unit="id") as pbar:
    while (id := get_next_available_id('ic_h_max', pending_timeout_minutes=60)) is not None:
        calculate_and_save_features(id, calculate_information_content)
        pbar.update(1)
        