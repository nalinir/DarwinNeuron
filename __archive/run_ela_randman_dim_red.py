from pflacco.classical_ela_features import *
from src.LandscapeAnalysis.Pipeline import get_next_available_id, get_xy_by_id
from tqdm import tqdm
from scipy.stats import rankdata
from sklearn.decomposition import PCA
import sqlite3

def scale_x(x, y):
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    r = rankdata(y)
    nb_points = len(y)
    w = np.log(nb_points) - np.log(r)
    w /= np.sum(w)
    return x_centered * w.reshape(-1, 1)

# Tanabe 2021: https://github.com/ryojitanabe/ela_drframework/blob/main/feature_computation.py
def tanabe_dim_reduction(x, y, dim):
    if dim >= x.shape[1]:
        result = np.copy(x)
        print(f"Target dimension {dim} >= original dimension {x.shape[1]}. No reduction is performed.")
    else:
        x = scale_x(x, y)
        pca = PCA(n_components=dim, svd_solver='full')
        result = pca.fit_transform(x)
    return result

def calculate_and_save_features_dim_red(loss_surface_id: int, sample_to_feature: Callable, dim_reduced = None, loss_dir="data/losses", sample_dir="data/samples", db_path="data/landscape-analysis.db"):
    x, y = get_xy_by_id(loss_surface_id, loss_dir, sample_dir, db_path)
    
    if dim_reduced is not None:
        print(f"Applying dimensionality reduction to {len(x)} samples with target dimension {dim_reduced}, current dimension {x.shape[1]}")
        x = tanabe_dim_reduction(x, y, dim_reduced)
    
    # calculate features
    print(f"Calculating features for loss_surface_id {loss_surface_id} with {len(x)} samples")
    features = sample_to_feature(x, y)
    
    # Replace all "." in features' keys with "_"
    features = {key.replace('.', '_'): value for key, value in features.items()}
    
    # Prefix feature keys with dimension reduction information
    if dim_reduced is not None:
        features = {f"r{dim_reduced}_{key}": value for key, value in features.items()}
    
    # Save the features to loss-surfaces.csv
    # Connect to the SQLite database
    con = sqlite3.connect(db_path, timeout=30)
    cur = con.cursor()
    
    # aquire write lock
    cur.execute("BEGIN EXCLUSIVE")

    # Check and add columns if they don't exist
    cur.execute(f"PRAGMA table_info(randman_loss_surfaces)")
    columns = [info[1] for info in cur.fetchall()]
    # Add missing columns 
    for key in features.keys():
        if key not in columns:
            cur.execute(f"ALTER TABLE randman_loss_surfaces ADD COLUMN {key} REAL")
            print(f"Added column {key} to randman_loss_surfaces table by calculate_and_save_features()")

    # Update the database with the feature values
    cur.execute(
        f"UPDATE randman_loss_surfaces SET {', '.join(f'{key} = ?' for key in features.keys())} WHERE id = ?",
        (*features.values(), loss_surface_id)
    )

    # Commit changes and close the connection
    con.commit()
    con.close()

if __name__ == '__main__':
    dim = 10 
    # tqdm with unknown total
    with tqdm(desc="Processing IDs", unit="id") as pbar:
        while (id := get_next_available_id(f'r{dim}_ic_h_max', pending_timeout_minutes=60)) is not None:
            calculate_and_save_features_dim_red(id, calculate_information_content, dim_reduced=dim)
            pbar.update(1)
        