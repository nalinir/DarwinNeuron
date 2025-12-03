import sqlite3
import pandas as pd
from src.Models import calc_randmansnn_parameters
from src.RandmanFunctions import RandmanConfig
from src.Utilities import match_config
import sys
from src.LandscapeAnalysis.NewModelFunctions.ELAModelConstruction import configure_model_randman
from torch.nn.utils import parameters_to_vector


valid_models = [
    "SNN",
    "ANN_with_LIF_output",
    "NSN_with_LIF_output",
    "Hybrid_RNN_SNN_rec",
    "Hybrid_NSN_SNN_rec",
    "Hybrid_RNN_SNN_V1_same_layer",
    "Hybrid_NSN_SNN_V1_same_layer"
]

def generate_randman_problem_new(randman_config: RandmanConfig, nb_hidden=40, loss_fn='cross_entropy', db_path='data/landscape-analysis.db', model_type='old', recurrent=False, percent_snn=None):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find matching randman configuration from the CSV
    randman_df = pd.read_csv("data/randman/meta-data.csv")
    randman_row = match_config(randman_df, randman_config)

    cur.execute("PRAGMA table_info(randman_problems)")
    columns = [row[1] for row in cur.fetchall()]
    if "percent_snn" not in columns:
        cur.execute("ALTER TABLE randman_problems ADD COLUMN percent_snn REAL")

    # Select the first (and the only) matching row
    if randman_row.empty:
        con.close()
        raise ValueError(f"No matching Randman configuration found for {randman_config}")
    randman_row = randman_row.iloc[0]

    # Calculate the dimension for the problem
    if model_type == 'old':
        dim = calc_randmansnn_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes)
    elif model_type in valid_models:
        dim = calc_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes, model_type, recurrent, percent_snn)
    else:
        con.close()
        raise ValueError(f"Invalid model type: {model_type}. Valid options are: {valid_models}")
    # Check if the problem already exists in the database
    # Note: randman_row["id"] already comes from the meta-data.csv, which is what we want to use as randman_id
    cur.execute(f"""
        SELECT id FROM randman_problems 
        WHERE randman_id = {int(randman_row["id"])} AND nb_hidden = {nb_hidden} AND loss_fn = '{loss_fn}' AND model_type = '{model_type}' AND dim = {dim} AND recurrent = {recurrent} and percent_snn = {percent_snn}
    """)
    existing_problem = cur.fetchone()

    if existing_problem:
        con.close()
        raise ValueError(f"Problem with randman_id {int(randman_row['id'])}, model_type {model_type}, and recurrent {recurrent} already exists.")

    # Ensure the percent_snn column exists in the randman_problems table
    # Insert the new problem into the database
    cur.execute(f"""
        INSERT INTO randman_problems (randman_id, nb_hidden, loss_fn, dim, model_type, recurrent, percent_snn) 
        VALUES ({int(randman_row["id"])}, {nb_hidden}, '{loss_fn}', {dim}, '{model_type}', {recurrent}, {percent_snn})
    """)

    con.commit()
    con.close()


def calc_parameters(nb_inputs: int, nb_hidden: int, nb_outputs: int, model_type = "SNN", recurrent=True, percent_snn=None) -> int:
    """
    Calculates the total number of parameters for the BaseTemporalModel.

    Args:
        nb_inputs: Number of input features.
        nb_hidden: Number of hidden features.
        nb_outputs: Number of output features (classes).

    Returns:
        The total number of parameters in the model.
    """
    
    model = configure_model_randman(model_type=model_type, nb_hidden=nb_hidden, nb_units=nb_inputs, nb_classes=nb_outputs, recurrent=recurrent, percent_snn=percent_snn)

    # Convert model parameters to a vector
    vector = parameters_to_vector(model.parameters())
    
    return vector.size(0)

def create_seeded_rows(randman_config: RandmanConfig, nb_hidden=40, loss_fn='cross_entropy', db_path='data/landscape-analysis.db', model_type='old', recurrent=False, percent_snn=None):
    # Create a column for seed and related model path, but everything else stays the same
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    pct_data = 1.0
    cur.execute("PRAGMA table_info(randman_problems)")
    columns = [row[1] for row in cur.fetchall()]
    if "seed" not in columns:
        cur.execute("ALTER TABLE randman_problems ADD COLUMN seed INTEGER")
    if "model_path" not in columns:
        cur.execute("ALTER TABLE randman_problems ADD COLUMN model_path TEXT")

    dim = calc_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes, model_type, recurrent, percent_snn)

    base_path = "/scratch/nar8991/snn/snn_ann_hybrid/optuna_results/randman/1_d/2_classes/3/cross_entropy"
    randman_df = pd.read_csv("data/randman/meta-data.csv")
    randman_row = match_config(randman_df, randman_config)
    if randman_row.empty:
        con.close()
        raise ValueError(f"No matching Randman configuration found for {randman_config}")
    randman_row = randman_row.iloc[0]

    # Create rows for each seed
    for seed in range(0, 5):
        if not percent_snn:
            model_path = f"{base_path}/{model_type}/recurrent_{recurrent}/seed_{seed}/grid_sampler/{nb_hidden}_hidden/{pct_data}_pct_data/best_model_of_study.ckpt"
        else:
            model_path = f"{base_path}/{model_type}/recurrent_{recurrent}/seed_{seed}/grid_sampler/{nb_hidden}_hidden/{pct_data}_pct_data/{percent_snn}_percent_snn/best_model_of_study.ckpt"
        print(f"Creating row for seed {seed}: {int(randman_row['id'])}, {nb_hidden}, '{loss_fn}', {dim}, '{model_type}', {recurrent}, {percent_snn}, {seed}, '{model_path}'")
        cur.execute("""
            INSERT INTO randman_problems (randman_id, nb_hidden, loss_fn, dim, model_type, recurrent, percent_snn, seed, model_path) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(randman_row["id"]), 
            nb_hidden, 
            loss_fn, 
            dim, 
            model_type, 
            recurrent, 
            percent_snn,  # This will properly handle None as NULL
            seed, 
            model_path
        ))

    con.commit()
    con.close()
