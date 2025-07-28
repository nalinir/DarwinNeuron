import sqlite3
import pandas as pd
from src.Models import calc_randmansnn_parameters


def generate_shd_problem(shd_config: SHDConfig, nb_hidden=256, nb_classes=20, loss_fn = 'cross_entropy', db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find matching shd configuration
    shd_df = pd.read_csv("data/shd/meta-data.csv")
    shd_row = match_config(shd_df, shd_config)

    # Select the first (and the only) matching row
    shd_row = shd_row.iloc[0]

    # Calculate the dimension for the problem - the randman name doesn't actually matter here
    dim = calc_randmansnn_parameters(shd_config.nb_units, nb_hidden, nb_classes)

    # Check if the problem already exists in the database
    cur.execute(f"""
        SELECT id FROM problems 
        WHERE shd_id = {shd_row["id"]} AND nb_hidden = {nb_hidden} AND loss_fn = '{loss_fn}'
    """)
    existing_problem = cur.fetchone()

    if existing_problem:
        con.close()
        raise ValueError(f"Problem with shd_id {shd_row['id']} and nb_hidden {nb_hidden} already exists.")

    # Insert the new problem into the database
    cur.execute(f"""
        INSERT INTO problems (shd_id, nb_hidden, loss_fn, dim) 
        VALUES ({int(shd_row["id"])}, {nb_hidden}, '{loss_fn}', {dim})
    """)

    con.commit()
    con.close()