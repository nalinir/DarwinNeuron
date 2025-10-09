# import sqlite3
# import pandas as pd
# from src.Models import calc_randmansnn_parameters
# from src.LandscapeAnalysis.SHDFunctions import SHDConfig
# from src.Utilities import match_config
# import sys

# from src.LandscapeAnalysis.NewModelFunctions import calc_parameters

# valid_models = ["SNN", "ANN_with_LIF_output", "Hybrid_RNN_SNN_rec", "Hybrid_RNN_SNN_V1_same_layer", "NSN_with_LIF_output", "Hybrid_NSN_SNN_rec", "Hybrid_NSN_SNN_V1_same_layer"]


# def generate_shd_problem(shd_config: SHDConfig, nb_hidden=128, nb_classes=20, loss_fn='cross_entropy', db_path='data/landscape-analysis.db', model_type='old'):
#     con = sqlite3.connect(db_path)
#     cur = con.cursor()

#     # --- Start of Changes ---

#     # 1. Remove the old 'shd_problems' table if it exists
#     cur.execute("DROP TABLE IF EXISTS shd_problems")
#     con.commit()

#     # 2. Create the new 'shd_problems' table with the corrected foreign key relationship
#     #    We'll assume your 'meta-data.csv' corresponds to a table named 'shd_configs'
#     #    in your database, and its 'id' column is what 'shd_id' refers to.

#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS shd_problems (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             shd_id INTEGER,
#             nb_hidden INTEGER,
#             loss_fn TEXT,
#             dim INTEGER,
#             FOREIGN KEY (shd_id) REFERENCES shd_configs(id)
#         )
#     """)
#     con.commit()

#     # --- End of Changes ---

#     # Find matching shd configuration from the CSV
#     shd_df = pd.read_csv("data/SHD/meta-data.csv")
#     shd_row = match_config(shd_df, shd_config)

#     # Select the first (and the only) matching row
#     if shd_row.empty:
#         con.close()
#         raise ValueError(f"No matching SHD configuration found for {shd_config}")
#     shd_row = shd_row.iloc[0]

#     # Calculate the dimension for the problem
#     if model_type == 'old':
#         dim = calc_randmansnn_parameters(shd_config.nb_inputs, nb_hidden, nb_classes)
#     elif model_type in valid_models:
#         dim = calc_parameters(shd_config.nb_inputs, nb_hidden, nb_classes, model_type)
#     else:
#         con.close()
#         raise ValueError(f"Invalid model type: {model_type}. Valid options are: {valid_models}")
#     # Check if the problem already exists in the database
#     # Note: shd_row["id"] already comes from the meta-data.csv, which is what we want to use as shd_id
#     cur.execute(f"""
#         SELECT id FROM shd_problems 
#         WHERE shd_id = {int(shd_row["id"])} AND nb_hidden = {nb_hidden} AND loss_fn = '{loss_fn}'
#     """)
#     existing_problem = cur.fetchone()

#     if existing_problem:
#         con.close()
#         raise ValueError(f"Problem with shd_id {int(shd_row['id'])} and nb_hidden {nb_hidden} already exists.")

#     # Insert the new problem into the database
#     cur.execute(f"""
#         INSERT INTO shd_problems (shd_id, nb_hidden, loss_fn, dim) 
#         VALUES ({int(shd_row["id"])}, {nb_hidden}, '{loss_fn}', {dim})
#     """)

#     con.commit()
#     con.close()