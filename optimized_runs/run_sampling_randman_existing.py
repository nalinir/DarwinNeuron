import sqlite3
from math import ceil, log2
from src.LandscapeAnalysis import ParameterSampleConfig
from src.LandscapeAnalysis.NewModelFunctions.OptimalSampleConstruction import add_samples_existing
# from src.LandscapeAnalysis.SHDFunctions import SHDConfig, generate_and_save_SHD
# data_configs = [{'nb_inputs': 70, 'max_time': 1.0, 'noise': False, 'time_step': 0.005},]


# We can eventually let this be reinstated each time (since we don't do data runs anymore)
# for data_config in data_configs:
#     generate_and_save_SHD(SHDConfig(**data_config))

# for data_config in data_configs:
#     generate_shd_problem(SHDConfig(**data_config), nb_hidden=128, nb_classes=20, loss_fn='cross_entropy', db_path='data/landscape-analysis.db')


# Connect to the SQLite database
db_path = "data/landscape-analysis.db"
con = sqlite3.connect(db_path)
cur = con.cursor()

# Query the "problems" table
cur.execute("SELECT * FROM randman_problems where dim = 500 and model_path is not NULL")

# Get column names from cursor description
column_names = [description[0] for description in cur.description]
print("Column Names:", column_names)

problems = cur.fetchall()
con.close()
del cur, con, db_path

for row in problems:
    print(row)

print("\nAll Problems (list of tuples):")
print(problems)

for problem_row in problems:
    dim = problem_row[column_names.index('dim')]
    model_path = problem_row[column_names.index('model_path')]
    problem_id = problem_row[column_names.index('id')]
    nb_samples = 2**(ceil(log2(50 * dim)))  # sobol's sample size must be a power of 2
    print(f"dim: {dim}, model_path: {model_path}, nb_samples: {nb_samples}")
    sample_config = ParameterSampleConfig(dim, nb_samples, "sobol", None, None)
    # Only 1 version
    add_samples_existing(model_path, problem_id, sample_config)
