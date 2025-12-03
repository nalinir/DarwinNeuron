import sqlite3
from math import ceil, log2
from src.LandscapeAnalysis import ParameterSampleConfig, add_samples, assign_samples_to_problem, generate_shd_problem
from src.LandscapeAnalysis.SHDFunctions import SHDConfig, generate_and_save_SHD
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
cur.execute("SELECT * FROM problems")

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

# # Generate samples for each dimension
# # WARNING: this processes all the problems. To select specific problems, need to implement a way to filter problems
for dim in set(row[column_names.index('dim')] for row in problems): # Access 'dim' by its index
    nb_samples = 2**(ceil(log2(50 * dim)))  # sobol's sample size must be a power of 2
    print(nb_samples)
    nb_versions = 19
    lower_bound = -2  # TODO: use the result from surrogate gradient descent
    upper_bound = 2
    sample_config = ParameterSampleConfig(dim, nb_samples, "sobol", lower_bound, upper_bound)
    add_samples(sample_config, nb_versions)

    # Match problems and samples in the database
    for problem_row in problems: # Iterate through full problem rows
        problem_id = problem_row[column_names.index('id')]
        problem_dim = problem_row[column_names.index('dim')]
        if problem_dim == dim:
            assign_samples_to_problem(problem_id, sample_config)