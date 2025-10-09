
# from src.Utilities import match_config, next_id
# from dataclasses import dataclass
# import os
# import pandas as pd
# import torch
# from dataclasses import asdict

# # Eventually - could make these functions generalized between the 2
# @dataclass
# class SHDConfig:
#     nb_inputs: int = 700
#     # pct_data: float = 1.0
#     max_time: float = 1.0
#     noise: bool = False
#     time_step: float = 0.002

#     @classmethod
#     def lookup_by_id(cls, id: int, table_path: str = 'data/SHD/meta-data.csv'):
#         """
#         Lookup a row by id in a CSV file.
#         Args:
#             id (int): The ID of the configuration to look up.
#             table_path (str): Path to the CSV file containing SHD configurations.
#         Returns:
#             SHDConfig: An instance of SHDConfig with parameters from the specified row.
#         """
#         df = pd.read_csv(table_path, index_col="id")
#         if id not in df.index:
#             raise ValueError(f"ID {id} not found in {table_path}.")
#         kwargs = df.loc[id].to_dict()
#         kwargs.pop("filename")
#         return cls(**kwargs)
    
#     def read_dataset(self, save_dir="data/SHD"):
#         meta_path = os.path.join(save_dir, "meta-data.csv")
#         if not os.path.isfile(meta_path):
#             raise FileNotFoundError(f"Meta-data file not found at {meta_path}")

#         df = pd.read_csv(meta_path)
#         match = match_config(df, self)
#         if match.empty:
#             raise ValueError("No dataset found with the specified parameters.")

#         filename = match.iloc[0]["filename"]
#         filepath = os.path.join(save_dir, filename)
#         if not os.path.isfile(filepath):
#             raise FileNotFoundError(f"Dataset file not found at {filepath}")

#         data = torch.load(filepath, weights_only=False)
#         return data

# def get_SHD_filepath(config: SHDConfig):
#     """
#     Generate a spiking neural network dataset using SHD configuration.
#     Args:
#         config (SHDConfig): Configuration object containing parameters for
#             dataset generation including nb_steps, nb_units, and other settings.
#             The 'id' and 'filename' fields are excluded from dataset generation.
#     Returns:
#         TensorDataset: A PyTorch dataset containing spike train tensors and
#             corresponding label tensors, ready for use in neural network training.
#     Note:
#         - The function generates exactly 1 spike per unit (nb_spikes=1)
#     """
    
#     # There is a train, test, and validation loader in the SHD dataset.
#     # We will only generate the training set.

#     file_path = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/{config.nb_inputs}_inputs/{config.max_time}_max_time/{config.noise}_noise/cache_{config.time_step}"

#     return file_path

# def generate_and_save_SHD(config: SHDConfig, save_dir="data/SHD"):    
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
#     meta_path = os.path.join(save_dir, "meta-data.csv")
    
#     # convert to dict
#     config_dict = asdict(config)
    
#     # Check for existing parameter combination
#     if os.path.isfile(meta_path):
#         df = pd.read_csv(meta_path, index_col='id')
#         match = match_config(df, config)
#         if not match.empty:
#             raise ValueError("Dataset already exists with the same parameters.")

#     # Generate dataset
#     file_path = get_SHD_filepath(config)
    
#     # Generate random filename
#     config_dict['filename'] = file_path
    
#     # Log parameters to meta-data.csv 
#     if os.path.isfile(meta_path):
#         df = pd.read_csv(meta_path, index_col='id')
#         row_df = pd.DataFrame([config_dict], index=[next_id(df)])
#         df = pd.concat([df, row_df], ignore_index=False)
#     else:
#         # Start with ID 0 for the first entry
#         df = pd.DataFrame([config_dict], index=[0])

#     df.to_csv(meta_path, index=True, index_label='id')
#     return df