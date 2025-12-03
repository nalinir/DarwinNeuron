# from src.LandscapeAnalysis.SHDFunctions import SHDConfig
# from src.LandscapeAnalysis.Pipeline import LossSurfaceConfig
# from src.Models import RandmanSNN
# from torch.nn.functional import cross_entropy
# from src.LandscapeAnalysis.Utils import get_parameter_to_loss_fn
# import sqlite3
# from dataclasses import dataclass
# from src.LandscapeAnalysis.NewModelFunctions import configure_model_randman # Adjust connections at end

# import os
# import numpy as np
# import torch
# import uuid
# import sys

# sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")
# from maren_data.helpers import choose_data_params

# # TO DO: 
# # 1. Adjust code to create get_parameter_to_loss_fn for other models
# ## This does not train the model, it just evaluates

# from class_based_implementation.models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer
# from class_based_implementation.surr_grad import SurrGradSpike

# # Define your MODEL_CLASSES and LOSS_FUNCTIONS dictionaries here
# MODEL_CLASSES = {
#     "SNN": SNN,
#     "ANN_with_LIF_output": ANN_with_LIF_output,
#     "NSN_with_LIF_output": NSN_with_LIF_output,
#     "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
#     "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
#     "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
#     "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,
# }


# SHD_CLASSES = 20

# @dataclass
# class SHDProblemConfig:
#     shd_id: int
#     nb_hidden: int
#     loss_fn: str

#     @classmethod
#     def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
#         con = sqlite3.connect(db_path)
#         cur = con.cursor()
#         cur.execute(f"SELECT shd_id, nb_hidden, loss_fn FROM problems WHERE id = {id}")
#         row = cur.fetchone()
#         con.close()
#         return cls(*row) 
    
# def calculate_and_save_loss_shd(loss_surface_id: int, 
#                             sample_dir='data/samples',
#                             shd_dir='data/SHD',
#                             loss_dir='data/losses',
#                             db_path='data/landscape-analysis.db',
#                             device='cuda',
#                             num_workers=4,
#                             model_type='old' # This is going to be based on the problem, to revisit
#                             ):
#     loss_surface = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
#     problem = SHDProblemConfig.lookup_by_id(loss_surface.problem_id, db_path)
#     shd = SHDConfig.lookup_by_id(problem.shd_id, os.path.join(shd_dir, "meta-data.csv"))


#     # Prepare data and model
#     settings = {}
#     settings["max_time"] = shd.max_time
#     settings["noise"] = shd.noise
#     settings["time_step"] = shd.time_step
#     settings["nb_inputs"] = shd.nb_inputs
#     settings["nb_outputs"] = SHD_CLASSES
#     settings["device"] = device
#     settings["dtype"] = torch.float32
#     settings["batch_size"] = 256
#     settings["percent_data"] = 1.0  # Use all data

#     pre_path_data = f"/scratch/nar8991/snn/snn_ann_hybrid/maren_data/shd/{settings['nb_inputs']}_inputs/{settings['max_time']}_max_time/{settings['noise']}_noise"

#     train_loader, val_loader, test_loader = choose_data_params(
#         "shd", settings, num_workers=num_workers, pre_path=pre_path_data
#         )

#     if problem.loss_fn == 'cross_entropy':
#         loss_fn = cross_entropy
    
#     # Get loss function based on model type (no training)
#     if model_type == 'old':
#         model = RandmanSNN(shd.nb_inputs, problem.nb_hidden, SHD_CLASSES, learn_beta=False, beta=0.95)
#         f = get_parameter_to_loss_fn(train_loader, model, loss_fn, device)
#     elif model_type in MODEL_CLASSES:
#         model = configure_model(model_type)
#         f = get_parameter_to_loss_fn_other_models(train_loader, model, loss_fn, device)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

#     samples = loss_surface.read_sample(sample_dir, db_path)
    
#     # The computation part
#     print(f"calculating loss for loss_surface_id {loss_surface_id} with {len(samples)} samples")
#     loss = np.apply_along_axis(f, 1, samples)

#     # Save loss to the database
#     loss_filename = f"{uuid.uuid4().hex}.npy"
#     loss_surface.write_loss_filename(loss_filename, db_path)

#     # Save loss to file
#     os.makedirs(loss_dir, exist_ok=True)
#     np.save(os.path.join(loss_dir, loss_filename), loss)


