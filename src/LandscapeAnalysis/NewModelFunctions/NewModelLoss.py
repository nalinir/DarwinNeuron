from src.RandmanFunctions import RandmanConfig, split_and_load
from src.LandscapeAnalysis.Pipeline import LossSurfaceConfig
# from src.Models import RandmanSNN
from torch.nn.functional import cross_entropy
from src.LandscapeAnalysis.Utils import get_parameter_to_loss_fn, get_parameter_to_loss_fn_other_models, get_parameter_to_loss_fn_batched_other_models, get_parameter_to_loss_fn_batched_snn
import sqlite3
from dataclasses import dataclass
from src.LandscapeAnalysis.NewModelFunctions import configure_model_randman # Adjust connections at end



import os
import numpy as np
import torch
import uuid
import sys

MODEL_CLASSES = {
    "SNN",
    "ANN_with_LIF_output",
    "NSN_with_LIF_output",
    "Hybrid_RNN_SNN_rec",
    "Hybrid_NSN_SNN_rec",
    "Hybrid_RNN_SNN_V1_same_layer",
    "Hybrid_NSN_SNN_V1_same_layer",
}

@dataclass
class RandmanProblemConfigNew:
    randman_id: int
    nb_hidden: int
    loss_fn: str
    model_type: str
    recurrent: bool

    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT randman_id, nb_hidden, loss_fn, model_type, recurrent FROM randman_problems WHERE id = {id}")
        row = cur.fetchone()
        con.close()
        return cls(*row) 


def calculate_and_save_loss_new(loss_surface_id: int, 
                            sample_dir='data/samples',
                            randman_dir='data/randman',
                            loss_dir='data/losses',
                            db_path='data/landscape-analysis.db',
                            device='cuda'
                            ):
    loss_surface = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
    problem = RandmanProblemConfigNew.lookup_by_id(loss_surface.problem_id, db_path)
    randman = RandmanConfig.lookup_by_id(problem.randman_id, os.path.join(randman_dir, "meta-data.csv"))
    print(f"Calculating loss for loss_surface_id {loss_surface_id}, model_type {problem.model_type}, randman_id {problem.randman_id}")

    # Prepare data and model
    # train_loader, _ = split_and_load(randman.read_dataset(randman_dir), batch_size=516)
    if problem.loss_fn == 'cross_entropy':
        loss_fn = cross_entropy

    samples = loss_surface.read_sample(sample_dir, db_path)
    
    # Store configuration data to pass to workers
    model_config = {
        'model_type': problem.model_type,
        'nb_hidden': problem.nb_hidden,
        'nb_units': randman.nb_units,
        'nb_classes': randman.nb_classes,
        'recurrent': problem.recurrent
    }

    data_loader_config = {
        'randman_id': 0,
        'randman_dir': randman_dir
    }
    randman = RandmanConfig.lookup_by_id(data_loader_config['randman_id'], os.path.join(data_loader_config['randman_dir'], "meta-data.csv"))

    if problem.model_type == 'old':
        f = get_parameter_to_loss_fn_batched_snn(loss_fn, device, model_config, data_loader_config)
        loss = f(samples)

    elif problem.model_type in MODEL_CLASSES:
        f = get_parameter_to_loss_fn_batched_other_models(device, model_config, data_loader_config)
        loss = f(samples)
    else:
        raise ValueError(f"Unknown model type: {problem.model_type}")

    print(f"calculating loss for loss_surface_id {loss_surface_id} with {len(samples)} samples in a parallel batch operation")
    
    os.makedirs(loss_dir, exist_ok=True)
    loss_filename = f"{uuid.uuid4().hex}.npy"
    np.save(os.path.join(loss_dir, loss_filename), loss)
    loss_surface.write_loss_filename(loss_filename, db_path)
