import os
import numpy as np
import torch
import uuid
import sys
import json
import torch.nn as nn

sys.path.append("/scratch/nar8991/snn/snn_ann_hybrid")

from class_based_implementation.models import SNN, ANN_with_LIF_output, Hybrid_RNN_SNN_rec, Hybrid_RNN_SNN_V1_same_layer, NSN_with_LIF_output, Hybrid_NSN_SNN_rec, Hybrid_NSN_SNN_V1_same_layer
from class_based_implementation.surr_grad import SurrGradSpike

# Define your MODEL_CLASSES and LOSS_FUNCTIONS dictionaries here
MODEL_CLASSES = {
    "SNN": SNN,
    "ANN_with_LIF_output": ANN_with_LIF_output,
    "NSN_with_LIF_output": NSN_with_LIF_output,
    "Hybrid_RNN_SNN_rec": Hybrid_RNN_SNN_rec,
    "Hybrid_NSN_SNN_rec": Hybrid_NSN_SNN_rec,
    "Hybrid_RNN_SNN_V1_same_layer": Hybrid_RNN_SNN_V1_same_layer,
    "Hybrid_NSN_SNN_V1_same_layer": Hybrid_NSN_SNN_V1_same_layer,
}

# We will seed this eventually, currently it's done through optuna

# Should be generalized eventually for both (and further modularized)
def configure_model_randman(model_type: str, nb_hidden: int, nb_units: int, nb_classes: int, recurrent: bool, percent_snn: float = None):
    """
    Configure model based on model type and return results
    """
    config_path = "/scratch/nar8991/snn/snn_ann_hybrid/data_construction/randman_config_simplified.json"
    # Load the configuration for the model
    with open(config_path, 'r') as f:
        data_config = json.load(f)
    if model_type != "ANN_with_LIF_output" and model_type != "NSN_with_LIF_output":
        # Zenke regularization specific hyperparameters
        l2_lower = 0
        v2_lower = 0
        l1_upper = 1
        v1_upper = 100
        l2_upper = 0
        v2_upper = 0
        zenke_config = {
            "l2_lower": l2_lower,
            "v2_lower": v2_lower,
            "l1_upper": l1_upper,
            "v1_upper": v1_upper,
            "l2_upper": l2_upper,
            "v2_upper": v2_upper,
        }
        spike_grad_scale = 10
        spike_fn = SurrGradSpike.apply
    else:
        zenke_config = None
        spike_grad_scale = None
        spike_fn = None 
    
    learning_rate = 0.05
    
    model_args = {
        "input_features": nb_units,
        "hidden_features": nb_hidden,
        "output_features": nb_classes,
        "data_config": data_config,
        "recurrent": recurrent,
        "learning_rate": learning_rate,
        "loss_fn": nn.CrossEntropyLoss(),
        "zenke_config": zenke_config,
        "optimizer_name": "Adam",
        "spike_grad_scale": spike_grad_scale,
        "model_type": model_type,
        "spike_fn": spike_fn,
        "percent_snn": percent_snn
    }

    model = MODEL_CLASSES[model_type](**model_args)
    return model
