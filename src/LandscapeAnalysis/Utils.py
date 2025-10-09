import torch
from copy import deepcopy
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader
from typing import Callable
from src.Utilities import evaluate_snn
from tqdm import tqdm
import numpy as np
from torch.multiprocessing import Pool, Manager # Use torch's multiprocessing
from functools import partial
import os
from src.LandscapeAnalysis.NewModelFunctions.ELAModelConstruction import configure_model_randman # Adjust connections at end
from src.RandmanFunctions import RandmanConfig, split_and_load
from src.Models import RandmanSNN
from torch.nn.functional import cross_entropy


def get_parameter_to_loss_fn(loader, model, loss, device="cuda"):
    """
    Returns a function that computes the loss for a given parameter vector on the provided model and data loader.

    Args:
        loader: DataLoader for validation or evaluation.
        model: PyTorch model whose parameters will be set.
        loss: Loss function to evaluate (e.g., cross_entropy).
        device (str): Device to run computations on (default: "cuda").

    Returns:
        function: A function that takes a parameter vector and returns the computed loss.
    """
    model = deepcopy(model)

    def parameter_to_loss_fn(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
        vector_to_parameters(vector, model.parameters())
        # TO DO: Adjust this for other models based on model type?
        stats = evaluate_snn(model, loader, loss, device)
        # All it should return is the loss, don't worry about other metrics
        return stats.loss.item()

    return parameter_to_loss_fn

def get_parameter_to_loss_fn_other_models(loader, model, device="cuda"):
    """
    Returns a function that computes the loss for a given parameter vector on the provided model and data loader.

    Args:
        loader: DataLoader for validation or evaluation.
        model: PyTorch model whose parameters will be set.
        device (str): Device to run computations on (default: "cuda").

    Returns:
        function: A function that takes a parameter vector and returns the computed loss.
    """
    model = deepcopy(model)

    def parameter_to_loss_fn(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
        vector_to_parameters(vector, model.parameters())
        avg_loss = get_loss_from_other_model(model, loader, device)
        return avg_loss

    return parameter_to_loss_fn    


def get_loss_from_other_model(model, dataloader, device):
    """
    Calculates the average loss for a model over a dataset.

    Args:
        model (BaseTemporalModel): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input data and labels.
        device (torch.device): Device on which to perform computation (e.g., 'cpu' or 'cuda').

    Returns:
        tuple[float, float]: A tuple containing the average loss and the accuracy.
    """
    model.to(device)

    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    
    # Reset the test_accuracy metric before evaluation
    model.test_accuracy.reset() 

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).long() # <--- Crucial change here!

            # Call _common_step with 'test' to update the test_accuracy metric
            loss = model._common_step((x, y), "test")
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = model.test_accuracy.compute().item() * 100 # Get the final accuracy and convert to percentage

    print(f"Average Loss: {avg_loss:>8f}, Accuracy: {accuracy:>0.1f}%")
    return avg_loss



def _init_worker(model_config, data_loader_config, loss_fn_name, device_name):
    """
    Initializer function for the multiprocessing pool.
    Each worker process calls this to create its own local model instance.
    """
    global _GLOBAL_MODEL, _GLOBAL_DATA_LOADER, _GLOBAL_LOSS_FN, _GLOBAL_DEVICE
    
    # Force CUDA initialization in worker process
    if device_name == "cuda":
        try:
            torch.cuda.init()
            if not torch.cuda.is_available():
                print(f"Worker {os.getpid()}: CUDA not available, falling back to CPU")
                device_name = "cpu"
            else:
                print(f"Worker {os.getpid()}: CUDA initialized successfully")
        except Exception as e:
            print(f"Worker {os.getpid()}: CUDA initialization failed: {e}, using CPU")
            device_name = "cpu"
    
    _GLOBAL_DEVICE = torch.device(device_name)

    # Recreate the model based on the configuration
    if model_config['model_type'] == 'old':
        _GLOBAL_MODEL = RandmanSNN(model_config['nb_units'], model_config['nb_hidden'], model_config['nb_classes'], learn_beta=False, beta=0.95)
    else:
        _GLOBAL_MODEL = configure_model_randman(
            model_type=model_config['model_type'], 
            nb_hidden=model_config['nb_hidden'], 
            nb_units=model_config['nb_units'], 
            nb_classes=model_config['nb_classes'], 
            recurrent=model_config['recurrent']
        )
    
    randman = RandmanConfig.lookup_by_id(data_loader_config['randman_id'], os.path.join(data_loader_config['randman_dir'], "meta-data.csv"))

    # Recreate the data loader
    _GLOBAL_DATA_LOADER, _ = split_and_load(randman.read_dataset(data_loader_config['randman_dir']), batch_size=516)

    # Set the loss function and device
    if loss_fn_name == 'cross_entropy':
        _GLOBAL_LOSS_FN = cross_entropy
    _GLOBAL_DEVICE = device_name
    _GLOBAL_MODEL.to(_GLOBAL_DEVICE)

    # Set model to evaluation mode
    _GLOBAL_MODEL.eval()

    try:
        if hasattr(torch, 'compile') and device_name == "cuda":
            _GLOBAL_MODEL = torch.compile(_GLOBAL_MODEL, mode="reduce-overhead")
            print(f"Worker {os.getpid()}: Model compiled successfully")
    except Exception as e:
        print(f"Worker {os.getpid()}: Model compilation failed: {e}, continuing without compilation")


def _calculate_snn_loss_for_one_vector(param_vector_np):
    """Helper function to calculate loss for a single parameter vector for SNN."""
    # Access the model and data_loader from the global scope of the worker
    param_vector = torch.from_numpy(param_vector_np).float().to(_GLOBAL_DEVICE)
    vector_to_parameters(param_vector, _GLOBAL_MODEL.parameters())
    stats = evaluate_snn(_GLOBAL_MODEL, _GLOBAL_DATA_LOADER, _GLOBAL_LOSS_FN, _GLOBAL_DEVICE)
    return stats.loss.item()

def _calculate_other_model_loss_for_one_vector(param_vector_np):
    """Helper function to calculate loss for a single parameter vector for other models."""
    # Access the model and data_loader from the global scope of the worker
    param_vector = torch.from_numpy(param_vector_np).float().to(_GLOBAL_DEVICE)
    vector_to_parameters(param_vector, _GLOBAL_MODEL.parameters())

    _GLOBAL_MODEL.eval()
    total_loss = 0
    _GLOBAL_MODEL.test_accuracy.reset()
    
    with torch.no_grad():
        for x, y in _GLOBAL_DATA_LOADER:
            x = x.to(_GLOBAL_DEVICE)
            y = y.to(_GLOBAL_DEVICE).long()
            loss = _GLOBAL_MODEL._common_step((x, y), "test")
            total_loss += loss.item()

    avg_loss = total_loss / len(_GLOBAL_DATA_LOADER)
    return avg_loss

# =====================================================================================
# PARALLELIZED BATCHED FUNCTION IMPLEMENTATIONS
# =====================================================================================
def get_optimal_gpu_process_count():
    """Get optimal process count for GPU workloads"""
    # For GPU work, use very few processes
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        # Use half of allocated CPUs for GPU work, capped at 3
        return min(int(slurm_cpus) - 1, 5)  # 5 workers with 6 CPUs
    else:
        return 4  # Safe default


def get_parameter_to_loss_fn_batched_snn(loss_fn: Callable, device: str, model_config: dict, data_loader_config: dict) -> Callable:
    """
    Returns a function that computes loss for a batch of parameter vectors for SNN models,
    using multiprocessing for parallelization.
    """
    def batched_loss_fn(parameters_batch: np.ndarray) -> np.ndarray:
        # Pass the configurations to the Pool's initializer, which avoids pickling the model itself.
        max_processes = min(get_optimal_gpu_process_count(), len(parameters_batch))
        print(f"DEBUG: Creating pool with {max_processes} workers for batch size {len(parameters_batch)}")

        with Pool(processes=max_processes, initializer=_init_worker, initargs=(model_config, data_loader_config, loss_fn.__name__, device)) as pool:
            losses = list(tqdm(pool.imap(_calculate_snn_loss_for_one_vector, parameters_batch), total=len(parameters_batch), desc="Calculating SNN loss"))
        
        return np.array(losses)
    
    return batched_loss_fn


def get_parameter_to_loss_fn_batched_other_models(device: str, model_config: dict, data_loader_config: dict) -> Callable:
    """
    Returns a function that computes loss for a batch of parameter vectors for other models,
    using multiprocessing for parallelization.
    """
    def batched_loss_fn(parameters_batch: np.ndarray) -> np.ndarray:
        max_processes = min(get_optimal_gpu_process_count(), len(parameters_batch))
        print(f"DEBUG: Creating pool with {max_processes} workers for batch size {len(parameters_batch)}")
        with Pool(processes=max_processes, initializer=_init_worker, initargs=(model_config, data_loader_config, 'cross_entropy', device)) as pool:
            losses = list(tqdm(pool.imap(_calculate_other_model_loss_for_one_vector, parameters_batch), total=len(parameters_batch), desc="Calculating other model loss"))
        
        return np.array(losses)

    return batched_loss_fn