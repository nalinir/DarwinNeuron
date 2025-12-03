import torch
from src.LandscapeAnalysis import get_next_available_id, calculate_and_save_loss_shd
# from tqdm.auto import tqdm

total_ids = 19

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

while (id := get_next_available_id('loss_filename')) is not None:
    calculate_and_save_loss_shd(id, device=device, num_workers=4)
