import os
import random
import torch
import numpy as np
from PIL import Image
from typing import List

# ------------------------------
# Seed for reproducibility
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Seed set to {seed}")

# ------------------------------
# Load images safely
# ------------------------------
def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} does not exist")
    return Image.open(image_path).convert("RGB")

# ------------------------------
# Save a dictionary as JSON
# ------------------------------
import json
def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------------
# Compute average loss
# ------------------------------
def average_loss(loss_list: List[float]) -> float:
    if not loss_list:
        return 0.0
    return sum(loss_list) / len(loss_list)

# ------------------------------
# Create checkpoint directory
# ------------------------------
def make_checkpoint_dir(base_dir: str, epoch: int):
    path = os.path.join(base_dir, f"epoch_{epoch}")
    os.makedirs(path, exist_ok=True)
    return path

