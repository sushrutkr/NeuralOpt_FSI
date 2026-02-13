import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dataloader.dataload import dataGenerate

if __name__ == "__main__":
    with open("./input/config.json", "r") as f:
        config = json.load(f)

    params_data = config["params_data"]
    train_radius = config["train_radius"]

    print("Generating and saving dataset to disk (CPU only, no DDP)...")
    train_loader, val_loader = dataGenerate(
        train_radius,
        params_data['batch_size'],
        params_data['ntsteps'],
        params_data['val_split'],
        world_size=1,
        rank=0,
        loadData=False,
        cache_file=params_data["cache_loc"]
    )
    print("Dataset generated and saved to:", params_data["cache_loc"])