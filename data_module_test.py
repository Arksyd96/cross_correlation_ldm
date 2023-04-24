import torch
import os
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from models.data_module import DataModule
from models.vector_quantized_autoencoder import VQAutoencoder

if __name__ == "__main__":
    

    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)
    autoencoder = VQAutoencoder(**config.models.autoencoder)
    dm = DataModule(
        **config.data, 
        autoencoder=autoencoder, 
        use_2d_slices=True,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    