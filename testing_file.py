import torch
import os
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from models.data_module import DataModule
from models.vector_quantized_autoencoder import VQAutoencoder

from models.unet_3d import ResUNet3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    

    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)

    unet = ResUNet3D(**config.models.unet).to(device)
    print(unet.hparams)

    x = torch.randn(1, 2, 64, 32, 32).to(device)
    time = torch.randint(0, config.models.unet.T, (1,), dtype=torch.long, device=device)

    print(unet(x, time).shape)

    