import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from models.unet import ResUNet


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    cfg = './config/config.yaml'
    config = OmegaConf.load(cfg)
    unet = ResUNet(**config.models.unet)
    
    print(unet.encoder)