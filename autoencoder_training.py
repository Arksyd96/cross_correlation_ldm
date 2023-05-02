import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger
from omegaconf import OmegaConf

from modules.models.autoencoder.vector_quantized_autoencoder import VQAutoencoder
from modules.models.autoencoder.gaussian_autoencoder import GaussianAutoencoder
from modules.data_module import DataModule
from modules.loggers import ReconstructionImageLogger

def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    global_seed(42)
    torch.set_float32_matmul_precision('high')

    # loading config file
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)

    # logger
    logger = wandb_logger.WandbLogger(
        project='cross_correlation_ldm', 
        name='autoencoding'
    )

    # data module
    data_module = DataModule(
        **config.data,
        autoencoder=None,
        use_2d_slices=True, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8
    )

    # autoencoder
    if config.models.autoencoder.target == 'VQAutoencoder':
        autoencoder = VQAutoencoder(**config.models.autoencoder, **config.models.autoencoder.loss)
    elif config.models.autoencoder.target == 'GaussianAutoencoder':
        autoencoder = GaussianAutoencoder(**config.models.autoencoder, **config.models.autoencoder.loss)
    else:
        raise ValueError('Unknown autoencoder target')
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(
        **config.callbacks.checkpoint,
        filename='{}-{}'.format(config.models.autoencoder.target, '{epoch:02d}')
    )
    image_logger = ReconstructionImageLogger(n_samples=5, modalities=['FLAIR', 'T1CE'])

    # training
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=200,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, image_logger]
    )

    trainer.fit(model=autoencoder, datamodule=data_module)