import numpy as np
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import wandb as wandb_logger
from nibabel.processing import resample_to_output
from nibabel import load
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from models.vector_quantized_autoencoder import VQAutoencoder
from models.data_module import DataModule

def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class SampleImageCallback(pl.Callback):
    def __init__(self,
        modalities=['t1', 't1ce', 't2', 'flair'], 
        n_samples=5,
        **kwargs
        ):
        super().__init__()
        self.n_samples = n_samples
        self.modalities = modalities

    def on_train_epoch_end(self, trainer, pl_module):
        # sample images
        pl_module.eval()
        with torch.no_grad():
            x, pos = next(iter(trainer.train_dataloader))
            x, pos = x.to(pl_module.device, torch.float32), pos.to(pl_module.device, torch.long)

            x, pos = x[:self.n_samples], pos[:self.n_samples]
            x_hat, _, _ = pl_module(x, pos)

            for idx, m in enumerate(self.modalities):
                img = torch.cat([
                    torch.hstack([img for img in x[:, idx, ...]]),
                    torch.hstack([img for img in x_hat[:, idx, ...]]),
                ], dim=0)
                wandb.log({
                    'Reconstruction examples': wandb.Image(
                        img.detach().cpu().numpy(), 
                        caption='{} - {} (Top are originals)'.format(m, trainer.current_epoch)
                    )
                })

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

    data_module = DataModule(
        **config.data,
        autoencoder=None,
        use_2d_slices=True, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8
    )

    autoencoder = VQAutoencoder(**config.models.autoencoder, **config.models.autoencoder.loss)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=None,
        dirpath='./checkpoints',
        filename='autoencoder-{epoch:02d}',
        save_top_k=1,
        mode='min',
        every_n_epochs=10
    )

    image_callback = SampleImageCallback(n_samples=5, modalities=['FLAIR', 'T1CE'])



    # training
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=200,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, image_callback]
    )

    trainer.fit(model=autoencoder, datamodule=data_module)