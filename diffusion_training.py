import numpy as np
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from tqdm import tqdm
import sys

from models.unet import ResUNet
from models.vector_quantized_autoencoder import VQAutoencoder
from models.gaussian_autoencoder import GaussianAutoencoder
from models.data_module import DataModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class ImageLogger(pl.Callback):
    def __init__(self,
        embed_dim,
        latent_resolution,
        n_slices,
        autoencoder,
        **kwargs
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_resolution = latent_resolution
        self.n_slices = n_slices
        self.autoencoder = autoencoder.to(device)

    def on_train_epoch_end(self, trainer, pl_module):
        # sample images
        pl_module.eval()
        with torch.no_grad():
            latents = pl_module.diffusion.sample(
                pl_module,
                torch.randn(
                    1,
                    pl_module.hparams.in_channels, 
                    pl_module.hparams.resolution, 
                    pl_module.hparams.resolution
                ).to(pl_module.device)
            ).reshape(
                self.n_slices, 
                self.embed_dim, 
                self.latent_resolution, 
                self.latent_resolution
            ) # => will be of shape (64, 2, 32, 32)

            # selecting sequence of 10 slices with corresponding positions
            latents = latents[::int(self.n_slices/10), ...] # => will be of shape (10, 2, 32, 32)
            position = torch.arange(0, self.n_slices, int(self.n_slices/10)).to(pl_module.device)

            # decoding the latent spaces
            if isinstance(self.autoencoder, GaussianAutoencoder):
                pemb = self.autoencoder.encode_position(position)
                generated = self.autoencoder.decode(latents, pemb)
                generated = torch.tanh(generated)
            elif isinstance(self.autoencoder, VQAutoencoder):
                pemb = self.autoencoder.encode_position(position)
                generated = self.autoencoder.decode_pre_quantization(latents, pemb)
            else:
                raise NotImplementedError('Unknown autoencoder type')

            img = torch.cat([
                torch.hstack([img for img in generated[:, c, ...]]) for c in range(generated.shape[1])
            ], dim=0)

            wandb.log({
                'Generation examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='Generation examples'
                )
            })
    
if __name__ == "__main__":
    global_seed(42)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_float32_matmul_precision('high')
    
    # loading config file
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)
    
    wandb_logger = wandb.WandbLogger(
        project='cross_correlation_ldm', 
        name='diffusion'
    )

    # autoencoder
    try:
        autoencoder = getattr(sys.modules[__name__], config.models.autoencoder.target)
    except:
        raise AttributeError('Unknown autoencoder target')
    
    # load autoencoder weights/hyperparameters
    autoencoder = autoencoder.load_from_checkpoint('./checkpoints/GaussianAutoencoder-epoch=169.ckpt') # latest weights
    
    # data module
    data_module = DataModule(
        **config.data,
        autoencoder=autoencoder,
        use_latents=True, 
        batch_size=8, 
        shuffle=True, 
        num_workers=8
    )

    checkpoint_callback = ModelCheckpoint(
        **config.callbacks.checkpoint,
        filename='unet-{epoch:02d}'
    )
    image_logger = ImageLogger(n_samples=5, modalities=['FLAIR', 'T1CE'])

    unet = ResUNet(**config.models.unet)
    
    # trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=1000,
        log_every_n_steps=1,
        enable_progress_bar=True,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, image_logger]
    )
    trainer.fit(unet, data_module)

    