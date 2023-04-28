import numpy as np
import torch
import os
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import sys
import glob

from models.unet import ResUNet
from models.vector_quantized_autoencoder import VQAutoencoder
from models.gaussian_autoencoder import GaussianAutoencoder
from models.data_module import DataModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True

class ImageLogger(pl.Callback):
    def __init__(self,
        embed_dim,
        latent_dim,
        n_slices,
        autoencoder,
        every_n_epochs=10,
        **kwargs
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_slices = n_slices
        self.autoencoder = autoencoder.to(device)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
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
                    self.latent_dim, 
                    self.latent_dim
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


class EMA(pl.Callback):
    def __init__(self, decay_rate=0.999):
        super().__init__()
        self.decay_rate = decay_rate

    def on_train_start(self, trainer, pl_module):
        self.shadow_params = {}
        for name, param in pl_module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                decay = self.decay_rate
                self.shadow_params[name] -= (1 - decay) * (self.shadow_params[name] - param.data)

    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.grad = self.shadow_params[name] - param.data

    def on_train_end(self, trainer, pl_module):
        for name, param in pl_module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    
if __name__ == "__main__":
    global_seed(42)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_float32_matmul_precision('high')
    
    # loading config file
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)
    
    logger = wandb_logger.WandbLogger(
        project='cross_correlation_ldm', 
        name='diffusion'
    )

    # autoencoder
    try:
        target_class = getattr(sys.modules[__name__], config.models.autoencoder.target)
    except:
        raise AttributeError('Unknown autoencoder target')
    
    # load autoencoder weights/hyperparameters
    weights_list = glob.glob(config.callbacks.checkpoint.dirpath + '/' + config.models.autoencoder.target + '*.ckpt')
    autoencoder = target_class.load_from_checkpoint(weights_list[-1]) # latest weights
    print('Using autoencoder: ', type(autoencoder).__name__)
    print('Loaded autoencoder weights from: ', weights_list[-1])
    
    # data module
    data_module = DataModule(
        **config.data,
        autoencoder=autoencoder,
        use_latents=True, 
        batch_size=16, 
        shuffle=True, 
        num_workers=8
    )


    checkpoint_callback = ModelCheckpoint(
        **config.callbacks.checkpoint,
        filename='UNet-{}-{}'.format(type(autoencoder).__name__, '{epoch:02d}')
    )
    image_logger = ImageLogger(
        embed_dim=config.models.autoencoder.embed_dim,
        latent_dim=config.models.autoencoder.latent_dim,
        n_slices=config.data.num_slices,
        autoencoder=autoencoder,
        every_n_epochs=50
    )

    unet_weights = glob.glob(config.callbacks.checkpoint.dirpath + '/UNet*.ckpt')
    if unet_weights.__len__() != 0:
        print('Loaded UNet weights from: ', unet_weights[-1])
        unet = ResUNet.load_from_checkpoint(unet_weights[-1])
    else:    
        unet = ResUNet(**config.models.unet)
    
    # trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=6000,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, image_logger, EMA(decay_rate=0.9999)]
    )
    trainer.fit(unet, data_module)

    