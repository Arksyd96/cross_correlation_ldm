import numpy as np
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import sys
import glob

# from models.unet import ResUNet
from modules.models.unet.unet_3d import ResUNet3D
from modules.models.unet.unet import ResUNet
from modules.models.autoencoder.vector_quantized_autoencoder import VQAutoencoder
from modules.models.autoencoder.gaussian_autoencoder import GaussianAutoencoder
from modules.data_module import DataModule
from modules.loggers import ImageLogger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    global_seed(42)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_float32_matmul_precision('high')
    
    # loading config file
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)

    dims = '3D' if len(config.models.unet.input_shape) == 4 else ''
    
    logger = wandb_logger.WandbLogger(
        project='cross_correlation_ldm', 
        name='diffusion{}-{}'.format(
            dims,
            config.models.autoencoder.target
        ),
    )

    try: # autoencoder
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
        depth_as_channels=False,
        batch_size=16, 
        shuffle=True, 
        num_workers=8
    )

    checkpoint_callback = ModelCheckpoint(
        **config.callbacks.checkpoint,
        filename='UNet{}-{}-{}'.format(
            dims, type(autoencoder).__name__, '{epoch:02d}'
        )
    )

    image_logger = ImageLogger(
        shape=config.models.autoencoder.latent_shape,
        num_slices=config.data.shape[0],
        autoencoder=autoencoder,
        every_n_epochs=1
    )

    # fid_logger = FIDLogger(
    #     **config.callbacks.fid,
    #     data_module=data_module,
    # )

    # unet
    try:
        unet_target_class = getattr(sys.modules[__name__], 'ResUNet{}'.format(dims))
    except:
        raise AttributeError('Unknown UNet target')
    
    # loading unet weights
    unet_weights = glob.glob(config.callbacks.checkpoint.dirpath + '/UNet{}*.ckpt'.format(dims))

    if unet_weights.__len__() != 0 and config.models.unet.from_checkpoint:
        unet = unet_target_class.load_from_checkpoint(unet_weights[-1])
        print('Loaded UNet weights from: ', unet_weights[-1])
    else:
        unet = unet_target_class(**config.models.unet)

    print('Using UNet: ', type(unet).__name__)
    
    # trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=20000,
        log_every_n_steps=1,
        enable_progress_bar=True,
        fast_dev_run=5,
        callbacks=[checkpoint_callback, image_logger]
    )
    trainer.fit(unet, data_module)

    