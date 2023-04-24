import numpy as np
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from models.unet import ResUNet
from models.vector_quantized_autoencoder import VQAutoencoder

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self, 
        npy_path,
        autoencoder,
        n_samples=500,
        resolution=128,
        modalities=['t1', 't1ce', 't2', 'flair'],
        batch_size=32,
        shuffle=True,
        num_workers=4,
        **kwargs
        ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.autoencoder = autoencoder.to(self.device)
        
        # pl
        self.save_hyperparameters(ignore=['autoencoder'])

    def prepare_data(self):
        print('Loading dataset from npy file... and encoding to latents')
        data = np.load(self.hparams.npy_path)

        self.volumes = {}
        for _, m in enumerate(self.hparams.modalities):
            self.volumes[m] = torch.from_numpy(data[:, _, None, :, :])

        for _, m in enumerate(self.hparams.modalities):
            for idx in range(self.hparams.n_samples):
                self.volumes[m][idx] = self.normalize(self.volumes[m][idx]).type(torch.float32)

        # switching to 2D
        for _, m in enumerate(self.hparams.modalities):
            self.volumes[m] = self.volumes[m].permute(0, 4, 1, 2, 3)

        # encoding each slice
        z_latents = []
        self.autoencoder.eval()
        for idx in tqdm(range(self.hparams.n_samples), position=0, leave=True):
            input = torch.cat([self.volumes[m][idx] for m in self.hparams.modalities], dim=1).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                pos = torch.arange(0, 64, device=self.device, dtype=torch.long)
                pemb = autoencoder.encode_position(pos)
                z, _ = autoencoder.encode_pre_quantization(input, pemb)
                z_latents.append(z.detach().cpu())
                
        z_latents = torch.stack(z_latents)
        self.z_latents = z_latents.reshape(self.hparams.n_samples, -1, self.hparams.resolution, self.hparams.resolution)    

        print('Modalities: ', self.hparams.modalities)
        print('Data shape: ', self.z_latents.shape)
        print('Data prepared')

    # normalizes data between -1 and 1
    def normalize(self, data):
        return data * 2 / data.max() - 1
        
    def setup(self, stage='fit'):
        self.dataset = IdentityDataset(self.z_latents)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            drop_last=True
        )
        
class CheckpointCallback(pl.Callback):
    def __init__(self, 
        log_path,
        ckpt_path,
        save_ckpt_at_every,
        save_log_at_every,
        **kwargs
        ):
        self.epoch_counter = 0
        self.log_path = log_path
        self.ckpt_path = ckpt_path
        self.save_ckpt_at_every = save_ckpt_at_every
        self.save_log_at_every = save_log_at_every
        
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

    def on_epoch_end(self, trainer, pl_module):
        self.epoch_counter += 1
        if self.epoch_counter % self.save_ckpt_at_every == 0:
            pl_module.checkpoint(
                save_path=os.path.join(self.ckpt_path, f"epoch_{self.epoch_counter}.pth")
            )
    
def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
if __name__ == "__main__":
    global_seed(42)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_float32_matmul_precision('high')
    
    # loading config file
    CONFIG_PATH = './config/config.yaml'
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError('Config file not found')
    
    config = OmegaConf.load(CONFIG_PATH)
    
    # TODO: specify first stage training or diffusion training
    
    wandb_logger = wandb.WandbLogger(
        project='cross_correlation_ldm', 
        name='diffusion'
    )
    
    autoencoder = VQAutoencoder(**config.models.autoencoder)
    datamodule = DiffusionDataModule(**config.data, autoencoder=autoencoder, batch_size=8)
    unet = ResUNet(**config.models.unet)
    ckpt_callback = CheckpointCallback(**config.callbacks.checkpoint)
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        precision='16-mixed',
        max_epochs=1000,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[ckpt_callback]
    )
    trainer.fit(unet, datamodule)

    