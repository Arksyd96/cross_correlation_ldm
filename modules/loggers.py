import pytorch_lightning as pl
import torch
import wandb
from .models.autoencoder.gaussian_autoencoder import GaussianAutoencoder
from .models.autoencoder.vector_quantized_autoencoder import VQAutoencoder
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageLogger(pl.Callback):
    def __init__(self,
        shape=(2, 32, 32),
        n_slices=64,
        autoencoder=None,
        every_n_epochs=10,
        **kwargs
        ):
        super().__init__()
        self.shape = shape
        self.autoencoder = autoencoder.to(device)
        self.every_n_epochs = every_n_epochs
        self.n_slices = n_slices

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            # sample images
            pl_module.eval()
            with torch.no_grad():
                C, H, W = self.shape
                latents = pl_module.diffusion.sample(
                    pl_module
                ).reshape(
                    C, self.n_slices, H, W
                ).permute(1, 0, 2, 3)

                # selecting sequence of 10 slices with corresponding positions
                latents = latents[::int(self.n_slices/10), ...] # => will be of shape (10, 2, 32, 32)
                position = torch.arange(0, self.n_slices, int(self.n_slices/10)).to(pl_module.device)

                # decoding the latent spaces
                pemb = self.autoencoder.encode_position(position)
                if isinstance(self.autoencoder, GaussianAutoencoder):
                    generated = self.autoencoder.decode(latents, pemb)
                    generated = torch.tanh(generated)
                elif isinstance(self.autoencoder, VQAutoencoder):
                    generated, _, _ = self.autoencoder.decode_pre_quantization(latents, pemb)
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

class ReconstructionImageLogger(pl.Callback):
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
            x_hat = pl_module(x, pos)[0]


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

class FIDLogger(pl.Callback):
    def __init__(self,
        dirpath,
        data_module,
        n_samples,
        normalize=True,
        every_n_epochs=10,
        filename = 'fid_state_dict.pt'
        ):
        super().__init__()
        self.fid = FrechetInceptionDistance(normalize=normalize).float().to(device)
        self.every_n_epochs = every_n_epochs
        self.dirpath = dirpath
        self.filename = filename
        self.data_module = data_module
        self.n_samples = n_samples
        self.noise = torch.randn(n_samples, 2, 64, 32, 32)

    def on_train_start(self, trainer, pl_module):
        # if state exists
        filepath = os.path.join(self.dirpath, self.filename)
        if os.path.exists(filepath):
            fid_state_dict = torch.load(filepath, map_location=device)
            for k, v in fid_state_dict.items():
                setattr(self.fid, k, v)
            
            print('Loaded precomputed FID from {}'.format(filepath))
            del fid_state_dict

        else:
            print('No precomputed FID found at {} ... creating new one!'.format(filepath))
            with torch.no_grad():
                for x in tqdm(self.data_module.data[: self.n_samples], position=0, leave=True, desc='Precomputing FID states'):
                    x = x.to(device)
                    self.fid.update(x, True)
                   
                torch.save({
                    "real_features_sum": self.fid.real_features_sum,
                    "real_features_cov_sum": self.fid.real_features_cov_sum,
                    "real_features_num_samples": self.fid.real_features_num_samples
                }, filepath)


    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            with torch.no_grad():
                for z in tqdm(self.noise, position=0, leave=True, desc='Computing FID'):
                    z = z.to(device)
                    z_hat = pl_module.diffusion.sample(pl_module, z)
                    z_hat = z_hat.transpose(0, 2, 1, 3, 4)
                    
                    # decoding the latent spaces
                    position = torch.arange(0, pl_module.diffusion.T, dtype=torch.long, device=device)
                    pemb = self.autoencoder.encode_position(position)
                    if isinstance(self.autoencoder, GaussianAutoencoder):
                        generated = self.autoencoder.decode(z_hat, pemb)
                        generated = torch.tanh(generated)
                    elif isinstance(self.autoencoder, VQAutoencoder):
                        generated = self.autoencoder.decode_pre_quantization(z_hat, pemb)
                    else:
                        raise NotImplementedError('Unknown autoencoder type')
                    
                    self.fid.update(generated, False)

            fid = self.fid.compute().item()
            pl_module.log('FID', fid, on_step=False, on_epoch=True, logger=True)
