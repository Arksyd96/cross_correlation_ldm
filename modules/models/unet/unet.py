import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl

from ..modules import (
    TimePositionalEmbedding, 
    EncodingBlock, DecodingBlock
)
from ..diffusion import DiffusionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResUNet(pl.LightningModule):
    def __init__(self, 
        input_shape,
        T=1000,
        num_channels=128,
        channel_mult=[1, 2, 2, 4],
        temb_dim=128,
        temb_dim_mult=4,
        attn=[False, False, True, False],
        num_res_blocks=2,
        beta_schedule='cosine',
        learning_rate=1e-5,
        **kwargs
        ) -> None:
        super().__init__()
        assert attn.__len__() == channel_mult.__len__(), 'attn must have the same length as channel_mult'
        self.channel_mult = [1] + channel_mult
        self.reverse_channel_mult = list(reversed(self.channel_mult + [self.channel_mult[-1]]))
        self.temb_latent_dim = temb_dim * temb_dim_mult
        self.learning_rate = learning_rate
        self.T = T
        in_channels = out_channels = input_shape[0]
        
        # diffuser
        self.diffusion = DiffusionModule(
            noise_shape=input_shape,
            T=self.T, 
            beta_schedule=beta_schedule
        )
        
        # architecture modules
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=temb_dim, T=T, device=device),
            nn.Linear(temb_dim, self.temb_latent_dim),
            nn.GELU(),
            nn.Linear(self.temb_latent_dim, self.temb_latent_dim),
        )
        
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same')
        self.encoder = nn.ModuleList([
            EncodingBlock(
                in_channels=num_channels * self.channel_mult[idx],
                out_channels=num_channels * self.channel_mult[idx + 1],
                temb_dim=self.temb_latent_dim,
                downsample=True,
                attn=attn[idx],
                num_blocks=num_res_blocks
            ) for idx in range(self.channel_mult.__len__() - 1)
        ])
        
        self.bottleneck = EncodingBlock(
            in_channels=num_channels * self.channel_mult[-1], 
            out_channels=num_channels * self.channel_mult[-1], 
            temb_dim=self.temb_latent_dim, 
            downsample=False, 
            attn=True, 
            num_blocks=num_res_blocks
        )
        
        self.decoder = nn.ModuleList([
            DecodingBlock(
                in_channels=num_channels * self.reverse_channel_mult[idx] + num_channels * self.reverse_channel_mult[idx + 1],
                out_channels=num_channels * self.reverse_channel_mult[idx + 1],
                temb_dim=self.temb_latent_dim,
                upsample=True,
                attn=attn[-idx - 1],
            ) for idx in range(self.reverse_channel_mult.__len__() - 2)
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=num_channels * 2),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels * 2, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
        # pytorch lightning
        self.save_hyperparameters()

    def forward(self, x, time):
        assert x.shape[0] == time.shape[0], 'Batch size of x and time must be the same'
        temb = self.positional_encoder(time)
        skip_connections = []

        x = self.in_conv(x)
        skip_connections.append(x)
        
        # encoding part
        for block in self.encoder:
            x = block(x, temb)
            skip_connections.append(x)

        # bottleneck
        x = self.bottleneck(x, temb)

        # decoding part
        for block in self.decoder:
            x = block(torch.cat([x, skip_connections.pop()], dim=1), temb)

        x = torch.cat([x, skip_connections.pop()], dim=1)
        assert len(skip_connections) == 0, 'Skip connections must be empty'
        return self.out_conv(x)
    
    def training_step(self, batch, batch_idx):
        z_q = batch[0].type(torch.float16)
        B = z_q.shape[0]
        
        # forward step
        times = torch.randint(low=0, high=self.T, size=(B,), device=device, dtype=torch.long)
        x_t, noise = self.diffusion.forward_process(z_q, times)
        x_t = x_t.to(device, dtype=torch.float16)
        
        # backward step
        noise_hat = self.diffusion.reverse_process(self, x_t, times)
        
        # loss
        loss = F.mse_loss(noise_hat, noise)
        self.log('mse_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_steps, eta_min=1e-9, last_epoch=-1
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
    