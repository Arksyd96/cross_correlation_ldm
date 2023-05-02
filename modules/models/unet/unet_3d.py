import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from ..diffusion import DiffusionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimePositionalEmbedding(nn.Module):
    def __init__(self, dimension, T=1000, device=None) -> None:
        super().__init__()
        self.embedding = torch.zeros(T, dimension)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * (-np.log(10000.0) / dimension))
        self.embedding[:, 0::2] = torch.sin(position * div_term)
        self.embedding[:, 1::2] = torch.cos(position * div_term)
        if device is not None:
            self.embedding = self.embedding.to(device)
    
    def forward(self, timestep):
        return self.embedding[timestep]
    
class WeightStandardizedConv3d(nn.Conv3d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = self.weight.mean(dim=[1, 2, 3, 4], keepdim=True)
        var = torch.var(self.weight, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        normalized_weight = (self.weight - mean) * (var + eps).rsqrt()

        return F.conv3d(
            x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8) -> None:
        super().__init__()
        self.conv = WeightStandardizedConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super().__init__()
        self.downsampler = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
    
    def forward(self, x):
        return self.downsampler(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super().__init__()
        self.upsampler = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=scale_factor, padding=1)
    
    def forward(self, x):
        return self.upsampler(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, groups=8) -> None:
        super().__init__()
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_channels),
        ) if temb_dim is not None else nn.Identity()

        self.block_a = ConvBlock(in_channels, out_channels, groups=groups)
        self.block_b = ConvBlock(out_channels, out_channels, groups=groups)
        self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb=None):
        h = self.block_a(x)
        if temb is not None:
            h = h + self.temb_proj(temb)[:, :, None, None, None]
        h = self.block_b(h)
        return h + self.residual_proj(x)
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=32, groups=32) -> None:
        super().__init__()
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Conv3d(in_channels, num_heads * head_dim, kernel_size=1)
        self.k = nn.Conv3d(in_channels, num_heads * head_dim, kernel_size=1)
        self.v = nn.Conv3d(in_channels, num_heads * head_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, in_channels)
        self.proj = nn.Conv3d(num_heads * head_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W, D = x.shape
        q = self.q(x).view(B, self.num_heads, self.head_dim, H * W * D).permute(0, 1, 3, 2)
        k = self.k(x).view(B, self.num_heads, self.head_dim, H * W * D)
        v = self.v(x).view(B, self.num_heads, self.head_dim, H * W * D).permute(0, 1, 3, 2)

        attention = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        attention = torch.matmul(attention, v)
        attention = attention.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, H, W, D)
        return self.norm(x + self.proj(attention))
    
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, downsample=True, attn=False, num_blocks=2, groups=32) -> None:
        super().__init__()

        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim=temb_dim, groups=groups)
            for i in range(num_blocks)
        ])

        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])

        self.downsample = Downsample(out_channels, out_channels) if downsample else nn.Identity()

    def forward(self, x, temb=None):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb)
            x = attn_block(x)
        return self.downsample(x)
    
class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None, upsample=True, attn=False, num_blocks=2, groups=32) -> None:
        super().__init__()
        self.resnet = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, temb_dim=temb_dim, groups=groups)
            for i in range(num_blocks)
        ])
        self.attn = nn.ModuleList([
            SelfAttention(out_channels, num_heads=8, head_dim=32, groups=groups)
            if attn else nn.Identity()
            for _ in range(num_blocks)
        ])
        self.upsample = Upsample(out_channels, out_channels) if upsample else nn.Identity()

    def forward(self, x, temb=None):
        for resnet_block, attn_block in zip(self.resnet, self.attn):
            x = resnet_block(x, temb)
            x = attn_block(x)
        return self.upsample(x)
    

class ResUNet3D(pl.LightningModule):
    def __init__(self, 
        in_channels, 
        out_channels, 
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
        
        # diffuser
        self.diffusion = DiffusionModule(T=self.T, beta_schedule=beta_schedule)
        
        # architecture modules
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=temb_dim, T=T, device=device),
            nn.Linear(temb_dim, self.temb_latent_dim),
            nn.GELU(),
            nn.Linear(self.temb_latent_dim, self.temb_latent_dim),
        )
        
        self.in_conv = nn.Conv3d(in_channels, num_channels, kernel_size=3, padding='same')
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
            nn.Conv3d(in_channels=num_channels * 2, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
        # pl
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
        z = batch[0].type(torch.float16)
        B = z.shape[0]
        
        # forward step
        times = torch.randint(low=0, high=self.T, size=(B,), device=device, dtype=torch.long)
        x_t, noise = self.diffusion.forward_process(z, times)
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
    
    
    