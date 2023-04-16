import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .modules import (
    TimePositionalEmbedding, 
    EncodingBlock, DecodingBlock
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels, T=1000) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, 128, kernel_size=3, padding='same')
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=128, T=T, device='cuda'),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        self.encoder = nn.ModuleList([
            EncodingBlock(in_channels=128, out_channels=128, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=128, out_channels=256, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=256, temb_dim=128 * 4, downsample=True, attn=True, num_blocks=2, groups=32),
            EncodingBlock(in_channels=256, out_channels=512, temb_dim=128 * 4, downsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.bottleneck = EncodingBlock(in_channels=512, out_channels=512, temb_dim=128 * 4, downsample=False, attn=True, num_blocks=2, groups=32)

        self.decoder = nn.ModuleList([
            DecodingBlock(in_channels=512 + 512, out_channels=512, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=512 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=True, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 256, out_channels=256, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32),
            DecodingBlock(in_channels=256 + 128, out_channels=128, temb_dim=128 * 4, upsample=True, attn=False, num_blocks=2, groups=32)
        ])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, padding=1)
        )

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