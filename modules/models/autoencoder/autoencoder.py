import numpy as np
import torch
import torch.nn as nn

from ..modules import EncodingBlock, DecodingBlock

class Encoder(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1), # 64x128x128
            EncodingBlock(64, 64, downsample=True, attn=False, num_blocks=2, groups=8), # 64x64x32
            EncodingBlock(64, 128, downsample=True, attn=False, num_blocks=2, groups=8), # 32x32x16
            EncodingBlock(128, 128, downsample=True, attn=False, num_blocks=2, groups=8), # 16x16x8
            EncodingBlock(128, 256, downsample=True, attn=False, num_blocks=2, groups=8), # 8x8x4
            EncodingBlock(256, 512, downsample=False, attn=False, num_blocks=2, groups=8), # 8x8x4
            # EncodingBlock(256, 512, downsample=False, attn=True, num_blocks=2, groups=8), # 8x8x4
        ])

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, out_channels) -> None:
        super().__init__()
        self.decoder = nn.ModuleList([
            DecodingBlock(512 + 512, 256, upsample=False, attn=False, num_blocks=2, groups=8), # 8x8x4
            # DecodingBlock(512, 256, upsample=False, attn=True, num_blocks=2, groups=8), # 8x8x4
            DecodingBlock(256, 128, upsample=True, attn=False, num_blocks=2, groups=8), # 16x16x8
            DecodingBlock(128, 128, upsample=True, attn=False, num_blocks=2, groups=8), # 32x32x16
            DecodingBlock(128, 64, upsample=True, attn=False, num_blocks=2, groups=8), # 64x64x32
            DecodingBlock(64, 64, upsample=True, attn=False, num_blocks=2, groups=8), # 128x128x64
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1) # 1x128x128x64
        ])

    def forward(self, x):
        for block in self.decoder:
            x = block(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([
            Encoder(in_channels=1) for _ in range(in_channels)
        ])
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        assert x.shape[1] == len(self.encoders), "Number of channels in input must match number of encoders"
        z = torch.cat([
            encoder(x[:, i, ...].unsqueeze(1)) for i, encoder in enumerate(self.encoders)
        ], dim=1)
        # => correlation between channels here
        return self.decoder(z)

