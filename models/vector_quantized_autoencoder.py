import numpy as np
import torch
import torch.nn as nn

from .modules import (
    TimePositionalEmbedding, EncodingBlock, DecodingBlock,
    ResidualBlock, SelfAttention
)
from .vector_quantizer import VectorQuantizer
from .lpips import VQLPIPSWithDiscriminator

class Encoder(nn.Module):
    def __init__(
        self, in_channels, z_channels=4, z_double=True, pemb_dim=None, num_channels=128, channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, attn=None
        ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = attn
        else:
            self.attn = [False] * channels_mult.__len__()

        self.z_channels = z_channels if not z_double else z_channels * 2
        self.channels_mult = [1, *channels_mult]
        self.attn = attn
        
        # architecture modules
        self.in_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same')
        self.enocoder = nn.ModuleList([
            EncodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                downsample=True if idx != self.channels_mult.__len__() - 2 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        bottleneck_channels = num_channels * self.channels_mult[-1]
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=bottleneck_channels, out_channels=self.z_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        for encoder in self.enocoder:
            x = encoder(x, pemb)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        x = self.out_conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(
        self, out_channels, z_channels, z_double=True, pemb_dim=None, num_channels=128, channels_mult=[1, 2, 4, 4],
        num_res_blocks=2, attn=None
        ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = list(reversed(attn))
        else: 
            self.attn = [False] * channels_mult.__len__()

        self.channels_mult = list(reversed([1, *channels_mult]))
        self.z_channels = z_channels if not z_double else z_channels * 2
        
        # architecture modules
        bottleneck_channels = num_channels * self.channels_mult[0]
        self.in_conv = nn.Conv2d(self.z_channels, bottleneck_channels, kernel_size=3, padding='same')
        self.bottleneck_res_a = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)
        self.bottleneck_sa = SelfAttention(in_channels=bottleneck_channels, num_heads=8, head_dim=32, groups=8)
        self.bottleneck_res_b = ResidualBlock(in_channels=bottleneck_channels, out_channels=bottleneck_channels, temb_dim=pemb_dim, groups=8)

        self.decoder = nn.ModuleList([
            DecodingBlock(
                in_channels=num_channels * self.channels_mult[idx],
                out_channels=num_channels * self.channels_mult[idx + 1],
                temb_dim=pemb_dim,
                num_blocks=num_res_blocks,
                attn=self.attn[idx],
                upsample=True if idx != 0 else False
            ) for idx in range(self.channels_mult.__len__() - 1)
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=num_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, pemb=None):
        x = self.in_conv(x)
        x = self.bottleneck_res_a(x, pemb)
        x = self.bottleneck_sa(x)
        x = self.bottleneck_res_b(x, pemb)
        for decoder in self.decoder:
            x = decoder(x, pemb)
        x = self.out_conv(x)
        return x

class VQAutoencoder(nn.Module):
    def __init__(self,
        in_channels, 
        out_channels, 
        n_embed, 
        embed_dim,
        z_channels=4, 
        z_double=False, 
        pemb_dim=128, 
        T=64,
        num_channels=128, 
        channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, 
        attn=None,
        disc_start=11, 
        codebook_weight=1., 
        pixel_weight=1., 
        perceptual_weight=1., 
        disc_weight=1., 
        cos_weight=1.,
        disc_input_channels=3, 
        disc_channels=64, 
        disc_num_layers=3, 
        disc_factor=1., 
        **kwargs
    ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
            self.attn = attn
        else:
            self.attn = [False] * channels_mult.__len__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.z_channels = z_channels if not z_double else z_channels * 2

        # architecture modules
        self.positional_encoder = nn.Sequential(
            TimePositionalEmbedding(dimension=pemb_dim, T=T, device='cuda'),
            nn.Linear(pemb_dim, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, pemb_dim)
        )
        self.encoders = nn.ModuleList([
            Encoder(1, z_channels, z_double, pemb_dim, num_channels, channels_mult, num_res_blocks, self.attn) for _ in range(in_channels)
        ])
        decoder_in_channels = in_channels * z_channels
        vq_embed_dim = self.embed_dim * in_channels
        self.decoder = Decoder(out_channels, decoder_in_channels, z_double, pemb_dim, num_channels, channels_mult, num_res_blocks, self.attn)
        self.quantizer = VectorQuantizer(self.n_embed, vq_embed_dim, beta=0.25, remap=None)
        self.quant_conv = nn.Conv2d(decoder_in_channels, vq_embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, decoder_in_channels, kernel_size=1)

        # loss functions
        self.loss = VQLPIPSWithDiscriminator(
            disc_start, codebook_weight, pixel_weight, perceptual_weight, disc_weight, cos_weight,
            disc_input_channels, disc_channels, disc_num_layers, disc_factor
        )

        # TODO: Add EMA

    def encode(self, x, pemb):
        z_i = []
        # forwarding each channel through its own encoder
        for c_i, encoder in enumerate(self.encoders):
            x_i = x[:, c_i, None]
            z_i.append(encoder(x_i, pemb))

        # concatenating the channels
        z = torch.cat(z_i, dim=1)
        z = self.quant_conv(z)
        z_q, qloss, info = self.quantizer(z)
        return z_q, z_i, qloss, info
    
    def encode_pre_quantization(self, x, pemb):
        z_i = []
        # forwarding each channel through its own encoder
        for c_i, encoder in enumerate(self.encoders):
            z_i.append(encoder(x[:, c_i, None], pemb))

        # concatenating the channels
        z = torch.cat(z_i, dim=1)
        z = self.quant_conv(z)
        return z, z_i
    
    def decode(self, z_q, pemb):
        z_q = self.post_quant_conv(z_q)
        # TODO: Cross-attention avec mask ici
        x = self.decoder(z_q, pemb)
        return x
    
    def decode_code(self, code_b, pemb):
        z_q = self.quantizer.embedding(code_b)
        x = self.decode(z_q, pemb)
        return x
    
    def forward(self, x, position, return_indices=False):
        pemb = self.positional_encoder(position)
        z_q, z_i, qloss, (_, _, indices) = self.encode(x, pemb)
        x = self.decode(z_q, pemb)
        if return_indices:
            return x, z_i, qloss, indices
        return x, z_i, qloss
    
    def checkpoint(self, path, epoch, optimizer, scheduler):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, path)
    
