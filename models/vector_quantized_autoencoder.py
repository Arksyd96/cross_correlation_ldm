import numpy as np
import torch
import torch.nn as nn

from .modules import Encoder, Decoder
from .vector_quantizer import VectorQuantizer

class VQAutoencoder(nn.Module):
    def __init__(self,
        in_channels, out_channels, n_embed, embed_dim,
        z_channels=4, z_double=False, 
        num_channels=128, channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, attn=None
    ) -> None:
        super().__init__()
        if attn is not None:
            assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
        else:
            self.attn = [False] * channels_mult.__len__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.z_channels = z_channels if not z_double else z_channels * 2

        # architecture modules
        self.encoder = Encoder(in_channels, z_channels, z_double, num_channels, channels_mult, num_res_blocks, self.attn)
        self.decoder = Decoder(out_channels, z_channels, z_double, num_channels, channels_mult, num_res_blocks, self.attn)
        self.quantizer = VectorQuantizer(self.n_embed, self.embed_dim, beta=0.25, remap=None)
        self.quant_conv = nn.Conv2d(self.z_channels, self.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_channels, kernel_size=1)

        # TODO: Add EMA

    def encode(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)
        z_q, loss, info = self.quantizer(z)
        return z_q, loss, info
    
    def encode_pre_quantization(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)
        return z
    
    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)
        x = self.decoder(z_q)
        return x
    
    def decode_code(self, code_b):
        z_q = self.quantizer.embedding(code_b)
        x = self.decode(z_q)
        return x
    
    def forward(self, x, return_indices=False):
        z_q, loss, (_, _, indices) = self.encode(x)
        x = self.decode(z_q)
        if return_indices:
            return x, loss, indices
        return x, loss