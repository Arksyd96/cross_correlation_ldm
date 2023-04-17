import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Encoder, Decoder

class GaussianAutoencoder(nn.Module):
    def __init__(self, 
        in_channels, out_channels, embed_dim ,
        z_channels=4, z_double=True, 
        num_channels=128, channels_mult=[1, 2, 4, 4], 
        num_res_blocks=2, attn=[False, False, False, False],
        learn_logvar=True
    ) -> None:
        super().__init__()
        assert channels_mult.__len__() == attn.__len__(), 'channels_mult and attn must have the same length'
        self.z_channels = z_channels if not z_double else z_channels * 2
        self.learn_logvar = learn_logvar
        self.embed_dim = embed_dim
        
        # architecture modules
        self.encoder = Encoder(in_channels, z_channels, z_double, num_channels, channels_mult, num_res_blocks, attn)
        self.decoder = Decoder(out_channels, z_channels, z_double, num_channels, channels_mult, num_res_blocks, attn)
        self.quant_conv = nn.Conv2d(self.z_channels, 2 * self.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_channels, kernel_size=1)
        
        # semf.image_key = image_key
        # self.loss = instantiate_from_config(lossconfig)
        # assert ddconfig["double_z"]
        # TODO: Add EMA
        
    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x = self.decode(z)
        return x, posterior
    
    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        return x
        
        
        
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
