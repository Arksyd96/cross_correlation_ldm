"""
    Adapted from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple
import os, hashlib
import requests
from tqdm import tqdm
from .discriminator import NLayerDiscriminator, weights_init

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        # if input.shape[1] < 3:  # the standard vgg in channels is 3
        #     B, C, H, W = input.shape
        #     in0_input = torch.cat([input, torch.zeros(size=(B, 3 - C, H, W)).to(input.device)], dim=1)
        #     in1_input = torch.cat([target, torch.zeros(size=(B, 3 - C, H, W)).to(target.device)], dim=1)
    
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        # self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])
        self.register_buffer('shift', torch.Tensor([-.030, -.088])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, weights=models.VGG16_Weights.DEFAULT):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=weights).features
        
        # getting weights of input channel
        selected_weights = vgg_pretrained_features[0].weight[:, :2, ...]
        vgg_pretrained_features[0] = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg_pretrained_features[0].weight = nn.Parameter(selected_weights, requires_grad=False)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def spatial_average(x, keepdim=True):
    return x.mean([2, 3],keepdim=keepdim)

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)
                        
def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
            disc_start, codebook_weight=1., pixel_weight=1., perceptual_weight=1., disc_weight=1.,
            d_input_channels=3, d_channels=64, d_num_layers=3, disc_factor=1.
        ) -> None:
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        
        # modules
        self.lpips = LPIPS().eval()
        self.discriminator = NLayerDiscriminator(d_input_channels, d_channels, d_num_layers, use_actnorm=False).apply(weights_init)
        self.disc_start = disc_start

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.disc_weight
        return d_weight

    def forward(self, codebook_loss, x, recon_x, optimizer_idx, global_step, last_layer=None):
        rec_loss = F.l1_loss(recon_x, x, reduction='none')
        if self.perceptual_weight > 0:
            p_loss = self.lpips(x, recon_x)
            rec_loss = rec_loss + p_loss * self.perceptual_weight # pixel-wise addition of the p_loss

        nll_loss = rec_loss.mean()

        # discriminator loss
        split = "train" if self.training else "val"
        if optimizer_idx == 0:
            logits_fake = self.discriminator(recon_x.contiguous())
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
            except RuntimeError:
                d_weight = torch.tensor(0.0)

            d_factor = self.disc_factor if global_step > self.disc_start else 0.0
            loss = nll_loss + d_weight * d_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(d_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log
        
        if optimizer_idx == 1:
            logits_real = self.discriminator(x.contiguous().detach())
            logits_fake = self.discriminator(recon_x.contiguous().detach())

            d_factor = self.disc_factor if global_step > self.disc_start else 0.0
            d_loss = d_factor * self.hinge_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

    def hinge_loss(self, logits_real, logits_fake):
        loss = 0.5 * (torch.mean(F.relu(1.0 - logits_real)) + torch.mean(F.relu(1.0 + logits_fake)))
        return loss
        