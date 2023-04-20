import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import pytorch_lightning as pl
from nibabel.processing import resample_to_output
from nibabel import load
from omegaconf import OmegaConf

from models.vector_quantized_autoencoder import VQAutoencoder
from models.lpips import VQLPIPSWithDiscriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

# function that format the display of a dict
def format_dict(d):
    return ', '.join([f'{k}: {v:.4f}' for k, v in d.items()])

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]



##########################################################################################
if __name__ == "__main__":
    cfg = './config/config.yaml'
    c = OmegaConf.load(cfg)
    print(c.models.autoencoder.attn)
    
    # loading dataset
    data = np.load('./data/brats_3d_dataset.npy')
    flair, t1ce = data[:, 0, None, ...], data[:, 1, None, ...]
    N_VOLUMES, _, W, H, N_SLICES = flair.shape
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # normalizing the data between -1 and 1
    n_max_flair = [flair[i].max() for i in range(N_VOLUMES)]
    n_max_t1ce = [t1ce[i].max() for i in range(N_VOLUMES)]

    for idx in range(0, N_VOLUMES):
        flair[idx] = (2 * flair[idx] / n_max_flair[idx] - 1).astype(np.float32)
        t1ce[idx] = (2 * t1ce[idx] / n_max_t1ce[idx] - 1).astype(np.float32)

    flair, t1ce = flair.clip(-1, 1), t1ce.clip(-1, 1)

    # keeping track on slice positions for positional embedding
    slice_positions = np.arange(N_SLICES)[None, :].repeat(N_VOLUMES, axis=0)
    slice_positions = slice_positions.flatten()

    # switching to 2D
    flair = flair.transpose(0, 4, 1, 2, 3).reshape(N_VOLUMES * N_SLICES, -1, W, H)
    t1ce = t1ce.transpose(0, 4, 1, 2, 3).reshape(N_VOLUMES * N_SLICES, -1, W, H)

    # removing empty slices
    # empty_slices_indices = np.where(np.any(flair, axis=(1, 2, 3)) == True)[0]
    # flair, t1ce, mask = flair[empty_slices_indices], t1ce[empty_slices_indices], mask[empty_slices_indices]

    # dataset and dataloader
    train_dataset = IdentityDataset(flair, t1ce, slice_positions)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8, pin_memory=True
    )

    print('Data shape: {}'.format(flair.shape))
    print('Max: {}'.format(flair.max()))
    print('Min: {}'.format(flair.min()))
    print('Data loaded')
    
    # data_module = BratsDataModule(c.data)
    # data_module.prepare_data()
    # data_module.setup()
    
    # train_loader = data_module.train_dataloader()
    
    b = next(iter(train_loader))
    print(b[0].shape, b[0].dtype)
    
    autoencoder = VQAutoencoder(**c.models.autoencoder, **c.models.autoencoder.loss).to(device)
    
    AMP = True
    accumulation_steps = 1
    torch.backends.cudnn.benchmark = True

    # model = torch.compile(model)
    
    loss = VQLPIPSWithDiscriminator(**c.models.autoencoder.loss).to(device)

    lr_g_factor = 1.0
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.00001, weight_decay=1e-7, betas=(0.5, 0.9))
    d_optimizer = torch.optim.AdamW(loss.discriminator.parameters(), lr=lr_g_factor * 0.00001, weight_decay=1e-7, betas=(0.5, 0.9))

    # ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     ae_optimizer, T_max=train_loader.__len__() * AE_EPOCHS, eta_min=1e-9, last_epoch=-1
    # )

    ae_scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    d_scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    total_loss_history = []

    for epoch in range(10):
        autoencoder.train()
        
        ae_optimizer.zero_grad(set_to_none=True)
        total_accumulated_loss = []
        ae_accumulate_loss, d_accumulated_loss = [], []

        progress = tqdm(train_loader, position=0, leave=True)
        for idx, (x_a, x_b, pos) in enumerate(train_loader):
            B = x_a.shape[0]
            x_a, x_b = x_a.to(device, dtype=torch.float32), x_b.to(device, dtype=torch.float32)
            pos = pos.to(device, dtype=torch.long)

            with torch.autocast(device.type, enabled=AMP):
                x = torch.cat([x_a, x_b], dim=1)
                x_recon, z_i, qloss, ind = autoencoder(x, pos, return_indices=True)
                
                ae_loss, log_dict_ae = loss.autoencoder_loss(
                    qloss, x, torch.tanh(x_recon), z_i, epoch * len(train_loader) + idx, last_layer=autoencoder.decoder.out_conv[-1].weight
                )
                
                d_loss, log_dict_disc = loss.discriminator_loss(
                    x, torch.tanh(x_recon.detach()), epoch * len(train_loader) + idx
                )

            log_dict = {**log_dict_ae, **log_dict_disc}

            ae_scaler.scale(ae_loss).backward()
            d_scaler.scale(d_loss).backward()

            if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(train_loader):
                ae_scaler.unscale_(ae_optimizer)
                ae_scaler.step(ae_optimizer)
                ae_scaler.update()
                # ae_scheduler.step()
                ae_optimizer.zero_grad(set_to_none=True)

                d_scaler.unscale_(d_optimizer)
                d_scaler.step(d_optimizer)
                d_scaler.update()
                d_optimizer.zero_grad(set_to_none=True)

                ae_accumulate_loss.append(ae_loss.item())
                d_accumulated_loss.append(d_loss.item())
                total_accumulated_loss.append((ae_loss + d_loss).item())

                with torch.no_grad():
                    progress.update(accumulation_steps)
                    progress.set_description(
                        format_dict(log_dict)
                    )

        total_loss_history.append(np.mean(total_accumulated_loss))

        n_sample = c.data.batch_size if c.data.batch_size < 5 else 5
        if (epoch + 1) % 5 == 0:
            for c in range(c.data.in_channels):
                with torch.no_grad():
                    plt.figure(figsize=(10, 2))
                    for i in range(n_sample):
                        plt.subplot(1, n_sample, i + 1)
                        plt.imshow(torch.tanh(x_recon[i, c, :, :]).detach().cpu().numpy(), cmap='gray')
                        plt.axis('off')
                    plt.show()
