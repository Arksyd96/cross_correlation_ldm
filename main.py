import numpy as np
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from nibabel.processing import resample_to_output
from nibabel import load
from omegaconf import OmegaConf


def get_parser(**parser_kwargs):
    pass

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class BratsDataModule(pl.LightningDataModule):
    def __init__(self, dataConf: OmegaConf, **kwargs):
        super().__init__()
        self.dc = dataConf

    def prepare_data(self) -> None:
        if not os.path.exists(self.dc.npy_path):
            print('Loading dataset from NiFTI files...')
            placeholder = np.zeros(shape=(
                self.dc.n_samples, self.dc.num_modalities, self.dc.resolution, self.dc.resolution
            ))

            for idx, instance in enumerate(tqdm(os.listdir(self.dc.root_path)[: self.dc.n_samples], position=0, leave=True)):
                # loading models
                volumes = {}
                for _, m in enumerate(self.dc.modalities):
                    volumes[m] = load(os.path.join(self.dc.root_path, instance, instance + f'_{m}.nii.gz'))

                # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
                scale_factor = (self.dc.orig_resolution / self.dc.resolution,
                                self.dc.orig_resolution / self.dc.resolution,
                                self.dc.orig_num_slices / self.dc.num_slices)

                # Resample the image using trilinear interpolation
                # Drop the last extra rows/columns/slices to get the exact desired output size
                for _, m in enumerate(self.dc.modalities):
                    volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
                    volumes[m] = volumes[m][:self.dc.resolution, :self.dc.resolution, :self.dc.num_slices]

                # binarizing the mask (for simplicity), you can comment out this to keep all labels
                if self.dc.binarize and 'seg' in self.dc.modalities:
                    volumes['mask'] = (volumes['mask'] > 0).astype(np.float32)

                # saving models
                for _, m in enumerate(self.dc.modalities):
                    placeholder[idx, _, :, :] = volumes[m]

            # saving the dataset as a npy file
            np.save(self.dc.npy_path, placeholder)

        else:
            print('Loading dataset from npy file...')
            data = np.load(self.dc.npy_path)

        self.volumes = {}
        for _, m in enumerate(self.dc.modalities):
            self.volumes[m] = data[:, _, None, :, :]

        for _, m in enumerate(self.dc.modalities):
            for idx in range(self.dc.n_samples):
                self.volumes[m][idx] = self.normalize(self.volumes[m][idx])

        # switching to 2D
        for _, m in enumerate(self.dc.modalities):
            self.volumes[m] = self.volumes[m].transpose(0, 4, 1, 2, 3)
            self.volumes[m] = self.volumes[m].reshape(self.dc.n_samples * self.dc.num_slices, -1, self.dc.resolution, self.dc.resolution)

        # keeping track on slice positions for positional embedding
        self.slice_positions = np.arange(self.dc.num_slices)[None, :].repeat(self.dc.n_samples, axis=0)
        self.slice_positions = self.slice_positions.flatten()

        # removing empty slices
        # empty_slices_indices = np.where(np.any(flair, axis=(1, 2, 3)) == True)[0]
        # flair, t1ce, mask = flair[empty_slices_indices], t1ce[empty_slices_indices], mask[empty_slices_indices]

        print('Modalities: ', self.dc.modalities)
        print('Data shape: ', self.volumes[self.dc.modalities[0]].shape)
        print('Data prepared')

    # normalizes data between -1 and 1
    def normalize(self, data):
        return (data * 2 / data.max() - 1).astype(np.float32)
        
    def setup(self, stage='fit'):
        self.dataset = IdentityDataset(*[self.volumes.values] + [self.slice_positions])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.dc.batch_size, shuffle=self.dc.shuffle, num_workers=self.dc.num_workers, pin_memory=True
        )

class CheckpointCallback(pl.Callback):
    def __init__(self, ckptConfig: OmegaConf, **kwargs):
        self.cc = ckptConfig
        self.epoch_counter = 0

    def on_epoch_end(self, trainer, pl_module):
        self.epoch_counter += 1
        if self.epoch_counter % self.cc.save_ckpt_at_every == 0:
            pl_module.checkpoint(
                save_path=os.path.join(self.cc.save_path, f"epoch_{self.epoch_counter}.pth")
            )
        if self.epoch_counter % self.cc.save_log_at_every == 0:
            pl_module.log(
                save_path=os.path.join(self.cc.save_path, f"epoch_{self.epoch_counter}.pth")
            )


if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()
    cfg = './config/config.yaml'
    config = OmegaConf.load(cfg)
    data_module = BratsDataModule(config.data)
    ckpt_callback = CheckpointCallback(config.checkpointCallback)