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
    def __init__(self, dataconf: OmegaConf, **kwargs):
        super().__init__()
        self.dc = dataconf

    def prepare_data(self) -> None:
        if not os.path.exists(self.dc.npy_path):
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

    # normalizes data between -1 and 1
    def normalize(self, data):
        return (data * 2 / data.max() - 1).astype(np.float32)
        
    def setup(self, stage='fit'):
        # loading dataset
        data = np.load(self.dc.npy_path)

        volumes = {}
        for _, m in enumerate(self.dc.modalities):
            volumes[m] = data[:, _, None, :, :]

        for _, m in enumerate(self.dc.modalities):
            for idx in range(self.dc.n_samples):
                volumes[m][idx] = self.normalize(volumes[m][idx])

        # keeping track on slice positions for positional embedding
        slice_positions = np.arange(self.dc.num_slices)[None, :].repeat(self.dc.n_samples, axis=0)
        slice_positions = slice_positions.flatten()

        # switching to 2D
        for _, m in enumerate(self.dc.modalities):
            volumes[m] = volumes[m].transpose(0, 4, 1, 2, 3)
            volumes[m] = volumes[m].reshape(self.dc.n_samples * self.dc.num_slices, -1, self.dc.resolution, self.dc.resolution)

        # removing empty slices
        # empty_slices_indices = np.where(np.any(flair, axis=(1, 2, 3)) == True)[0]
        # flair, t1ce, mask = flair[empty_slices_indices], t1ce[empty_slices_indices], mask[empty_slices_indices]

        # dataset and dataloader
        self.dataset = IdentityDataset(*[volumes.values] + [slice_positions])

        print('Modalities: ', self.dc.modalities)
        print('Data shape: ', volumes[self.dc.modalities[0]].shape)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.dc.batch_size, shuffle=self.dc.shuffle, num_workers=self.dc.num_workers, pin_memory=True
        )


if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()
    cfg = './config/config.yaml'
    config = OmegaConf.load(cfg)
    dm = BratsDataModule(config.data)
    dm.setup()