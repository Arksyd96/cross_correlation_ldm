import numpy as np
import torch
import pytorch_lightning as pl
from nibabel import load
from nibabel.processing import resample_to_output
from omegaconf import OmegaConf
from tqdm import tqdm
import os

from .models.autoencoder.gaussian_autoencoder import GaussianAutoencoder
from .models.autoencoder.vector_quantized_autoencoder import VQAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = data

    def __len__(self):
        return self.data[-1].__len__()

    def __getitem__(self, index):
        return [d[index] for d in self.data]

class DataModule(pl.LightningDataModule):
    def __init__(self,
        shape=(64, 128, 128),
        n_samples=500,
        modalities=['t1', 't1ce', 't2', 'flair'],
        binarize=True,
        npy_path='../data/brats_3d_dataset.npy',
        save_npy=True,
        root_path='../../common_data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021',
        autoencoder=None,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        use_2d_slices=False,
        use_latents=False,
        depth_as_channels=False,
        **kwargs
    ) -> None:
        super().__init__()
        assert use_2d_slices == False or use_latents == False, 'You can only use 2D slices or latents, not both'  
        if use_latents == True:            
            assert autoencoder is not None, 'You must provide an autoencoder to encode the data'
            self.autoencoder = autoencoder.to(device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_modalities = len(modalities)

        # just for the fast access
        self.save_hyperparameters(ignore=['autoencoder'])

    def prepare_data(self) -> None:
        if not os.path.exists(self.hparams.npy_path):
            print('Loading dataset from NiFTI files...')
            placeholder = np.zeros(shape=(
                self.hparams.n_samples, 
                self.hparams.num_modalities, 
                self.hparams.shape[1], 
                self.hparams.shape[2], 
                self.hparams.shape[0]
            ))

            for idx, instance in enumerate(tqdm(os.listdir(self.hparams.root_path)[: self.hparams.n_samples], position=0, leave=True)):
                # loading models
                volumes = {}
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = load(os.path.join(self.hparams.root_path, instance, instance + f'_{m}.nii.gz'))

                # Compute the scaling factors (output will not be exactly the same as defined in OUTPUT_SHAPE)
                orig_resolution = volumes[self.hparams.modalities[0]].shape
                scale_factor = (orig_resolution[0] / self.hparams.shape[1], # height
                                orig_resolution[1] / self.hparams.shape[2], # width
                                orig_resolution[2] / self.hparams.shape[0]) # depth

                # Resample the image using trilinear interpolation
                # Drop the last extra rows/columns/slices to get the exact desired output size
                for _, m in enumerate(self.hparams.modalities):
                    volumes[m] = resample_to_output(volumes[m], voxel_sizes=scale_factor, order=1).get_fdata()
                    volumes[m] = volumes[m][:self.hparams.resolution, :self.hparams.resolution, :self.hparams.num_slices]

                # binarizing the mask (for simplicity), you can comment out this to keep all labels
                if self.hparams.binarize and 'seg' in self.hparams.modalities:
                    volumes['mask'] = (volumes['mask'] > 0).astype(np.float32)

                # saving models
                for idx_m, m in enumerate(self.hparams.modalities):
                    placeholder[idx, idx_m, :, :] = volumes[m]

            self.data = placeholder

            if self.hparams.save_npy:
                print('Saving dataset as npy file...')    
                # saving the dataset as a npy file
                np.save(self.hparams.npy_path, self.data)
                print('Saved!')

        else:
            print('Loading dataset from npy file...')
            self.data = np.load(self.hparams.npy_path)
            self.data = self.data[:self.hparams.n_samples]

    def normalize(self) -> None:
            # norm [-1, 1]
            norm = lambda data: data * 2 / data.max() - 1
            self.data = torch.from_numpy(self.data)

            for m in range(self.num_modalities):
                for idx in range(self.hparams.n_samples):
                    self.data[idx, m] = norm(self.data[idx, m]).type(torch.float32)

            self.data.clamp(-1, 1)
            self.data = self.data.permute(0, 4, 1, 2, 3) # depth first
    
    def to_2d_slices(self) -> None:
            # if switching to 2D
            D, W, H = self.hparams.shape
            self.data = self.data.reshape(self.hparams.n_samples * D, -1, W, H)

            # keeping track on slice positions for positional embedding
            self.slice_positions = torch.arange(D)[None, :].repeat(self.hparams.n_samples, 1)
            self.slice_positions = self.slice_positions.flatten()

            print('Data shape:', self.data.shape)
            print('Slice positions shape:', self.slice_positions.shape)

    def to_latents(self) -> None:
        self.z_latents = []
        self.autoencoder.eval()

        for idx in tqdm(range(self.hparams.n_samples), position=0, leave=True):
            input = self.data[idx].to(self.device, dtype=torch.float32)

            with torch.no_grad():
                # encode a whole volume at once
                pos = torch.arange(0, 64, device=self.device, dtype=torch.long)
                pemb = self.autoencoder.encode_position(pos)
                if isinstance(self.autoencoder, GaussianAutoencoder):
                    z = self.autoencoder.encode(input, pemb).sample()
                elif isinstance(self.autoencoder, VQAutoencoder):
                    z, _ = self.autoencoder.encode_pre_quantization(input, pemb)
                else:
                    raise NotImplementedError
                self.z_latents.append(z.detach().cpu())

        # turning into ndarray        
        self.z_latents = torch.stack(self.z_latents) # of shape (B, D, C, H, W)
        self.z_latents = self.z_latents.permute(0, 2, 1, 3, 4) # of shape (B, C, D, H, W)

        if self.hparams.depth_as_channels:
            self.z_latents = self.z_latents.reshape(
                self.hparams.n_samples, 
                -1,                         # we combine channels and depth : C * D
                self.hparams.shape[1],      # height
                self.hparams.shape[2]       # width
            )

        # rescaling latents
        self.scale = self.z_latents.std()
        self.z_latents = self.z_latents / self.scale

        print('Scale:', self.scale)
        print('Latents standard deviation:', self.z_latents.std())
        print('Latents shape:', self.z_latents.shape)
        
    def setup(self, stage='fit'):
        self.normalize()
        if self.hparams.use_2d_slices:
            self.to_2d_slices()
            self.dataset = IdentityDataset(self.data, self.slice_positions)
        else:
            self.to_latents()
            self.dataset = IdentityDataset(self.z_latents)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=self.hparams.shuffle, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )
    