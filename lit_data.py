import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import InstaFolderDataset, InstaDataset


#PROGRESSIVE_RESIZE_SHAPE = [(96,96), (128,128), (224, 224)]
IMG_SHAPE = (224, 224)

class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./dataset_clean', batch_size=32,
                 num_workers=4, prefetch_factor=16):
        """DataModule for PyTorch Lightning
        Parameters
        ----------
        dataset : InstaImageDataset
        batch_size : int, optional
            By default 32
        num_workers : int, optional
            Number of multi-CPU to fetch data
            By default 8
        prefetch_factor : int, optional
            Number of batches to prefecth, by default 16
        """

        self.prepare_data_per_node = True
        self._log_hyperparams = False
        self.data_dir = data_dir
        self.train_transform = transforms.Compose([
            # transforms.RandAugment(),
            transforms.Resize(IMG_SHAPE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        self.test_transform = transforms.Compose([
            transforms.Resize(IMG_SHAPE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.train_dataset = InstaFolderDataset(
            os.path.join(self.data_dir, 'train'), 
            additional_dir=os.path.join(self.data_dir, 'train_low_conf'),
            transform=self.train_transform, 
            is_train=True)


        self.val_dataset = InstaFolderDataset(
            os.path.join(self.data_dir, 'val'),
            transform=self.test_transform,
            is_train=False)
        # test_data_dir = './dataset_clean'
        self.test_dataset = InstaFolderDataset(
            os.path.join(self.data_dir, 'testx2'),
            transform=self.test_transform,
            is_train=False
        )
        

        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        }

        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_kwargs, shuffle=False)
