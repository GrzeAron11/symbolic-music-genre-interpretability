import copy as cp
from pathlib import Path
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split

from wimu_smgi.dataset import MusicGenreDataset


class MusicGenreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        npy_dir: str = "data/processed/npy_arrays",
        csv_path: str = "data/processed/labels/msd-topMAGD.csv",
        batch_size: int = 32,
        seq_len: int = 500,
        num_workers: int = 2,
    ):
        super().__init__()
        self.npy_dir = Path(npy_dir)
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_dataset = MusicGenreDataset(
            npy_dir=self.npy_dir,
            csv_path=self.csv_path,
            seq_len=self.seq_len,
            augment=False
        )
        train_sub, val_sub, test_sub = random_split(full_dataset, [0.8, 0.1, 0.1])
        
        train_sub.dataset = cp.copy(full_dataset)
        train_sub.dataset.augment = True
        
        self.train_data = train_sub
        self.val_data = val_sub
        self.test_data = test_sub


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )