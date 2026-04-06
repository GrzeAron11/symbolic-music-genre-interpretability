from pathlib import Path
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from wimu_smgi.dataset import MusicGenreDataset


class MusicGenreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        npy_dir: str = "data/processed/npy_arrays",
        csv_path: str = "data/processed/labels/msd-topMAGD.csv",
        batch_size: int = 2,
        seq_len: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        self.npy_dir = Path(npy_dir)
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = MusicGenreDataset(
            npy_dir=self.npy_dir,
            csv_path=self.csv_path,
            seq_len=self.seq_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
