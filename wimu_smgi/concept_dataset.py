from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(self, folder_path: str, seq_len: int = 1024):
        self.files = list(Path(folder_path).glob("*.npz"))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            melody = data['melody']
            drums = data['drums']

        m_final = np.zeros((self.seq_len, 128), dtype=np.float32)
        d_final = np.zeros((self.seq_len, 128), dtype=np.float32)

        actual_len = min(melody.shape[0], self.seq_len)
        m_final[:actual_len, :] = melody[:actual_len, :128].astype(np.float32) / 127.0
        d_final[:actual_len, :] = drums[:actual_len, :128].astype(np.float32) / 127.0

        # Oczekiwany format: (2, 128, seq_len)
        m_tensor = torch.from_numpy(m_final).T
        d_tensor = torch.from_numpy(d_final).T
        return torch.stack([m_tensor, d_tensor], dim=0)
