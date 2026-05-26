from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(self, data_source: Union[str, Path, List[Path]], seq_len: int = 1024):
        # Jeśli podano ścieżkę do folderu - wczytaj wszystkie .npz
        if isinstance(data_source, (str, Path)):
            self.files = list(Path(data_source).glob("*.npz"))
        # Jeśli podano gotową listę ścieżek - użyj jej bezpośrednio
        else:
            self.files = data_source

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
        # Normalizacja wartości MIDI (0-127) do (0-1)
        m_final[:actual_len, :] = melody[:actual_len, :128].astype(np.float32) / 127.0
        d_final[:actual_len, :] = drums[:actual_len, :128].astype(np.float32) / 127.0

        # Oczekiwany format przez Twojego ResNeta: (2, 128, seq_len)
        m_tensor = torch.from_numpy(m_final).T
        d_tensor = torch.from_numpy(d_final).T
        return torch.stack([m_tensor, d_tensor], dim=0)
