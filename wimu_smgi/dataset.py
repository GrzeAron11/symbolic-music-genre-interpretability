from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MusicGenreDataset(Dataset):
    def __init__(self, npy_dir: Path, csv_path: Path, seq_len: int = 128, augment: bool = False):
        self.npy_dir = Path(npy_dir)
        self.seq_len = seq_len
        self.augment = augment

        logger.info(f"Loading labels from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.file_names = self.df['file_id'].values
        labels_df = self.df.drop(columns=['file_id', 'genre_list'], errors='ignore')
        self.genre_columns = labels_df.columns.tolist()

        # do prototypu dla niepełnego zbioru danych
        self.df.set_index('file_id', inplace=True)
        self.valid_paths = []
        self.labels = []
        all_npz_files = list(self.npy_dir.glob("*.npz"))
        for npz_path in all_npz_files:
            track_id = npz_path.stem.split("__")[0]
            # Jeśli ten utwór znajduje się w naszym CSV, dodajemy go do Datasetu
            if track_id in self.df.index:
                self.valid_paths.append(npz_path)
                label_vector = self.df.loc[track_id, self.genre_columns].values.astype(np.float32)
                self.labels.append(label_vector)


#        self.labels = labels_df.values.astype(np.float32)
        logger.info(f"Dataset initialized. Samples: {len(self.valid_paths)}, Genres: {len(self.genre_columns)}")

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        file_path = self.valid_paths[idx]

        m_final = np.zeros((self.seq_len, 128), dtype=np.float32)
        d_final = np.zeros((self.seq_len, 128), dtype=np.float32)

        try:
            with np.load(file_path) as data:
                melody = data['melody']
                drums = data['drums']

            has_m = melody is not None and melody.size > 0
            has_d = drums is not None and drums.size > 0

            actual_len = 0
            if has_m:
                actual_len = melody.shape[0]
            elif has_d:
                actual_len = drums.shape[0]

            if actual_len > 0:
                if actual_len > self.seq_len:
                    if self.augment:
                        # losowanie wycinka do treningu
                        start = np.random.randint(0, actual_len - self.seq_len)
                    else:
                        # walidacja cały czas na tym samym wycinku ze środka utworu
                        start = (actual_len - self.seq_len) // 2
                    end = start + self.seq_len
                    if has_m and melody.shape[0] >= end:
                        m_final[:] = melody[start:end, :128].astype(np.float32) / 127.0

                    if has_d and drums.shape[0] >= end:
                        d_final[:] = drums[start:end, :128].astype(np.float32) / 127.0
                else:
                    if has_m:
                        m_final[:actual_len, :] = melody[:actual_len, :128].astype(np.float32) / 127.0
                    if has_d:
                        d_final[:actual_len, :] = drums[:actual_len, :128].astype(np.float32) / 127.0

            if self.augment:
                # transpozycja od -6 do +6 poltonow (TYLKO dla melodii, perkusyjnej nie można bo do nut są przypisane odpowiednie bębny/talerze)
                shift = np.random.randint(-6, 7)

                if shift > 0:
                    m_final_shifted = np.zeros_like(m_final)
                    m_final_shifted[:, shift:] = m_final[:, :-shift]
                    m_final = m_final_shifted

                elif shift < 0:
                    m_final_shifted = np.zeros_like(m_final)
                    m_final_shifted[:, :shift] = m_final[:, -shift:]
                    m_final = m_final_shifted

            m_tensor = torch.from_numpy(m_final).T
            d_tensor = torch.from_numpy(d_final).T

            x = torch.stack([m_tensor, d_tensor], dim=0)
            y = torch.tensor(self.labels[idx], dtype=torch.float32)

            # print(f"DEBUG: {file_path.name} | Active notes in batch: {torch.count_nonzero(x)}")

            return x, y

        except Exception as e:
            logger.error(f"Critical error in {file_path.name}: {e}")
            return torch.zeros((2, 128, self.seq_len)), torch.tensor(self.labels[idx], dtype=torch.float32)
