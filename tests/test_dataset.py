import torch
from wimu_smgi.dataset import MusicGenreDataset


def test_dataset_loading_and_length(dummy_data):
    npy_dir, csv_path = dummy_data
    dataset = MusicGenreDataset(npy_dir=npy_dir, csv_path=csv_path, seq_len=128)

    assert len(dataset) == 2
    assert len(dataset.genre_columns) == 2


def test_dataset_tensor_shapes(dummy_data):
    npy_dir, csv_path = dummy_data
    seq_len = 256

    dataset = MusicGenreDataset(npy_dir=npy_dir, csv_path=csv_path, seq_len=seq_len)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (2, 128, seq_len)
    assert y.shape == (2,)


def test_dataset_augmentation_shift(dummy_data):
    npy_dir, csv_path = dummy_data
    dataset = MusicGenreDataset(npy_dir=npy_dir, csv_path=csv_path, seq_len=128, augment=True)
    x, _ = dataset[0]

    assert x.shape == (2, 128, 128)
