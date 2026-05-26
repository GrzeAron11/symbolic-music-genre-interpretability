from wimu_smgi.modeling.datamodule import MusicGenreDataModule


def test_datamodule_setup_creates_splits(dummy_data):
    npy_dir, csv_path = dummy_data

    datamodule = MusicGenreDataModule(
        npy_dir=str(npy_dir),
        csv_path=str(csv_path),
        batch_size=1,
        seq_len=128,
        num_workers=0
    )

    datamodule.setup()

    assert hasattr(datamodule, 'train_data')
    assert hasattr(datamodule, 'val_data')
    assert hasattr(datamodule, 'test_data')
    assert datamodule.train_data is not None
