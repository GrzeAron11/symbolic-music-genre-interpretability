import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def dummy_data(tmp_path):
    csv_path = tmp_path / "dummy_labels.csv"
    df = pd.DataFrame({
        "file_id": ["track_1", "track_2"],
        "Blues": [1, 0],
        "Electronic": [0, 1]
    })
    df.to_csv(csv_path, index=False)

    npy_dir = tmp_path / "npy_arrays"
    npy_dir.mkdir()

    for track_id in ["track_1", "track_2"]:
        dummy_npz_path = npy_dir / f"{track_id}__hash.npz"
        dummy_melody = np.random.randint(0, 127, size=(500, 128), dtype=np.uint8)
        dummy_drums = np.random.randint(0, 127, size=(500, 128), dtype=np.uint8)
        np.savez_compressed(dummy_npz_path, melody=dummy_melody, drums=dummy_drums)

    return npy_dir, csv_path
