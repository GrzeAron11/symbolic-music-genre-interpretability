from pathlib import Path
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from wimu_smgi.config import MODELS_DIR, PROCESSED_DATA_DIR
from wimu_smgi.modeling.model import MusicGenreClassifier

app = typer.Typer()

def preprocess_sample(npy_path: Path, seq_len: int = 128):
    """Przetwarza pojedynczy plik .npy tak samo jak Dataset."""
    data = np.load(npy_path, allow_pickle=True).item()
    melody = data.get('melody')
    drums = data.get('drums')

    actual_len = melody.shape[0] if melody is not None and melody.size > 0 else 0
    if actual_len == 0 and drums is not None:
        actual_len = drums.shape[0]

    m_final = np.zeros((seq_len, 128), dtype=np.float32)
    d_final = np.zeros((seq_len, 128), dtype=np.float32)

    if actual_len > 0:
        start = max(0, (actual_len // 2) - (seq_len // 2))
        end = start + seq_len
        
        if melody is not None and melody.size > 0:
            crop = melody[start:end, :128]
            m_final[:crop.shape[0], :] = crop
        if drums is not None and drums.size > 0:
            crop = drums[start:end, :128]
            d_final[:crop.shape[0], :] = crop

    m_tensor = torch.from_numpy(m_final).T
    d_tensor = torch.from_numpy(d_final).T
    x = torch.stack([m_tensor, d_tensor], dim=0)
    return x.unsqueeze(0)

@app.command()

def main(
    checkpoint_path: Path = typer.Option(..., "--model", help="Ścieżka do pliku .ckpt"),
    npy_path: Path = typer.Option(..., "--input", help="Ścieżka do pliku .npy do sprawdzenia"),
    csv_path: Path = typer.Option(..., "--labels", help="Ścieżka do msd-topMAGD.csv (żeby znać nazwy gatunków)"),
    threshold: float = typer.Option(0.5, help="Próg pewności dla gatunku")
):
    df = pd.read_csv(csv_path)
    genre_names = [c for c in df.columns if c not in ['file_id', 'genre_list']]
    num_classes = len(genre_names)


    logger.info(f"Loading model from {checkpoint_path}...")
    model = MusicGenreClassifier.load_from_checkpoint(
        checkpoint_path, 
        num_classes=num_classes
    )
    model.eval() 
    model.freeze()

    x = preprocess_sample(npy_path)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze()

    logger.success(f"Results for {npy_path.name}:")
    results = []
    for i, prob in enumerate(probs):
        if prob > threshold:
            results.append((genre_names[i], prob.item()))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)

    if not results:
        logger.warning("No genres detected with high enough confidence.")
        top_idx = torch.argmax(probs)
        logger.info(f"Top candidate was: {genre_names[top_idx]} ({probs[top_idx]:.4f})")
    else:
        for name, p in results:
            print(f" > {name}: {p:.2%}")




    # features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
    # predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",


if __name__ == "__main__":
    app()
