from pathlib import Path
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from loguru import logger

from wimu_smgi.modeling.model import MusicGenreClassifier

app = typer.Typer()


def _load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


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
            m_final[: crop.shape[0], :] = crop
        if drums is not None and drums.size > 0:
            crop = drums[start:end, :128]
            d_final[: crop.shape[0], :] = crop

    m_tensor = torch.from_numpy(m_final).T
    d_tensor = torch.from_numpy(d_final).T
    x = torch.stack([m_tensor, d_tensor], dim=0)
    return x.unsqueeze(0)


@app.command()
def main(
    npy_path: Path = typer.Option(..., "--input", help="Ścieżka do pliku .npy do sprawdzenia"),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Plik konfiguracyjny"),
    checkpoint_path: Path = typer.Option(None, "--model", help="Ścieżka do pliku .ckpt (nadpisuje config)"),
    csv_path: Path = typer.Option(None, "--labels", help="Ścieżka do CSV z gatunkami (nadpisuje config)"),
    threshold: float = typer.Option(None, "--threshold", help="Próg pewności (nadpisuje config)"),
    seq_len: int = typer.Option(None, "--seq-len", help="Długość sekwencji (nadpisuje config)"),
):
    cfg = _load_config(config).get("predict", {})

    checkpoint_path = checkpoint_path or Path(cfg.get("model_path", "models/resnet50_prototype.ckpt"))
    csv_path = csv_path or Path(cfg.get("csv_path", "data/processed/labels/msd-topMAGD.csv"))
    threshold = threshold if threshold is not None else cfg.get("threshold", 0.5)
    seq_len = seq_len if seq_len is not None else cfg.get("seq_len", 128)

    df = pd.read_csv(csv_path)
    genre_names = [c for c in df.columns if c not in ["file_id", "genre_list"]]

    logger.info(f"Loading model from {checkpoint_path}...")
    model = MusicGenreClassifier.load_from_checkpoint(
        checkpoint_path,
        num_classes=len(genre_names)
    )
    model.eval()
    model.freeze()

    x = preprocess_sample(npy_path, seq_len=seq_len)

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


if __name__ == "__main__":
    app()
