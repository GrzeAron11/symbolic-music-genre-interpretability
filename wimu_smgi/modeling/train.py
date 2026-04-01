from pathlib import Path

from loguru import logger
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import typer

from wimu_smgi.config import MODELS_DIR, PROCESSED_DATA_DIR
from wimu_smgi.modeling.model import MusicGenreClassifier
from wimu_smgi.dataset import MusicGenreDataset

app = typer.Typer()

@app.command()
def main(
    epochs: int = typer.Option(10, help="Epoch number"),
    batch_size: int = typer.Option(2, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    seq_len: int = typer.Option(128, help="Sequence length (time steps)"),
    model_path: Path = typer.Option(
        MODELS_DIR / "resnet50_prototype.ckpt", 
        help="Path to save the trained model"
    ),
    npy_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "npy_arrays",
        help="Path to the directory with processed .npy files"
    ),
    csv_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "labels" / "msd-topMAGD.csv",
        help="Path to the one-hot encoded CSV labels file"
    )
):
    logger.info(f"Training: epochs={epochs}, batch size={batch_size}, learning rate={learning_rate}")
    
    # x_dummy = torch.randn(batch_size * 2, 1, 88, 256) #random piano roll for tests
    # y_dummy = torch.randint(0, 10, (batch_size * 2,))
    # dataset = TensorDataset(x_dummy, y_dummy)
    
    logger.info("Initializing MusicGenreDataset...")
    dataset = MusicGenreDataset(
        npy_dir=npy_dir,
        csv_path=csv_path,
        seq_len=seq_len
    )
    num_classes = len(dataset.genre_columns)
    logger.info(f"Detected {num_classes} target classes (genres).")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = MusicGenreClassifier(num_classes=num_classes, learning_rate=learning_rate)
    trainer = pl.Trainer(
        max_epochs=epochs, 
        accelerator="auto", 
        devices=1,
        log_every_n_steps=1,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=1)]
    )

    logger.info("Training model...")
    trainer.fit(model, train_loader)
    logger.info(f"Saving model to {model_path}")
    trainer.save_checkpoint(model_path)
    logger.success("Modeling training complete.")

if __name__ == "__main__":
    app()
