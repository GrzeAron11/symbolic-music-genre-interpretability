from lightning.pytorch.cli import LightningCLI
from dotenv import load_dotenv

from wimu_smgi.modeling.datamodule import MusicGenreDataModule
from wimu_smgi.modeling.model import MusicGenreClassifier


def main():
    load_dotenv()
    LightningCLI(MusicGenreClassifier, MusicGenreDataModule)


if __name__ == "__main__":
    main()
