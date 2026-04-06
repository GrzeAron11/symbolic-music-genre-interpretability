from lightning.pytorch.cli import LightningCLI

from wimu_smgi.modeling.datamodule import MusicGenreDataModule
from wimu_smgi.modeling.model import MusicGenreClassifier


def main():
    LightningCLI(MusicGenreClassifier, MusicGenreDataModule)


if __name__ == "__main__":
    main()
