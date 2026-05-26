from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from wimu_smgi.modeling.datamodule import MusicGenreDataModule
from wimu_smgi.modeling.model import MusicGenreClassifier


def main():
    load_dotenv()
    LightningCLI(MusicGenreClassifier, MusicGenreDataModule, save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
