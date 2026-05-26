from pathlib import Path
import random
import shutil

from loguru import logger
from tqdm import tqdm


def generate_random_concepts(num_folders=10, samples_per_folder=100):
    source_dir = Path("data/processed/npy_arrays")
    concepts_dir = Path("data/concepts")
    concepts_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        logger.error(f"Katalog {source_dir} nie istnieje!")
        return

    all_files = list(source_dir.glob("*.npz"))
    logger.info(f"Znaleziono {len(all_files)} plików bazowych.")

    used_files = set()
    for concept_folder in concepts_dir.iterdir():
        if concept_folder.is_dir() and not concept_folder.name.startswith("random"):
            for f in concept_folder.glob("*.npz"):
                used_files.add(f.name)

    available_files = [f for f in all_files if f.name not in used_files]
    logger.info(f"Dostępnych plików do losowania szumu: {len(available_files)}")

    if len(available_files) < num_folders * samples_per_folder:
        logger.error("Masz za mało wygenerowanych plików, aby stworzyć zbiory szumu!")
        return

    random.seed(42)
    random.shuffle(available_files)

    for i in range(1, num_folders + 1):
        random_dir = concepts_dir / f"random_{i}"

        if random_dir.exists():
            shutil.rmtree(random_dir)
        random_dir.mkdir(parents=True)

        start_idx = (i - 1) * samples_per_folder
        end_idx = start_idx + samples_per_folder

        logger.info(f"Tworzenie {random_dir.name}...")
        for f in tqdm(available_files[start_idx:end_idx], desc=f"Kopiowanie do random_{i}"):
            shutil.copy(f, random_dir / f.name)

    logger.success(f"Pomyślnie wygenerowano {num_folders} folderów szumu w {concepts_dir}!")


if __name__ == "__main__":
    generate_random_concepts()
