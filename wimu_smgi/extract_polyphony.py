from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPT_DIR = Path("data/concepts/high_polyphony")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    POLYPHONY_THRESHOLD = 12

    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Wczytywanie bazy referencyjnej topMAGD...")
    df = pd.read_csv(CSV_PATH)
    valid_ids = set(df['file_id'].tolist())

    logger.info(f"Rozpoczęto analizę polifonii (próg: {POLYPHONY_THRESHOLD} nut).")
    scores = []

    for f in tqdm(list(PROCESSED_DIR.glob("*.npz"))):
        track_id = f.name.split("__")[0]

        if track_id not in valid_ids:
            continue

        try:
            with np.load(f, allow_pickle=True) as data:
                melody = data['melody']

            notes_per_step = np.sum(melody > 0, axis=1)

            if len(notes_per_step) > 0:
                chord_ratio = np.sum(notes_per_step >= POLYPHONY_THRESHOLD) / len(notes_per_step)
                scores.append((f, chord_ratio))

        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = scores[:100]

    if not top_100:
        logger.warning("Nie znaleziono pasujących utworów. Próg może być zbyt wysoki.")
        return

    logger.info(f"Kopiowanie {len(top_100)} utworów do {CONCEPT_DIR}...")
    for f, score in tqdm(top_100):
        shutil.copy(f, CONCEPT_DIR / f.name)

    avg_top_score = np.mean([s for _, s in top_100])
    logger.info(f"Zakończono. Średni udział akordów w wyselekcjonowanej próbie: {avg_top_score*100:.1f}%.")


if __name__ == "__main__":
    main()
