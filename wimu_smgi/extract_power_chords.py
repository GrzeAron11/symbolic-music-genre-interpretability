from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPT_DIR = Path("data/concepts/power_chords_fifth")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    INTERVAL = 7

    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Wczytywanie bazy referencyjnej topMAGD...")
    df = pd.read_csv(CSV_PATH)
    valid_ids = set(df['file_id'].tolist())

    logger.info(f"Rozpoczęto analizę interwałów (przesunięcie: {INTERVAL} półtonów).")
    scores = []

    for f in tqdm(list(PROCESSED_DIR.glob("*.npz"))):
        track_id = f.name.split("__")[0]
        if track_id not in valid_ids:
            continue

        try:
            with np.load(f, allow_pickle=True) as data:
                melody = data['melody'] > 0

            has_interval = melody[:, :-INTERVAL] & melody[:, INTERVAL:]
            steps_with_interval = np.sum(has_interval, axis=1) > 0

            if len(steps_with_interval) > 0:
                ratio = np.sum(steps_with_interval) / len(steps_with_interval)
                scores.append((f, ratio))

        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = scores[:100]

    if not top_100:
        logger.warning("Nie znaleziono pasujących utworów.")
        return

    logger.info(f"Kopiowanie {len(top_100)} utworów do {CONCEPT_DIR}...")
    for f, score in tqdm(top_100):
        shutil.copy(f, CONCEPT_DIR / f.name)

    avg_top_score = np.mean([s for _, s in top_100])
    logger.info(f"Zakończono. Średni udział badanego interwału: {avg_top_score*100:.1f}%.")


if __name__ == "__main__":
    main()
