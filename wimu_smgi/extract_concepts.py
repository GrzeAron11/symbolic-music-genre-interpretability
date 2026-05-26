from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger


def get_four_on_the_floor_score(drums_matrix, min_hits=20, tolerance=2):
    """
    Zwraca wynik (0.0 - 1.0) określający, jak bardzo utwór przypomina idealne Four-on-the-floor.
    Jeśli utwór w ogóle nie ma rytmu na stopie, zwraca 0.0.
    """
    kick_signal = drums_matrix[:, 35] + drums_matrix[:, 36]
    is_active = kick_signal > 0
    onsets = np.where(np.diff(is_active.astype(int)) > 0)[0]

    if len(onsets) < min_hits:
        return 0.0

    intervals = np.diff(onsets)
    valid_intervals = intervals[intervals > 15]

    if len(valid_intervals) < min_hits - 1:
        return 0.0

    median_interval = np.median(valid_intervals)

    if not (30 <= median_interval <= 70):
        return 0.0

    is_on_beat = np.abs(valid_intervals - median_interval) <= tolerance

    on_beat_ratio = np.sum(is_on_beat) / len(valid_intervals)
    return on_beat_ratio


def main():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPT_DIR = Path("data/concepts/four_on_the_floor")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    if CONCEPT_DIR.exists():
        shutil.rmtree(CONCEPT_DIR)
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Wczytywanie bazy dozwolonych utworów...")
    df = pd.read_csv(CSV_PATH)
    valid_ids = set(df['file_id'].tolist())

    logger.info("\nSzukanie Top 100 idealnych uderzeń bębna w zbiorze uczącym...")
    scores = []

    for f in tqdm(list(PROCESSED_DIR.glob("*.npz"))):
        track_id = f.name.split("__")[0]
        if track_id not in valid_ids:
            continue

        try:
            with np.load(f, allow_pickle=True) as data:
                drums = data['drums']

            score = get_four_on_the_floor_score(drums)

            if score > 0:
                scores.append((f, score))

        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = scores[:100]

    if not top_100:
        logger.info("\nNie znaleziono żadnych utworów!")
        return

    logger.info(f"\nKopiowanie do czystego folderu {CONCEPT_DIR}...")
    for f, score in tqdm(top_100):
        shutil.copy(f, CONCEPT_DIR / f.name)

    avg_top_score = np.mean([s for _, s in top_100])
    logger.info(f"\nŚrednia dokładność rytmu w Top 100: {avg_top_score * 100:.1f}%")


if __name__ == "__main__":
    main()
