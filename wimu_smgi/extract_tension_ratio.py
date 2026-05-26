from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_tension_ratio(melody_matrix):
    """
    Zwraca stosunek czasu (ilość klatek), w którym w utworze obecne jest 
    napięcie harmoniczne (dowolne wystąpienie interwału 6 półtonów, 
    również wewnątrz akordów dominantowych V7).
    """
    INTERVAL = 6

    # Przesunięcie macierzy o 6 klawiszy i logiczny AND.
    # Wychwytuje trwanie stanu, w którym klawisz N i N+6 są jednocześnie wciśnięte.
    has_tension = melody_matrix[:, :-INTERVAL] & melody_matrix[:, INTERVAL:]
    steps_with_tension = np.sum(has_tension, axis=1) > 0

    if len(steps_with_tension) == 0:
        return 0.0

    return np.sum(steps_with_tension) / len(steps_with_tension)


def main():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPT_DIR = Path("data/concepts/harmonic_tension")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    if CONCEPT_DIR.exists():
        shutil.rmtree(CONCEPT_DIR)
    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Wczytywanie topMAGD...")
    df = pd.read_csv(CSV_PATH)
    valid_ids = set(df['file_id'].tolist())

    logger.info("Rozpoczęto analizę czasu trwania Napięcia Harmonicznego (Tension)...")
    scores = []

    for f in tqdm(list(PROCESSED_DIR.glob("*.npz"))):
        track_id = f.name.split("__")[0]
        if track_id not in valid_ids:
            continue

        try:
            with np.load(f, allow_pickle=True) as data:
                melody = data['melody'] > 0

            ratio = get_tension_ratio(melody)

            if ratio > 0:
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
    logger.info(f"Zakończono. Średni czas trwania napięcia harmonicznego w Top 100: {avg_top_score*100:.1f}%.")


if __name__ == "__main__":
    main()
