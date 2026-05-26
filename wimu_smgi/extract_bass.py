from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPT_DIR = Path("data/concepts/bass_heavy")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    BASS_UPPER_LIMIT = 45
    BASS_RATIO_THRESHOLD = 0.40

    CONCEPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Wczytywanie bazy referencyjnej topMAGD...")
    df = pd.read_csv(CSV_PATH)
    valid_ids = set(df['file_id'].tolist())

    logger.info(f"Rozpoczęto analizę rejestrów (próg basu: klawisz MIDI <= {BASS_UPPER_LIMIT}).")
    scores = []

    for f in tqdm(list(PROCESSED_DIR.glob("*.npz"))):
        track_id = f.name.split("__")[0]
        if track_id not in valid_ids:
            continue

        try:
            with np.load(f, allow_pickle=True) as data:
                melody = data['melody'] > 0

            total_notes_per_step = np.sum(melody, axis=1)
            bass_notes_per_step = np.sum(melody[:, :BASS_UPPER_LIMIT+1], axis=1)
            active_steps = total_notes_per_step > 0

            if np.any(active_steps):
                total_bass_notes = np.sum(bass_notes_per_step[active_steps])
                total_all_notes = np.sum(total_notes_per_step[active_steps])

                if total_all_notes > 0:
                    bass_ratio = total_bass_notes / total_all_notes

                    if bass_ratio >= BASS_RATIO_THRESHOLD:
                        scores.append((f, bass_ratio))

        except Exception:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = scores[:100]

    if not top_100:
        logger.warning(f"Nie znaleziono utworów, w których bas przekracza {BASS_RATIO_THRESHOLD*100}% całkowitej liczby nut.")
        return

    logger.info(f"Kopiowanie {len(top_100)} utworów do {CONCEPT_DIR}...")
    for f, score in tqdm(top_100):
        shutil.copy(f, CONCEPT_DIR / f.name)

    avg_top_score = np.mean([s for _, s in top_100])
    logger.info(f"Zakończono. Średni udział basu w wyselekcjonowanej próbie: {avg_top_score*100:.1f}%.")


if __name__ == "__main__":
    main()
