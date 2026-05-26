
from pathlib import Path

from captum.concept import TCAV, Concept
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import torch
from torch.utils.data import DataLoader

from wimu_smgi.modeling.model import MusicGenreClassifier
from wimu_smgi.modeling.tcav_dataset import ConceptDataset


class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield batch.to(self.device)

    def __len__(self):
        return len(self.dataloader)


def run_tcav():
    PROCESSED_DIR = Path("data/processed/npy_arrays")
    CONCEPTS_DIR = Path("data/concepts")
    CSV_PATH = Path("data/processed/labels/msd-topMAGD.csv")

    BATCH_SIZE = 16
    SEQ_LEN = 1024

    ALL_GENRES = [
        "Blues", "Country", "Electronic", "Folk", "International",
        "Jazz", "Latin", "New Age", "Pop_Rock", "Rap", "Reggae", "RnB", "Vocal"
    ]
    GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(ALL_GENRES)}

    logger.info("Ładowanie modelu...")
    model = MusicGenreClassifier.load_from_checkpoint("models/resnet18_prototype-epoch=43-val_loss=1.11.ckpt")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Urządzenie: {device}")

    bottleneck = "layer3"

    logger.info("Skanowanie folderu z konceptami i szumem...")
    concept_dirs = [d for d in CONCEPTS_DIR.iterdir() if d.is_dir() and not d.name.startswith("random")]
    random_dirs = [d for d in CONCEPTS_DIR.iterdir() if d.is_dir() and d.name.startswith("random")]

    if len(random_dirs) < 10:
        logger.warning(f"Znaleziono tylko {len(random_dirs)} folderów szumu. Zalecane 10!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Ładowanie szumu...")
    random_concepts = []
    for i, r_dir in enumerate(random_dirs):
        ds_rand = ConceptDataset(r_dir, seq_len=SEQ_LEN)
        dl_rand = DataLoader(ds_rand, batch_size=BATCH_SIZE, shuffle=True)
        dl_rand_device = DeviceDataLoader(dl_rand, device)
        random_concepts.append(Concept(id=i, name=r_dir.name, data_iter=dl_rand_device))

    logger.info("Ładowanie badanych konceptów...")
    main_concepts = []
    for c_idx, c_dir in enumerate(concept_dirs):
        ds_concept = ConceptDataset(c_dir, seq_len=SEQ_LEN)
        dl_concept = DataLoader(ds_concept, batch_size=BATCH_SIZE, shuffle=True)
        dl_concept_device = DeviceDataLoader(dl_concept, device)
        unikalne_id = abs(hash(c_dir.name)) % 10000
        main_concepts.append(Concept(id=unikalne_id, name=c_dir.name, data_iter=dl_concept_device))

    tcav = TCAV(model=model.model, layers=[bottleneck])

    num_tests = len(ALL_GENRES)
    alpha_bonferroni = 0.05 / num_tests if num_tests > 0 else 0.05

    df = pd.read_csv(CSV_PATH)

    logger.info(f"ROZPOCZĘCIE PROFILOWANIA MODELU (Testów do wykonania: {num_tests})")
    logger.info(f"Rygor statystyczny (Bonferroni p-value threshold): {alpha_bonferroni:.5f}")

    final_report = {c.name: [] for c in main_concepts}

    for genre_name in ALL_GENRES:
        target_idx = GENRE_TO_IDX[genre_name]
        logger.info(f"Analiza gatunku: {genre_name}...")

        target_ids = df[df[genre_name] == 1]['file_id'].tolist()
        target_files = []
        for f in PROCESSED_DIR.glob("*.npz"):
            track_id = f.name.split("__")[0]
            if track_id in target_ids:
                target_files.append(f)
                if len(target_files) >= 32:
                    break

        if len(target_files) < 10:
            logger.warning(f"Pominięto {genre_name}: Zbyt mało plików .npz ({len(target_files)})")
            continue

        ds_target = ConceptDataset(target_files, seq_len=SEQ_LEN)
        dl_target = DataLoader(ds_target, batch_size=32, shuffle=True)
        target_inputs = next(iter(dl_target)).to(device)

        for main_concept in main_concepts:
            c_name = main_concept.name
            experimental_sets = [[main_concept, rand_c] for rand_c in random_concepts]

            tcav_scores = tcav.interpret(
                inputs=target_inputs,
                experimental_sets=experimental_sets,
                target=target_idx
            )

            extracted_scores = []
            for exp_key, layer_dict in tcav_scores.items():
                if bottleneck not in layer_dict:
                    continue
                raw_score = layer_dict[bottleneck]['sign_count']
                score = raw_score[0].item() if isinstance(raw_score, torch.Tensor) else float(raw_score[0])
                extracted_scores.append(score)

            if extracted_scores:
                avg_score = np.mean(extracted_scores)
                std_dev = np.std(extracted_scores)

                if std_dev == 0:
                    p_value = 0.0 if avg_score != 0.5 else 1.0
                else:
                    _, p_value = ttest_1samp(extracted_scores, 0.5)

                is_significant = p_value < alpha_bonferroni

                if is_significant and avg_score > 0.5:
                    status = "POZYTYWNY"
                elif is_significant and avg_score < 0.5:
                    status = "NEGATYWNY"
                else:
                    status = "SZUM"

                final_report[c_name].append({
                    'genre': genre_name,
                    'score': avg_score,
                    'std': std_dev,
                    'p_val': p_value,
                    'status': status
                })

    logger.info("MACIERZ ISTOTNOŚCI KONCEPTÓW (PODSUMOWANIE)")

    for c_name, results in final_report.items():
        logger.info(f"\nKONCEPT: {c_name.upper()}")
        logger.info(f"{'GATUNEK':<15} | {'TCAV SCORE':<12} | {'P-VALUE':<10} | {'STATUS'}")

        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

        for r in results_sorted:
            logger.info(f"{r['genre']:<15} | {r['score']:.3f} (±{r['std']:.2f}) | {r['p_val']:.5f}    | {r['status']}")


if __name__ == "__main__":
    run_tcav()
