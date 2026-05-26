import subprocess
import sys
from loguru import logger


def run_step(command: str, description: str):
    logger.info(f"Rozpoczęcie: {description}...")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        logger.error(f"Błąd krytyczny podczas: {description}. Przerywam potok.")
        sys.exit(1)

    logger.success(f"Zakończono: {description}\n")


def main():

    run_step("poetry run python -m wimu_smgi.generate_noise", "Generowanie folderów szumu")

    run_step("poetry run python -m wimu_smgi.extract_concepts", "Ekstrakcja: Four-on-the-floor")
    run_step("poetry run python -m wimu_smgi.extract_polyphony", "Ekstrakcja: Polifonia")
    run_step("poetry run python -m wimu_smgi.extract_power_chords", "Ekstrakcja: Kwinty")
    run_step("poetry run python -m wimu_smgi.extract_tension_ratio", "Ekstrakcja: Napięcie Harmoniczne")
    run_step("poetry run python -m wimu_smgi.extract_bass", "Ekstrakcja: Dominacja basu")

    logger.info("Przejście do analizy interpretowalności sieci neuronowej")
    run_step("poetry run python -m wimu_smgi.modeling.run_tcav", "Analiza TCAV")

    logger.info("=" * 60)
    logger.success("Wszystkie eksperymenty zakończyły się sukcesem.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
