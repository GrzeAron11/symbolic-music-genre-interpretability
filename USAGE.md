## Instrukcja użytkowania

Poniższa instrukcja opisuje kroki niezbędne do przygotowania środowiska, uruchomienia testów jednostkowych oraz ekstrakcji konceptów i analizy TCAV.

### 1. Wymagania wstępne
* **Python:** wersja 3.11 (lub nowsza, do 3.13)
* **Menedżer pakietów:** Poetry

### 2. Instalacja środowiska
Projekt wykorzystuje narzędzie Poetry do ścisłego zarządzania zależnościami. Aby zainstalować wszystkie wymagane biblioteki (w tym zoptymalizowaną wersję PyTorch) w wyizolowanym środowisku wirtualnym, otwórz terminal w głównym katalogu projektu i wykonaj polecenie:

```bash
poetry install
```

Następnie aktywuj utworzone środowisko wirtualne poleceniem:

```bash
poetry shell
```

### 3. Przetwarzanie Danych
Zanim uruchomisz eksperymenty, upewnij się, że surowe dane znajdują się w folderach `data/raw/labels/` oraz `data/raw/lmd_matched/`.

**Przetwarzanie etykiet do formatu One-Hot CSV:**
```bash
poetry run python -m wimu_smgi.prepare_labels labels_raw.cls --out msd-topMAGD.csv
```

**Przetwarzanie surowych plików MIDI (Piano Roll):**
```bash
poetry run python -m wimu_smgi.process_midi --sample-rate 100 --workers 4
```

### 4. Weryfikacja instalacji (Testy jednostkowe)
Przed uruchomieniem eksperymentów należy zweryfikować poprawność działania potoku danych. Aby uruchomić testy, wpisz:

```bash
poetry run pytest tests/
```
### 5. Trening i przygotowanie wag modelu
Wagi modelu (pliki o rozszerzeniu `.ckpt`) są automatycznie generowane i zapisywane w katalogu `models/` podczas procesu uczenia. Aby rozpocząć trening sieci klasyfikatora od zera, wykonaj polecenie:

```bash
poetry run python -m wimu_smgi.modeling.train fit --config configs/train.yaml
```
### 6. Ewaluacja wytrenowanego modelu (Testowanie)
Po zakończeniu treningu, możesz sprawdzić skuteczność klasyfikatora na testowym zbiorze danych (wyliczenie metryk takich jak Accuracy, Precision, Recall czy F1-Score). Aby to zrobić, użyj trybu test i wskaż ścieżkę do pliku z wagami (podmień nazwę pliku .ckpt na właściwą dla Twojego modelu):

```bash
poetry run python -m wimu_smgi.modeling.train test --config configs/train.yaml --ckpt_path models/PLIK_CKPT_WYTRENOWANEGO_MODELU.ckpt
```

### 7. Uruchomienie potoku eksperymentalnego (TCAV)
Projekt udostępnia zautomatyzowany skrypt uruchamiający pełen potok badawczy. Wykonuje on automatycznie generowanie referencyjnych zbiorów szumu, ekstrakcję zdefiniowanych konceptów muzycznych (m.in. polifonia, power chords, bas) oraz analizę statystyczną TCAV.

Aby uruchomić pełen potok interpretowalności, wykonaj:

```bash
poetry run python run_pipeline.py
```