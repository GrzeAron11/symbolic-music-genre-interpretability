# Interpretowalność klasyfikatora gatunku na muzyce symbolicznej

## Członkowie zespołu
- Karol Łukasik  
- Grzegorz Aronowski  
- Adam Czupryński  

---

## Temat
Projekt polega na wytrenowaniu klasyfikatora gatunku na wybranych datasetach MIDI i zastosowaniu metod **concept-based interpretability** do analizy, które cechy model uznaje za charakterystyczne dla każdego gatunku. Inspiracją jest praca Foscarina et al. (2022), gdzie TCAV zastosowano do klasyfikacji kompozytorów. Projekt przenosi to podejście na gatunki, definiując odpowiednie koncepty muzyczne.  

Dodatkowym elementem jest porównanie „definicji gatunku” zakodowanych w różnych datasetach oraz analiza próbek błędnie sklasyfikowanych.

---

## Planowany zakres eksperymentów
- Trenowanie klasyfikatora gatunku na co najmniej dwóch różnych zbiorach muzyki symbolicznej.  
- Eksperymenty z TCAV:
  - testowanie wpływu zdefiniowanych konceptów muzycznych na predykcje gatunków w obrębie jednego modelu,  
  - sprawdzenie, czy modele wytrenowane na różnych bazach MIDI zwracają uwagę na te same koncepty.  
- Analiza błędnie sklasyfikowanych utworów w celu weryfikacji, jakie cechy wpłynęły na błędną predykcję.  

---

## Planowana funkcjonalność programu

Program będzie zawierał interfejs wiersza poleceń (CLI). Główne funkcjonalności:

### Przetwarzanie danych
- Wczytywanie plików MIDI  
- Czyszczenie danych  
- Transformacja do reprezentacji wejściowej:
  - piano roll  
  - sekwencje tokenów  

### Trening i ewaluacja
- Uruchamianie treningu klasyfikatora  
- Automatyczne logowanie metryk:
  - loss  
  - accuracy  
- Zapisywanie najlepszych wag modelu  

### Ekstrakcja konceptów
- Definiowanie konceptów muzycznych  
- Tworzenie zbiorów reprezentujących koncepty  
- Trenowanie klasyfikatorów liniowych  

### Analiza interpretowalności
- Obliczanie TCAV  
- Ocena wpływu konceptów na predykcje gatunków  

### Narzędzia analityczne i wizualizacyjne
- Generowanie raportów  
- Wykresy:
  - trafność predykcji  
  - znaczenie konceptów  
- Analiza błędnych klasyfikacji  

---

## Planowany stack technologiczny

### Język i środowisko
- Python 3.10+  
- Poetry (zarządzanie zależnościami)  

### Struktura projektu
- Cookiecutter Data Science  

### Deep Learning
- PyTorch  
- PyTorch Lightning  

### Przetwarzanie muzyki symbolicznej
- pretty_midi lub muspy  

### Interpretowalność
- Captum (TCAV)  
- Alternatywnie własna implementacja (scikit-learn + CAV)  

### Śledzenie eksperymentów
- Weights & Biases (W&B)  

### Interfejs CLI
- Typer lub argparse  

### Jakość kodu i testy
- Ruff (lint + formatowanie)  
- pytest (testy)  
- tox (testy środowiskowe)  
- Styl zgodny z PEP8  

---

## Harmonogram projektu

### Tydzień 1 (18.03.2026 – 24.03.2026)
- Analiza literatury  
- Utworzenie tabeli z komentarzami do artykułów  

### Tydzień 2 (25.03.2026 – 31.03.2026)
- Pobranie datasetów  
- Wstępne przetwarzanie danych  
- Utworzenie wstępnego modelu  

### Tydzień 3 (01.04.2026 – 07.04.2026)
- Integracja z W&B  
- Przygotowanie skryptów konfiguracyjnych  

### Tydzień 4 (08.04.2026 – 14.04.2026)
- Trening sieci  

### Tydzień 5 (15.04.2026 – 21.04.2026)
- Implementacja funkcji określających „nasilenie” konceptów  
- Przygotowanie zbiorów konceptów (TCAV)  

### Tydzień 6 (22.04.2026 – 28.04.2026)
- Trenowanie klasyfikatorów liniowych (CAV)  
- Ewaluacja wpływu konceptów  

### Tydzień 7 (29.04.2026 – 05.05.2026)
- Analiza wyników  
- Analiza błędnych klasyfikacji  
- Porównanie datasetów  

### Tydzień 8 (06.05.2026 – 12.05.2026)
- Testy integracyjne (tox)  
- Przygotowanie instrukcji użytkowania  

### Tydzień 9 (13.05.2026 – 19.05.2026)
- Poprawki końcowe  
- Nagranie filmu demonstracyjnego  

### Tydzień 10 (20.05.2026 – 25.05.2026)
- Zebranie wyników  
- Oddanie projektu  

---

## Bibliografia
- Foscarin et al. (2022)  
  *Concept-Based Techniques for "Musicologist-friendly" Explanations in a Deep Music Classifier*  
  https://arxiv.org/abs/2208.12485  

- Dervakos et al. (2022)  
  *Genre Recognition from Symbolic Music with CNNs*  
  https://link.springer.com/article/10.1007/s42979-022-01490-6  

- Karystinaios et al. (2024)  
  *SMUG-Explain: A Framework for Symbolic Music Graph Explanations*  
  https://arxiv.org/abs/2405.09241  
