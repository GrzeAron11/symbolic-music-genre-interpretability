# Interpretowalność klasyfikatora gatunku na muzyce symbolicznej

**Członkowie zespołu:**
* Karol Łukasik
* Grzegorz Aronowski
* Adam Czupryński

## Temat
Projekt polega na wytrenowaniu klasyfikatora gatunku na wybranych datasetach MIDI i zastosowaniu metod concept-based interpretability do analizy, które cechy model uznaje za charakterystyczne dla każdego gatunku. Inspiracją jest praca Foscarina et al. (2022), gdzie TCAV zastosowano do klasyfikacji kompozytorów. Projekt przenosi to podejście na gatunki, definiując odpowiednie koncepty muzyczne. Dodatkowym elementem jest porównanie "definicji gatunku" zakodowanych w różnych datasetach oraz analiza próbek błędnie sklasyfikowanych.

## Planowany zakres eksperymentów
* Trenowanie klasyfikatora gatunku na co najmniej dwóch różnych zbiorach muzyki symbolicznej.
* Eksperymenty z TCAV – testowanie wpływu zdefiniowanych konceptów muzycznych na predykcje gatunków wewnątrz jednego modelu oraz sprawdzenie, czy modele wytrenowane na zupełnie innych bazach plików MIDI zwracają uwagę na te same koncepty.
* Analiza błędnie sklasyfikowanych utworów w celu weryfikacji jakie cechy wpłynęły na błędną predykcje.

## Planowana funkcjonalność programu
Program będzie zawierał interfejs wiersza poleceń (CLI). Główne funkcjonalności to:
* **Przetwarzanie danych:** Moduł do wczytywania plików muzyki symbolicznej (MIDI), czyszczenia danych i transformacji ich do reprezentacji wejściowej odpowiedniej dla sieci neuronowej (np. piano roll lub sekwencje tokenów).
* **Trening i ewaluacja:** Możliwość uruchomienia treningu klasyfikatora gatunków muzycznych na wybranych zbiorach danych, z automatycznym logowaniem metryk (loss, accuracy) oraz zapisywaniem najlepszych wag modelu.
* **Ekstrakcja konceptów:** Funkcjonalność pozwalająca na zdefiniowanie i wyodrębnienie zestawów danych reprezentujących określone koncepty muzyczne oraz wytrenowanie klasyfikatorów liniowych.
* **Analiza interpretowalności:** Moduł obliczający i zwracający wyniki TCAV, czyli oceniający, jak silnie dany koncept wpływa na predykcję konkretnego gatunku muzycznego dla danego modelu.
* **Narzędzia analityczne i wizualizacyjne:** Generowanie raportów i wykresów podsumowujących trafność predykcji, znaczenie konceptów oraz zestawienia błędnie sklasyfikowanych utworów.

## Planowany stack technologiczny
Stack został dobrany tak, aby spełniać wymogi dotyczące projektu oraz eksperymentów ML:
* **Język i środowisko:** Python 3.10+, zarządzanie zależnościami i środowiskiem wirtualnym za pomocą Poetry.
* **Struktura projektu:** Szablon Cookiecutter Data Science.
* **Deep Learning:** PyTorch wraz z PyTorch Lightning.
* **Przetwarzanie muzyki symbolicznej:** Biblioteka pretty_midi lub muspy do parsowania, analizy i manipulacji plikami MIDI.
* **Interpretowalność:** Captum (biblioteka od PyTorch oferująca m.in. implementację TCAV) lub autorska implementacja oparta na scikit-learn do tworzenia wektorów CAV.
* **Śledzenie eksperymentów:** Weights & Biases (W&B) do monitorowania treningu klasyfikatorów oraz logowania hiperparametrów i artefaktów.
* **Interfejs CLI:** Typer lub argparse do tworzenia wygodnych skryptów wywołujących poszczególne etapy potoku.
* **Jakość kodu i testy:**
  * Ruff pełniący rolę szybkiego lintera i autoformatera.
  * pytest do testów jednostkowych i integracyjnych.
  * tox do testowania automatycznego w odizolowanych środowiskach.
  * Styl zgodny z PEP8.

## Harmonogram projektu
* **Tydzień 1 (18.03.2026 – 24.03.2026):** Analiza literatury i utworzenie tabeli z własnymi komentarzami i informacjami o przestudiowanych artykułach.
* **Tydzień 2 (25.03.2026 – 31.03.2026):** Pobranie i wstępne przetworzenie datasetów, utworzenie wstępnego modelu klasyfikatora.
* **Tydzień 3 (01.04.2026 – 07.04.2026):** Integracja z W&B, przygotowanie skryptów konfiguracyjnych.
* **Tydzień 4 (08.04.2026 – 14.04.2026):** Trening sieci.
* **Tydzień 5 (15.04.2026 – 21.04.2026):** Implementacje funkcji które będą określać "nasilenie" danego konceptu muzycznego, przygotowanie podzbiorów danych które reprezentują te koncepty (do wykorzystania w TCAV).
* **Tydzień 6 (22.04.2026 – 28.04.2026):** Trenowanie klasyfikatorów liniowych które rozdzielą zbiór losowy od zbioru reprezentującego koncept muzyczny (CAV), ewaluacja tego jak ważne są zdefiniowane wcześniej koncepty względem poszczególnych gatunków.
* **Tydzień 7 (29.04.2026 – 05.05.2026):** Analiza wyników i poszukiwanie błędnych klasyfikacji. Weryfikacja różnic definicji gatunków pomiędzy różnymi zbiorami danych.
* **Tydzień 8 (06.05.2026 – 12.05.2026):** testy integracyjne i środowiskowe (tox), przygotowanie instrukcji użytkowania.
* **Tydzień 9 (13.05.2026 – 19.05.2026):** Ostatnie poprawki, nagranie filmu demonstracyjnego.
* **Tydzień 10 (20.05.2026 – 25.05.2026):** Zebranie i omówienie wyników, oddanie projektu.

## Bibliografia
1. Foscarin et al. (2022), Concept-Based Techniques for "Musicologist-friendly" Explanations in a Deep Music Classifier, https://arxiv.org/abs/2208.12485
2. Dervakos et al. (2022), Genre Recognition from Symbolic Music with CNNs, SN Computer Science, https://link.springer.com/article/10.1007/s42979-022-01490-6
3. Karystinaios et al. (2024), SMUG-Explain: A Framework for Symbolic Music Graph Explanations, https://arxiv.org/abs/2405.09241
