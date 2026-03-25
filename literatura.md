# Analiza literatury

| Artykuł | Kod | pre-trenowane modele | Metryki | Zasoby obliczeniowe | Komentarz |
|---|---|---|---|---|---|
| Concept-Based Techniques for "Musicologist-Friendly" Explanations in a Deep Music Classifier, [arxiv](https://arxiv.org/abs/2208.12485) | Dostępny na [github](https://github.com/CPJKU/composer_concept). | Brak pre-trenowanych modeli. | F1, Accuracy, TCAV score, fidelity NTD | Brak informacji | Praca opisuje zastosowanie TCAV do klasyfikacji kompozytorów na danych MIDI (piano roll, ResNet-50). Kod został udostępniony, co pozwala na wykorzystanie pipeline'u i zaadaptowanie go do klasyfikacji gatunku. Dodatkowo praca przedstawia sposób budowania zbiorów konceptowych i liczenia TCAV score, co jest kluczowe dla eksperymentów z interpretowalnością. |