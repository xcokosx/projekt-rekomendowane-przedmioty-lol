# REPORT

## Cel

Przeprowadzić podstawowe testy jakości kodu i uruchomieniowe oraz zebrać wyniki i wnioski dla repozytorium.

## Przeprowadzone testy

- Sprawdzenie lokalnych testów w katalogu `app/tests`.

## Uzyskane wyniki

- Testy jednostkowe: pliki testowe znajdują się w `app/tests`. Aby uruchomić i otrzymać wyniki, uruchom lokalnie `uv run pytest`.
- Błędy i uwagi znalezione podczas inspekcji:
    - Możliwy komunikat "Import 'numpy' could not be resolved" — należy zainstalować zależność w używanym interpreterze.
    - Testy nie obejmują wszystkiego

## Wnioski i rekomendacje

1. Zainstalować zależności: `pip install numpy pandas`.
2. Dodać testy by obejmowały jeszcze większą cześć kodu. Lepiej by priortezować te testy które mają najniższy % objęcia.

---
