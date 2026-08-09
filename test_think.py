"""
test_think.py - sprawdza, czy model przyjmuje POZIOMY rozumowania, nie tylko true/false.

Ollama dokumentuje w polu "think" wartosci: true, false oraz low / medium / high / max.
Dokumentacja mowi jednak "wiekszosc modeli", wiec dla konkretnego modelu trzeba to
sprawdzic empirycznie. Skrypt mierzy czas i pokazuje odpowiedz dla kazdego poziomu.

Pytanie testowe jest celowo takie, ktore BEZ rozumowania modele czesto przekrecaja -
liczenie liter w slowie. Dzieki temu widac nie tylko czas, ale i czy rozumowanie
w ogole cos daje.

Uzycie:
    python test_think.py
"""

from __future__ import annotations

import sys
import time

import core

PYTANIE = "Ile liter 'r' jest w slowie truskawka? Odpowiedz samym zdaniem z liczba."
POZIOMY = [False, "low", "medium", "high", "max", True]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    print(f"Model: {core.CHAT_MODEL}")
    print(f"Domyslny THINK z konfiguracji: {core.THINK!r}\n")
    print(f"  {'poziom':8s} {'czas':>7s}  {'num_predict':>11s}  odpowiedz")
    print("  " + "-" * 76)

    for poziom in POZIOMY:
        wartosc = core._think_wartosc(poziom)
        limit = core._chat_payload([], 0.1, wartosc)["options"]["num_predict"]

        t0 = time.perf_counter()
        try:
            odp = core.chat(
                [{"role": "user", "content": PYTANIE}],
                temperature=0.0,
                think=poziom,
            )
            czas = time.perf_counter() - t0
            tekst = " ".join(odp.split())[:52]
            print(f"  {str(poziom):8s} {czas:6.1f}s  {limit:11d}  {tekst}")
        except Exception as exc:  # noqa: BLE001
            czas = time.perf_counter() - t0
            print(f"  {str(poziom):8s} {czas:6.1f}s  {limit:11d}  BLAD: {str(exc)[:44]}")

    print()
    if core._think_supported is False:
        print("  Ten model NIE przyjmuje parametru 'think' - kod cofnal sie do domyslnych")
        print("  ustawien modelu. Poziomy rozumowania sa u Ciebie niedostepne.")
    else:
        print("  Parametr 'think' jest przyjmowany.")
        print("  Jesli czasy dla low/medium/high sa ROZNE - poziomy dzialaja.")
        print("  Jesli wszystkie sa takie same jak przy True - model traktuje je")
        print("  jak zwykle wlaczenie i suwak nic nie daje.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
