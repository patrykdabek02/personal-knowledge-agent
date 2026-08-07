"""
pokaz_odpowiedzi.py - wypisuje pytania i odpowiedzi z results.csv w czytelnej formie,
zeby mozna je bylo ocenic merytorycznie (kolumna ocena_reczna).

Uzycie:
    python pokaz_odpowiedzi.py                    # tylko pytania w zakresie
    python pokaz_odpowiedzi.py --wszystkie        # razem z pytaniami spoza zakresu
    python pokaz_odpowiedzi.py --zapisz do_oceny.txt

Pytan spoza zakresu nie ocenia sie recznie - tam liczy sie sama odmowa,
a ta jest juz zmierzona automatycznie.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

OUT_OF_SCOPE = "POZA_ZAKRESEM"


def main() -> int:
    ap = argparse.ArgumentParser(description="Podglad odpowiedzi do oceny merytorycznej.")
    ap.add_argument("--plik", default="results.csv")
    ap.add_argument("--wszystkie", action="store_true", help="pokaz tez pytania spoza zakresu")
    ap.add_argument("--zapisz", default=None, help="zapisz output do pliku tekstowego")
    args = ap.parse_args()

    path = Path(args.plik)
    if not path.exists():
        print(f"BLAD: nie ma pliku '{path}'. Uruchom najpierw evaluate.py.")
        return 1

    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    linie: list[str] = []
    numer = 0
    for row in rows:
        expected = (row.get("oczekiwane_zrodlo") or "").strip()
        is_out = expected.upper() == OUT_OF_SCOPE
        if is_out and not args.wszystkie:
            continue

        numer += 1
        pytanie = (row.get("pytanie") or "").strip()
        odpowiedz = (row.get("odpowiedz") or "").strip()
        zrodla = (row.get("zrodla") or "").strip()
        top1 = (row.get("trafienie_top1") or "").strip()
        dyst = (row.get("najlepszy_dystans") or "").strip()
        blad = (row.get("blad") or "").strip()

        linie.append("=" * 78)
        linie.append(f"[{numer}] {pytanie}")
        linie.append(f"    oczekiwane zrodlo : {expected}")
        linie.append(f"    zwrocone zrodla   : {zrodla or '(brak)'}")
        linie.append(f"    top1 trafiony     : {top1 or '-'}   dystans: {dyst or '-'}")
        if blad:
            linie.append(f"    BLAD: {blad}")
        linie.append("")
        linie.append("    ODPOWIEDZ:")
        # zawijanie dla czytelnosci
        tekst = odpowiedz if odpowiedz else "(pusta)"
        slowa = tekst.split()
        buf = "    "
        for w in slowa:
            if len(buf) + len(w) + 1 > 96:
                linie.append(buf)
                buf = "    "
            buf += w + " "
        if buf.strip():
            linie.append(buf)
        linie.append("")

    linie.append("=" * 78)
    linie.append(f"Do oceny: {numer} pytan.")
    linie.append("Dla kazdego wpisz w results.csv w kolumnie 'ocena_reczna':")
    linie.append("  1 = odpowiedz merytorycznie poprawna")
    linie.append("  0 = odpowiedz bledna, niepelna lub wprowadzajaca w blad")

    output = "\n".join(linie)

    if args.zapisz:
        Path(args.zapisz).write_text(output, encoding="utf-8")
        print(f"Zapisano do: {args.zapisz}   ({numer} pytan)")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
