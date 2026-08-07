"""
evaluate.py - harness testowy. To jest czesc, ktora odroznia projekt od tutoriala.

Dwa tryby:

1) KALIBRACJA PROGU (szybka, bez LLM - liczy sie sekundy, nie minuty):
       python evaluate.py --calibrate --questions questions.csv
   Przepuszcza pytania przez sam retrieval, zbiera najlepszy dystans dla kazdego
   pytania, a potem przemiata progi 0.20-1.00 i pokazuje, ktory najlepiej oddziela
   pytania w zakresie od pytan spoza zakresu.

2) PELNA EWALUACJA (z generowaniem odpowiedzi):
       python evaluate.py --questions questions.csv --out results.csv
   Liczy: trafnosc zrodla (top-1 i top-k), poprawnosc odmowy dla pytan spoza
   zakresu, czasy. Merytoryczna poprawnosc odpowiedzi oceniasz recznie
   w kolumnie 'ocena_reczna' w wynikowym CSV.

Format pliku wejsciowego (CSV, UTF-8, naglowek obowiazkowy):
    pytanie,oczekiwane_zrodlo,uwagi
Dla pytan spoza zakresu wpisz w 'oczekiwane_zrodlo' doslownie: POZA_ZAKRESEM
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import core

OUT_OF_SCOPE = "POZA_ZAKRESEM"


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            question = (row.get("pytanie") or "").strip()
            if not question:
                continue
            rows.append(
                {
                    "pytanie": question,
                    "oczekiwane_zrodlo": (row.get("oczekiwane_zrodlo") or "").strip(),
                    "uwagi": (row.get("uwagi") or "").strip(),
                }
            )
    return rows


def bez_ogonkow(s: str) -> str:
    """Usuwa polskie znaki diakrytyczne i normalizuje do porownania.

    Konieczne, bo stala REFUSAL jest zapisana ASCII ("Nie znalazlem"), a model
    generuje poprawna polszczyzne ("Nie znalazlem" z l przekreslonym). Bez tej
    normalizacji odmowa wygenerowana przez model byla liczona jako konfabulacja
    - zmierzone 2026-08-07: 6 z 8 poprawnych odmow bledenie zaklasyfikowanych.
    """
    mapa = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return " ".join(s.translate(mapa).lower().split())


def czy_odmowa(odpowiedz: str) -> bool:
    """Czy model odmowil, niezaleznie od pisowni polskich znakow."""
    return bez_ogonkow(odpowiedz).startswith(bez_ogonkow(REFUSAL_PREFIX))


REFUSAL_PREFIX = "Nie znalazlem tego w notatkach"


def source_matches(expected: str, actual: str) -> bool:
    """Dopasowanie tolerancyjne: 'rag.md' zaliczy sie do 'projekty/rag.md'."""
    if not expected:
        return False
    e = expected.strip().lower().replace("\\", "/")
    a = actual.strip().lower().replace("\\", "/")
    return e == a or a.endswith("/" + e) or e.endswith("/" + a)


# --------------------------------------------------------------------------
# Tryb 1: kalibracja progu
# --------------------------------------------------------------------------

def calibrate(rows: list[dict], k: int) -> None:
    in_scope: list[float] = []
    out_scope: list[float] = []

    print(f"Kalibracja na {len(rows)} pytaniach (sam retrieval, bez LLM)...\n")
    for row in rows:
        # max_distance=None -> nic nie odrzucamy, chcemy zobaczyc surowe dystanse
        _, all_hits = core.search(row["pytanie"], k=k, max_distance=None)
        best = all_hits[0].distance if all_hits else 2.0
        is_out = row["oczekiwane_zrodlo"].upper() == OUT_OF_SCOPE
        (out_scope if is_out else in_scope).append(best)
        tag = "POZA " if is_out else "W ZAK"
        print(f"  [{tag}] {best:.3f}  {row['pytanie'][:60]}")

    if not in_scope:
        print("\nBrak pytan w zakresie - nie ma czego kalibrowac.")
        return

    print("\n" + "=" * 62)
    print("ROZKLAD NAJLEPSZEGO DYSTANSU")
    print("=" * 62)
    print(f"  w zakresie   (n={len(in_scope):2d}): "
          f"min {min(in_scope):.3f}  mediana {statistics.median(in_scope):.3f}  max {max(in_scope):.3f}")
    if out_scope:
        print(f"  poza zakresem(n={len(out_scope):2d}): "
              f"min {min(out_scope):.3f}  mediana {statistics.median(out_scope):.3f}  max {max(out_scope):.3f}")
    else:
        print("  poza zakresem: BRAK - dodaj 5 pytan POZA_ZAKRESEM, inaczej progu nie zwalidujesz")
        return

    print("\n" + "=" * 62)
    print("PRZEMIATANIE PROGU")
    print("=" * 62)
    print("  prog   trafne_w_zakresie  poprawne_odmowy  laczna_dokladnosc")

    best_threshold, best_score = None, -1.0
    total = len(in_scope) + len(out_scope)
    for i in range(20, 101, 2):
        threshold = i / 100
        accepted = sum(1 for d in in_scope if d <= threshold)
        refused = sum(1 for d in out_scope if d > threshold)
        score = (accepted + refused) / total
        marker = ""
        if score > best_score:
            best_score, best_threshold, marker = score, threshold, "  <-"
        print(f"  {threshold:.2f}   {accepted:2d}/{len(in_scope):<2d}"
              f"              {refused:2d}/{len(out_scope):<2d}"
              f"             {score:.0%}{marker}")

    print("\n" + "=" * 62)
    print(f"REKOMENDACJA: MAX_DISTANCE = {best_threshold:.2f}  (dokladnosc {best_score:.0%})")
    print("=" * 62)
    print("\nUstaw to w core.py (stala MAX_DISTANCE) albo jako zmienna srodowiskowa:")
    print(f'  PowerShell:  $env:MAX_DISTANCE="{best_threshold:.2f}"')
    if min(out_scope) <= max(in_scope):
        print("\nUWAGA: zakresy sie nakladaja - zaden prog nie rozdzieli wszystkiego czysto.")
        print("       To normalne. Wybierz strone, po ktorej wolisz sie mylic:")
        print("       nizszy prog = czesciej 'nie wiem', wyzszy = czesciej zmyslona odpowiedz.")


# --------------------------------------------------------------------------
# Tryb 2: pelna ewaluacja
# --------------------------------------------------------------------------

def full_eval(rows: list[dict], k: int, max_distance: float, out_path: Path) -> None:
    results: list[dict] = []
    print(f"Pelna ewaluacja: {len(rows)} pytan, k={k}, prog={max_distance}\n")

    for i, row in enumerate(rows, 1):
        question = row["pytanie"]
        expected = row["oczekiwane_zrodlo"]
        is_out = expected.upper() == OUT_OF_SCOPE

        print(f"[{i}/{len(rows)}] {question[:60]}...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            res = core.answer(question, k=k, max_distance=max_distance)
            error = ""
        except Exception as exc:  # noqa: BLE001
            print(f"BLAD: {exc}")
            results.append(
                {
                    "pytanie": question, "oczekiwane_zrodlo": expected, "odpowiedz": "",
                    "zrodla": "", "top1_zrodlo": "", "trafienie_top1": "", "trafienie_topk": "",
                    "odmowa": "", "odmowa_poprawna": "", "najlepszy_dystans": "",
                    "czas_s": round(time.perf_counter() - t0, 2), "blad": str(exc),
                    "ocena_reczna": "",
                }
            )
            continue

        total_s = time.perf_counter() - t0
        sources = res["sources"]
        top1 = sources[0] if sources else ""
        refused = czy_odmowa(res["answer"])
        best_distance = res["hits"][0]["distance"] if res["hits"] else ""

        if is_out:
            hit1 = hitk = ""
            refusal_ok = "TAK" if (refused or not res["in_scope"]) else "NIE"
            verdict = f"odmowa={'ok' if refusal_ok == 'TAK' else 'BLAD - zmyslil'}"
        else:
            hit1 = "TAK" if source_matches(expected, top1) else "NIE"
            hitk = "TAK" if any(source_matches(expected, s) for s in sources) else "NIE"
            refusal_ok = ""
            verdict = f"top1={hit1.lower()} topk={hitk.lower()}"

        print(f"{verdict} ({total_s:.1f}s)")

        results.append(
            {
                "pytanie": question,
                "oczekiwane_zrodlo": expected,
                "odpowiedz": res["answer"].replace("\n", " "),
                "zrodla": "; ".join(sources),
                "top1_zrodlo": top1,
                "trafienie_top1": hit1,
                "trafienie_topk": hitk,
                "odmowa": "TAK" if refused else "NIE",
                "odmowa_poprawna": refusal_ok,
                "najlepszy_dystans": best_distance,
                "czas_s": round(total_s, 2),
                "blad": "",
                "ocena_reczna": "",
            }
        )

    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # --- metryki ---
    scoped = [r for r in results if r["oczekiwane_zrodlo"].upper() != OUT_OF_SCOPE and not r["blad"]]
    out = [r for r in results if r["oczekiwane_zrodlo"].upper() == OUT_OF_SCOPE and not r["blad"]]
    times = [r["czas_s"] for r in results if not r["blad"]]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({n / d:.0%})" if d else "n/d"

    print("\n" + "=" * 62)
    print("WYNIKI")
    print("=" * 62)
    print(f"  pytan w zakresie          : {len(scoped)}")
    print(f"  poprawne zrodlo na 1. miejscu : "
          f"{pct(sum(1 for r in scoped if r['trafienie_top1'] == 'TAK'), len(scoped))}")
    print(f"  poprawne zrodlo w top-{k}      : "
          f"{pct(sum(1 for r in scoped if r['trafienie_topk'] == 'TAK'), len(scoped))}")
    print(f"  falszywe odmowy (w zakresie)  : "
          f"{pct(sum(1 for r in scoped if r['odmowa'] == 'TAK'), len(scoped))}")
    print(f"\n  pytan poza zakresem       : {len(out)}")
    print(f"  poprawnie odmowil             : "
          f"{pct(sum(1 for r in out if r['odmowa_poprawna'] == 'TAK'), len(out))}")
    if times:
        print(f"\n  sredni czas odpowiedzi    : {statistics.mean(times):.2f} s")
        print(f"  mediana                   : {statistics.median(times):.2f} s")
        print(f"  najwolniejsze             : {max(times):.2f} s")
    errors = [r for r in results if r["blad"]]
    if errors:
        print(f"\n  BLEDY: {len(errors)}")

    print(f"\nSzczegoly zapisane do: {out_path}")
    print("Otworz plik i wypelnij kolumne 'ocena_reczna' (1 = merytorycznie poprawna, 0 = nie).")
    print("To jest ta druga metryka - % odpowiedzi merytorycznie poprawnych.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ewaluacja i kalibracja bazy wiedzy.")
    ap.add_argument("--questions", default="questions.csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--k", type=int, default=core.TOP_K)
    ap.add_argument("--max-distance", type=float, default=core.MAX_DISTANCE)
    ap.add_argument("--calibrate", action="store_true", help="tylko kalibracja progu, bez LLM")
    args = ap.parse_args()

    path = Path(args.questions)
    if not path.exists():
        print(f"BLAD: nie ma pliku '{path}'. Skopiuj questions.example.csv i uzupelnij.")
        return 1

    rows = load_questions(path)
    if not rows:
        print("BLAD: plik z pytaniami jest pusty.")
        return 1

    if args.calibrate:
        calibrate(rows, args.k)
    else:
        full_eval(rows, args.k, args.max_distance, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
