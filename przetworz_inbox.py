"""
przetworz_inbox.py - mielenie surowych zapiskow z inboxu na propozycje notatek.

Problem, ktory to rozwiazuje: przycisk "Dopisz do notatek" wrzuca tekst do
inbox/surowe-RRRR-MM-DD.md i od razu indeksuje go w takiej postaci, w jakiej
zostal napisany. Wpisy pisane w biegu sa surowe, powtarzaja tresc istniejacych
notatek i z czasem zaczynaja konkurowac z nimi o miejsce w top-6. Inbox mial byc
skrzynka wejsciowa, a staje sie druga baza wiedzy.

Ten skrypt dla kazdego wpisu proponuje: do KTOREJ notatki go dopisac, do ktorej
sekcji i w jakim brzmieniu. Nic nie zapisuje do vaultu - wypluwa propozycje do
przegladu.

Nowosc wobec messenger.py: plik docelowy nie jest zgadywany przez model, tylko
WYSZUKIWANY w istniejacej bazie. Ten sam retrieval, ktory odpowiada na pytania,
sluzy tu do znalezienia notatki tematycznie najblizszej zapiskowi. Model wybiera
sposrod realnie istniejacych plikow, wiec nie moze wymyslic sciezki.

Uzycie:
    python przetworz_inbox.py --statystyki
    python przetworz_inbox.py --przetworz --limit 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import core
import pamiec

PROJEKT = Path(__file__).resolve().parent
MANIFEST = PROJEKT / "inbox_manifest.json"
PROPOZYCJE = PROJEKT / "propozycje-inbox.md"

# Wpisy krotsze niz to sa zwykle jednym zdaniem, ktore i tak trzeba przeczytac
# samemu - wywolywanie dla nich modelu nic nie wnosi.
MIN_ZNAKOW = 60


def wpisy_inboxu() -> list[dict]:
    """Rozbija pliki surowe-*.md na pojedyncze wpisy.

    Format tworzony przez pamiec.dopisz():
        # Surowe zapiski RRRR-MM-DD
        ## HH:MM tytul
        tresc
    """
    if not pamiec.INBOX.exists():
        return []

    out: list[dict] = []
    for plik in sorted(pamiec.INBOX.glob("surowe-*.md")):
        tekst = plik.read_text(encoding="utf-8-sig")
        data = plik.stem.replace("surowe-", "")
        # dzielimy po naglowkach drugiego poziomu, zachowujac je
        czesci = re.split(r"^## ", tekst, flags=re.MULTILINE)[1:]
        for c in czesci:
            linie = c.split("\n", 1)
            naglowek = linie[0].strip()
            tresc = (linie[1] if len(linie) > 1 else "").strip()
            # odetnij cytat wprowadzajacy, jesli wpis jest pierwszy w pliku
            tresc = re.sub(r"^>.*$", "", tresc, flags=re.MULTILINE).strip()
            if len(tresc) < MIN_ZNAKOW:
                continue
            out.append(
                {
                    "plik": plik.name,
                    "data": data,
                    "naglowek": naglowek,
                    "tresc": tresc,
                    "hash": hashlib.sha256(tresc.encode("utf-8")).hexdigest()[:16],
                }
            )
    return out


def wszystkie_notatki() -> list[str]:
    """Pelna lista plikow w bazie, posortowana.

    Model dostawal wczesniej tylko pieciu kandydatow z wyszukiwania i nie widzial
    reszty vaultu - wiec przy fakcie o negocjacjach tworzyl NOWY plik, nie wiedzac,
    ze `decyzje/wynagrodzenie-widelki.md` juz istnieje. Przy 26 notatkach cala
    lista miesci sie w prompcie bez trudu i kosztuje mniej niz jeden zly plik.
    """
    try:
        r = core.get_collection(create=False).get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        return []
    sciezki = set()
    for m in r.get("metadatas") or []:
        rel = (m or {}).get("rel_path", "")
        if rel and not rel.replace("\\", "/").startswith("inbox/"):
            sciezki.add(rel)
    return sorted(sciezki)


def kandydaci(tresc: str, ile: int = 5) -> list[dict]:
    """Znajduje istniejace notatki tematycznie najblizsze zapiskowi.

    Sam inbox jest odsiewany - inaczej najlepszym kandydatem na miejsce docelowe
    dla wpisu z inboxu bylby ten sam wpis w inboxie.

    max_distance=None wylacza prog: tu nie chodzi o to, czy odpowiedz istnieje,
    tylko o to, ktore notatki sa najblizej. Nawet slabe dopasowanie jest
    informacja, a odmowa nie mialaby sensu.
    """
    _, wszystkie = core.search(tresc, k=ile * 3, max_distance=None)
    widziane: set[str] = set()
    out: list[dict] = []
    for h in wszystkie:
        rel = h.metadata.get("rel_path", h.source)
        if rel.replace("\\", "/").startswith("inbox/") or h.source.startswith("surowe-"):
            continue
        klucz = f"{rel}|{h.metadata.get('heading', '')}"
        if klucz in widziane:
            continue
        widziane.add(klucz)
        out.append(
            {
                "plik": rel,
                "sekcja": h.metadata.get("heading", ""),
                "dystans": h.distance,
            }
        )
        if len(out) >= ile:
            break
    return out


# Dwa kroki zamiast jednego.
#
# Powod: zapisek wrzucony w biegu czesto zawiera KILKA niepowiazanych rzeczy
# ("gadalem z ksiegowa, deadline 20.08, a przy okazji Rockwell chce start we
# wrzesniu"). Traktowany jako calosc trafial do jednej notatki i rozmywal jej
# wektor na trzy tematy naraz - a przy wyszukiwaniu przegrywal z fragmentami,
# ktore mowia o jednej rzeczy.
#
# Krok 1 rozbija zapisek na pojedyncze fakty. Krok 2 dla KAZDEGO faktu osobno
# szuka kandydatow w bazie i wybiera miejsce docelowe. Rozdzielenie ma te sama
# zalete co przy sedzim w messenger.py: zadanie otwarte (co tu jest?) jest
# oddzielone od zamknietego (gdzie to wlozyc?).

ROZBIJ_PROMPT = """Dostajesz surowy zapisek wrzucony w biegu do skrzynki wejsciowej.

Rozbij go na POJEDYNCZE, niezalezne fakty - po jednym w linii, kazdy zaczynajac
od myslnika. Jeden fakt to jedna rzecz, o ktora dalo by sie osobno zapytac.

Przyklad:
  zapisek: "gadalem z ksiegowa, rozliczenie do 20.08 mailem nie przez portal,
            faktury skanem. no i rockwell chce start we wrzesniu"
  wynik:
  - Rozliczenie skladam do 20 sierpnia mailem do biura, nie przez portal.
  - Faktury kosztowe musza byc podpisane skanem, zdjecie nie przejdzie.
  - Rockwell oczekuje rozpoczecia stazu we wrzesniu.

Zasady:
1. Zdania pelne i samodzielne - fakt ma byc zrozumialy bez reszty zapisku.
2. Rozwijasz skroty myslowe, ale NIE DODAJESZ niczego, czego w zapisku nie ma.
3. Zachowujesz pierwsza osobe. To sa notatki uzytkownika o sobie.
4. Pomijasz to, co bylo aktualne tylko w chwili pisania.
5. Jesli zapisek dotyczy jednej rzeczy - zwracasz jeden punkt. Nie dziel na sile.
6. PROG ISTOTNOSCI: kazdy punkt musi byc wart zapamietania SAM Z SIEBIE za rok.
   Data, nastroj i oceny w rodzaju "poszlo dobrze" to opis chwili, nie fakt.
   MAKSYMALNIE CZTERY punkty. Lepiej trzy mocne niz osiem drobnych.
7. Jesli nic nie nadaje sie na trwala notatke, odpowiadasz jednym slowem: POMIN

Po polsku, wylacznie alfabetem lacinskim. Zadnego wstepu, same punkty."""


UMIESC_PROMPT = """Dostajesz JEDEN fakt oraz liste ISTNIEJACYCH notatek najblizszych
mu tematycznie. Decydujesz, gdzie go zapisac.

Odpowiadasz DOKLADNIE w tym formacie, bez wstepu:

PLIK: <sciezka z listy albo NOWY: proponowana/sciezka.md>
SEKCJA: <naglowek sekcji>
TRESC:
<fakt, 1-2 zdania>

NAGLOWEK SEKCJI JEST WAZNY TECHNICZNIE, nie tylko porzadkowo. Indekser sklada
embedowany tekst jako "tytul notatki > naglowek sekcji" PRZED trescia, wiec
naglowek wspoltworzy wektor fragmentu. Dobierz go tak, zeby zawieral slowa,
ktorych uzylbys, pytajac o te rzecz.

  ZLE:    "Uwagi", "Rozne", "2026-08", "Notatka"
  DOBRZE: "Terminy rozliczen z ksiegowa", "Wymogi formalne faktur"

NOWY PLIK TO OSTATECZNOSC. Dostajesz PELNA liste istniejacych notatek - jesli
jakakolwiek dotyczy tego obszaru tematycznego, uzywasz jej. Osobny plik na jedno
zdanie psuje wyszukiwanie: powstaje notatka z jednym fragmentem, ktora konkuruje
z porzadnymi notatkami i prawie nigdy nie wygrywa.

  Fakt o negocjacjach placowych -> istniejaca notatka o wynagrodzeniu,
  nawet jesli trzeba w niej zalozyc nowa sekcje.
  NIE: nowy plik "strategia-negocjacji.md" obok istniejacej notatki o placach.

Zasada praktyczna: nowy plik zakladasz tylko wtedy, gdy fakt otwiera temat,
ktorego w calym vaulcie nie ma. Nowa SEKCJA w istniejacym pliku jest prawie
zawsze lepsza niz nowy plik.

Zasady:
1. Plik wybierasz Z LISTY ISTNIEJACYCH. Sciezke spoza listy poprzedzasz slowem
   NOWY: i robisz to wylacznie w sytuacji opisanej wyzej.
2. Jesli inne fakty z tego samego zapisku trafily juz gdzies, gdzie ten tez
   pasuje - wybierz TO SAMO miejsce. Lista wyborow jest podana nizej.
3. Preferuj ISTNIEJACA sekcje, jesli fakt do niej pasuje.
4. Nie dodajesz faktow, ktorych w podanym zdaniu nie ma.
5. Piszesz w PIERWSZEJ osobie ("przedstawilem"), nie w drugiej ("przedstawiles").
6. Jesli fakt jest juz w calosci pokryty przez istniejace notatki: POMIN

Po polsku, wylacznie alfabetem lacinskim."""


def rozbij_na_fakty(wpis: dict) -> list[str]:
    """Krok 1: zapisek -> lista pojedynczych faktow."""
    wynik = core.chat(
        [
            {"role": "system", "content": ROZBIJ_PROMPT},
            {"role": "user",
             "content": f"ZAPISEK ({wpis['data']}, {wpis['naglowek']}):\n{wpis['tresc']}"},
        ],
        temperature=0.1,
    ).strip()

    if core.bez_ogonkow_core(wynik).startswith("pomin"):
        return []
    fakty = [
        re.sub(r"^\s*[-*]\s*", "", l).strip()
        for l in wynik.split("\n")
        if l.strip().startswith(("-", "*")) and len(l.strip()) > 12
    ]
    # Model ignoruje limit podany slownie. Bierzemy pierwsze MAX_FAKTOW - sa
    # zwykle najwazniejsze, bo wypisuje je w kolejnosci wystepowania w zapisku.
    if len(fakty) > MAX_FAKTOW:
        print(f"[obciete z {len(fakty)}]", end=" ", flush=True)
    return fakty[:MAX_FAKTOW]


# Ponizej tego dystansu uznajemy, ze w vaulcie JEST juz notatka o tym obszarze,
# wiec zakladanie nowego pliku nie ma uzasadnienia. 0.55 to wartosc z tego samego
# rzedu co prog odpowiadania - dobrane tak, zeby przepuszczac naprawde nowe tematy.
# Praktycznie zawsze przekierowuj. Patrz uzasadnienie w _blokuj_nowy_plik:
# prog 0.55 nie zadzialal ani razu na realnych danych.
PROG_NOWY_PLIK = 999.0

# Twardy limit faktow z jednego zapisku. Instrukcja "MAKSYMALNIE CZTERY" w prompcie
# zostala zignorowana - model zwrocil osiem. Liczbe da sie wymusic tylko kodem.
MAX_FAKTOW = 4

# Druga osoba w tresci. Zrodlem bywa wklejona odpowiedz asystenta, ktora jest
# pisana DO uzytkownika - model ja wiernie przepisuje mimo zakazu w prompcie.
_DRUGA_OSOBA_RE = re.compile(
    r"\b(\w+(les|las|lem sie|isz|asz|esz)|twoj\w*|ciebie|tobie|cie)\b"
)


def _ma_druga_osobe(tresc: str) -> bool:
    t = core.bez_ogonkow_core(tresc)
    return len(_DRUGA_OSOBA_RE.findall(t)) >= 1


def _blokuj_nowy_plik(wynik: str, kand: list[dict]) -> tuple[str, bool]:
    """Zamienia NOWY: na najlepszego istniejacego kandydata, gdy ten jest blisko.

    Regula w prompcie nie wystarcza - to trzeci przebieg, w ktorym model mimo
    jawnego zakazu zakladal plik obok istniejacej notatki o tym samym. Dopisywanie
    kolejnych zdan do promptu wypychalo wczesniejsze reguly i psulo wynik bardziej,
    niz pomagalo.
    """
    m = re.search(r"^PLIK:\s*(.+)$", wynik, re.MULTILINE)
    if not m or not m.group(1).strip().upper().startswith("NOWY"):
        return wynik, False
    if not kand:
        return wynik, False  # pusta baza - nie ma na co przekierowac

    # Prog 0.55 nie zadzialal ANI RAZU: fakty opisujace zdarzenie ("rozmowa
    # odbyla sie 7 sierpnia") maja dystans powyzej, bo notatki opisuja stany,
    # nie zdarzenia. Efekt byl odwrotny do zamierzonego - blokada przepuszczala
    # dokladnie te przypadki, dla ktorych powstala.
    #
    # Nowa regula: jesli w vaulcie jest COKOLWIEK, nowy plik nie powstaje.
    # Przy 26 przemyslanych notatkach szansa, ze zapisek z inboxu otwiera
    # naprawde nowy obszar, jest znikoma - a koszt bledu asymetryczny:
    # zla sekcja to drobiazg, zbedny plik zasmieca baze na stale.
    # Nowy plik zakladasz sam, gdy uznasz, ze propozycja go potrzebuje.
    if kand[0]["dystans"] > PROG_NOWY_PLIK:
        return wynik, False

    return re.sub(
        r"^PLIK:\s*.+$", f"PLIK: {kand[0]['plik']}", wynik, count=1, flags=re.MULTILINE
    ), True


def przetworz(limit: int) -> None:
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"zrobione": []}
    zrobione = set(man["zrobione"])

    wpisy = [w for w in wpisy_inboxu() if w["hash"] not in zrobione]
    if not wpisy:
        print("Brak nowych wpisow do przetworzenia.")
        return

    nowy_plik = not PROPOZYCJE.exists()
    zapisane = pominiete = 0

    with PROPOZYCJE.open("a", encoding="utf-8") as out:
        if nowy_plik:
            out.write("# Propozycje rozdzielenia inboxu\n\n")
            out.write(
                "> NIC z tego nie trafilo do vaultu. Przejrzyj, popraw, przenies recznie.\n"
                "> Po przeniesieniu usun odpowiedni wpis z inbox/surowe-*.md, zeby ta sama\n"
                "> tresc nie siedziala w bazie dwa razy.\n\n"
            )

        for w in wpisy[:limit]:
            print(f"  {w['data']} {w['naglowek'][:30]:30s} ...", end=" ", flush=True)

            try:
                fakty = rozbij_na_fakty(w)
            except Exception as exc:  # noqa: BLE001
                print(f"blad rozbijania: {exc}")
                continue

            man["zrobione"].append(w["hash"])

            if not fakty:
                pominiete += 1
                print("pomijam")
                continue

            print(f"{len(fakty)} fakt(ow)")
            umieszczone = []
            juz_wybrane: list[str] = []   # miejsca uzyte przez wczesniejsze fakty
            wszystkie = wszystkie_notatki()

            for i, fakt in enumerate(fakty, 1):
                print(f"      [{i}/{len(fakty)}] {fakt[:46]:46s} ...", end=" ", flush=True)
                # Kandydaci szukani DLA POJEDYNCZEGO FAKTU, nie dla calego zapisku.
                # Przy zapisku z trzema tematami wyszukiwanie na calosci zwracalo
                # notatki pasujace "srednio do wszystkiego" i do zadnej dobrze.
                kand = kandydaci(fakt)
                # BEZ dystansu przy nazwie sekcji. Model przepisywal go jako czesc
                # nazwy - w wyniku powstala sekcja "Ustalone kwoty (dystans 0.299)".
                # Liczba byla informacja dla mnie, nie dla modelu.
                lista = "\n".join(
                    f"- {k['plik']} > {k['sekcja']}" for k in kand
                ) or "(brak kandydatow - baza jest pusta)"

                tresc_uzytkownika = (
                    f"FAKT:\n{fakt}\n\n"
                    f"NAJBLIZSZE TEMATYCZNIE SEKCJE:\n{lista}\n\n"
                    f"WSZYSTKIE ISTNIEJACE NOTATKI:\n"
                    + "\n".join(f"- {p}" for p in wszystkie)
                )
                if juz_wybrane:
                    tresc_uzytkownika += (
                        "\n\nMIEJSCA JUZ WYBRANE DLA INNYCH FAKTOW Z TEGO ZAPISKU "
                        "(uzyj tego samego, jesli pasuje):\n"
                        + "\n".join(f"- {m}" for m in dict.fromkeys(juz_wybrane))
                    )

                try:
                    wynik = core.chat(
                        [
                            {"role": "system", "content": UMIESC_PROMPT},
                            {"role": "user", "content": tresc_uzytkownika},
                        ],
                        temperature=0.1,
                    ).strip()
                except Exception as exc:  # noqa: BLE001
                    print(f"blad: {exc}")
                    continue

                if core.bez_ogonkow_core(wynik).startswith("pomin"):
                    print("pomijam")
                    continue

                wynik, przekierowany = _blokuj_nowy_plik(wynik, kand)

                # Druga osoba: jedno ponowienie z jawna korekta, jak przy cyrylicy.
                tresc_m = re.search(r"^TRESC:\s*(.*)$", wynik, re.MULTILINE | re.DOTALL)
                if tresc_m and _ma_druga_osobe(tresc_m.group(1)):
                    try:
                        poprawiona = core.chat(
                            [
                                {"role": "system",
                                 "content": "Przepisz podany tekst w PIERWSZEJ osobie "
                                            "(ja o sobie), nie zmieniajac tresci ani faktow. "
                                            "Odpowiedz samym przepisanym tekstem."},
                                {"role": "user", "content": tresc_m.group(1).strip()},
                            ],
                            temperature=0.0,
                        ).strip()
                        # Akceptujemy takze POPRAWE czesciowa. Wczesniej warunek
                        # brzmial "zero drugiej osoby" i przy nieudanej przerobce
                        # zostawal oryginal - czyli najgorszy z trzech wariantow.
                        przed = len(_DRUGA_OSOBA_RE.findall(
                            core.bez_ogonkow_core(tresc_m.group(1))))
                        po = len(_DRUGA_OSOBA_RE.findall(
                            core.bez_ogonkow_core(poprawiona)))
                        if poprawiona and po < przed:
                            wynik = wynik[: tresc_m.start(1)] + poprawiona
                    except Exception:  # noqa: BLE001
                        pass

                m = re.search(r"^PLIK:\s*(.+)$", wynik, re.MULTILINE)
                sek = re.search(r"^SEKCJA:\s*(.+)$", wynik, re.MULTILINE)
                if m:
                    juz_wybrane.append(
                        f"{m.group(1).strip()} > {sek.group(1).strip() if sek else ''}"
                    )
                    if przekierowany:
                        print(f"NOWY -> {m.group(1).strip()[:34]}")
                    elif m.group(1).strip().upper().startswith("NOWY"):
                        print("NOWY plik")
                    else:
                        print("ok")
                else:
                    print("ok")
                umieszczone.append(wynik)

            if not umieszczone:
                pominiete += 1
                continue

            zapisane += len(umieszczone)
            out.write(f"## {w['data']} - {w['naglowek']}\n\n")
            out.write(f"*Zrodlo: {w['plik']} - rozbite na {len(umieszczone)} fakt(ow)*\n\n")
            for u in umieszczone:
                out.write(u + "\n\n")
            out.write("<details><summary>oryginalny zapisek</summary>\n\n")
            out.write(f"{w['tresc']}\n\n</details>\n\n---\n\n")
            out.flush()

    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nPropozycji: {zapisane}, pominietych: {pominiete}")
    print(f"Plik: {PROPOZYCJE}")


def statystyki() -> None:
    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"zrobione": []}
    zrobione = set(man["zrobione"])
    wpisy = wpisy_inboxu()
    nowe = [w for w in wpisy if w["hash"] not in zrobione]

    print(f"\nInbox: {pamiec.INBOX}")
    print(f"  plikow      : {len(set(w['plik'] for w in wpisy))}")
    print(f"  wpisow      : {len(wpisy)}")
    print(f"  przetworzone: {len(wpisy) - len(nowe)}")
    print(f"  do zrobienia: {len(nowe)}")
    if nowe:
        print(f"\n  {'data':12s} {'naglowek':32s} {'znakow':>7s}")
        print("  " + "-" * 54)
        for w in nowe[:20]:
            print(f"  {w['data']:12s} {w['naglowek'][:32]:32s} {len(w['tresc']):7d}")
    print()


def main() -> int:
    # cp1250 w konsoli Windows nie zna emoji - patrz komentarz w messenger.py
    for strumien in (sys.stdout, sys.stderr):
        try:
            strumien.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    ap = argparse.ArgumentParser(description="Przetwarza inbox na propozycje notatek.")
    ap.add_argument("--statystyki", action="store_true", help="pokaz, co czeka; nie wywoluj modelu")
    ap.add_argument("--przetworz", action="store_true", help="generuj propozycje")
    ap.add_argument("--limit", type=int, default=10, help="ile wpisow na przebieg")
    a = ap.parse_args()

    if a.przetworz:
        przetworz(a.limit)
    else:
        statystyki()
    return 0


if __name__ == "__main__":
    sys.exit(main())
