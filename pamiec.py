"""
pamiec.py - dopisywanie notatek, historia rozmow, zrzuty sesji.

Trzy rzeczy, ktore celowo NIE trafily do core.py: core odpowiada za wyszukiwanie
i generowanie, a to jest warstwa trwalosci. Trzymanie ich osobno sprawia, ze
zmiana formatu zapisu rozmow nie moze zepsuc retrievalu.

Gdzie co ladu je:
    Notatki/inbox/surowe-RRRR-MM-DD.md   dopisane fragmenty - CZESC VAULTU, indeksowane
    <projekt>/rozmowy/*.json             historia rozmow    - NIE indeksowana
    <projekt>/zrzuty/sesja-*.md          zrzuty sesji       - NIE indeksowane

Rozmowy sa swiadomie poza baza wektorowa. Indeksowanie wlasnych odpowiedzi modelu
tworzy petle zwrotna: model zaczyna cytowac jako zrodlo to, co sam wczesniej
wygenerowal, razem z bledami. Notatki maja pozostac tym, co przemyslales, a nie
tym, co maszyna powiedziala.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent
NOTATKI = Path(os.environ.get("NOTATKI_PATH", Path.home() / "Notatki"))
INBOX = NOTATKI / "inbox"
ROZMOWY = PROJEKT / "rozmowy"
ZRZUTY = PROJEKT / "zrzuty"

# Rozmowy dotkniete od startu serwera - tylko one ida do zrzutu sesji.
_sesja: set[str] = set()


def _teraz() -> datetime:
    return datetime.now()


def _slug(tekst: str, maks: int = 48) -> str:
    t = re.sub(r"[^0-9a-zA-ZÀ-ſ ]+", "", tekst).strip()
    t = re.sub(r"\s+", "-", t).lower()
    return t[:maks] or "rozmowa"


# ------------------------------------------------------------------ dopisywanie


def dopisz(tekst: str, tytul: str | None = None) -> dict:
    """Dopisuje fragment do dzisiejszego pliku w inboxie vaultu.

    Jeden plik na dzien, nie jeden na wpis. Powod jest praktyczny: indekser
    scala krotkie sekcje z sasiednimi, wiec pojedyncze zdanie w osobnym pliku
    nie mialoby z czym sie polaczyc i trafiloby do bazy jako samotny, ubogi
    w kontekst fragment.
    """
    tekst = (tekst or "").strip()
    if not tekst:
        raise ValueError("pusty tekst")

    INBOX.mkdir(parents=True, exist_ok=True)
    t = _teraz()
    plik = INBOX / f"surowe-{t:%Y-%m-%d}.md"

    nowy = not plik.exists()
    with plik.open("a", encoding="utf-8") as f:
        if nowy:
            f.write(f"# Surowe zapiski {t:%Y-%m-%d}\n\n")
            f.write(
                "> Dopisywane z interfejsu agenta. Do przejrzenia i rozdzielenia\n"
                "> do wlasciwych notatek.\n\n"
            )
        f.write(f"## {t:%H:%M} {tytul.strip() if tytul else ''}".rstrip() + "\n\n")
        f.write(tekst + "\n\n")

    return {"plik": str(plik), "nowy_plik": nowy, "znakow": len(tekst)}


def reindeksuj() -> dict:
    """Uruchamia indekser przyrostowo, zeby dopisane zdanie bylo od razu wyszukiwalne.

    Osobny proces, nie import: indexer.main() czyta sys.argv, a poza tym
    przeliczenie embeddingow w procesie serwera zablokowaloby obsluge zapytan
    na czas trwania indeksowania.
    """
    if not NOTATKI.exists():
        return {"ok": False, "blad": f"brak folderu {NOTATKI}"}
    try:
        p = subprocess.run(
            [sys.executable, "indexer.py", "--path", str(NOTATKI)],
            cwd=str(PROJEKT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return {
            "ok": p.returncode == 0,
            "wyjscie": (p.stdout or p.stderr or "").strip()[-600:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "blad": "indeksowanie przekroczylo 10 minut"}


# --------------------------------------------------------------------- rozmowy


def zapisz_rozmowe(rozmowa_id: str | None, wpisy: list[dict], tryb: str = "") -> dict:
    """Zapisuje albo nadpisuje rozmowe. Zwraca jej id."""
    if not wpisy:
        raise ValueError("pusta rozmowa")

    ROZMOWY.mkdir(parents=True, exist_ok=True)
    t = _teraz()

    if not rozmowa_id:
        rozmowa_id = f"{t:%Y%m%d-%H%M%S}-{_slug(wpisy[0].get('pytanie', ''))}"

    plik = ROZMOWY / f"{rozmowa_id}.json"
    utworzona = t.isoformat(timespec="seconds")
    if plik.exists():
        try:
            utworzona = json.loads(plik.read_text(encoding="utf-8")).get("utworzona", utworzona)
        except Exception:  # noqa: BLE001 - uszkodzony plik nie moze wywrocic zapisu
            pass

    dane = {
        "id": rozmowa_id,
        "tytul": (wpisy[0].get("pytanie") or "rozmowa")[:90],
        "tryb": tryb,
        "utworzona": utworzona,
        "zmieniona": t.isoformat(timespec="seconds"),
        "wpisy": wpisy,
    }
    plik.write_text(json.dumps(dane, ensure_ascii=False, indent=1), encoding="utf-8")
    _sesja.add(rozmowa_id)
    return {"id": rozmowa_id, "wpisow": len(wpisy)}


def lista_rozmow(limit: int = 200) -> list[dict]:
    """Naglowki rozmow, najnowsze pierwsze. Bez tresci - lista ma byc lekka."""
    if not ROZMOWY.exists():
        return []
    out = []
    for p in ROZMOWY.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append(
            {
                "id": d.get("id", p.stem),
                "tytul": d.get("tytul", p.stem),
                "tryb": d.get("tryb", ""),
                "zmieniona": d.get("zmieniona", ""),
                "wpisow": len(d.get("wpisy", [])),
            }
        )
    out.sort(key=lambda r: r["zmieniona"], reverse=True)
    return out[:limit]


def wczytaj_rozmowe(rozmowa_id: str) -> dict:
    plik = ROZMOWY / f"{Path(rozmowa_id).name}.json"   # .name blokuje ../ w id
    if not plik.exists():
        raise FileNotFoundError(rozmowa_id)
    return json.loads(plik.read_text(encoding="utf-8"))


def usun_rozmowe(rozmowa_id: str) -> bool:
    plik = ROZMOWY / f"{Path(rozmowa_id).name}.json"
    if plik.exists():
        plik.unlink()
        _sesja.discard(rozmowa_id)
        return True
    return False


# ---------------------------------------------------------------------- zrzuty


def zrzut(powod: str = "recznie") -> dict:
    """Zapisuje wszystkie rozmowy z biezacego uruchomienia do jednego pliku md.

    Zrzut jest kopia bezpieczenstwa i materialem do przegladu - nie zrodlem dla
    agenta. Gdyby folder rozmow kiedys padl albo format JSON sie zmienil, tekst
    zostaje czytelny golym okiem.
    """
    ids = sorted(_sesja)
    if not ids:
        return {"ok": False, "powod": "w tej sesji nie bylo rozmow"}

    ZRZUTY.mkdir(parents=True, exist_ok=True)
    t = _teraz()
    plik = ZRZUTY / f"sesja-{t:%Y-%m-%d-%H%M}.md"

    linie = [
        f"# Zrzut sesji {t:%Y-%m-%d %H:%M}",
        "",
        f"Powod: {powod} - rozmow: {len(ids)}",
        "",
    ]
    dzisiejszy_inbox = INBOX / f"surowe-{t:%Y-%m-%d}.md"
    if dzisiejszy_inbox.exists():
        linie += [f"Dopisane dzis notatki: `{dzisiejszy_inbox}`", ""]

    for rid in ids:
        try:
            d = wczytaj_rozmowe(rid)
        except Exception:  # noqa: BLE001
            continue
        linie += ["---", "", f"## {d.get('tytul', rid)}", "", f"*{d.get('zmieniona','')}*", ""]
        for w in d.get("wpisy", []):
            zrodlo = w.get("zrodlo") or w.get("tryb") or ""
            linie += [f"**P:** {w.get('pytanie','')}", ""]
            linie += [f"**O:** {w.get('odpowiedz','')}", ""]
            if zrodlo:
                linie += [f"<sub>{zrodlo}</sub>", ""]

    plik.write_text("\n".join(linie), encoding="utf-8")
    return {"ok": True, "plik": str(plik), "rozmow": len(ids)}
