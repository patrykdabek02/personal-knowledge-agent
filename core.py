"""
core.py - wspolny modul dla calego pipeline'u Personal Knowledge Agent.

Wszystko dziala lokalnie:
  - embeddingi: Ollama (domyslnie bge-m3, wielojezyczny -> dziala po polsku)
  - baza wektorowa: Chroma (PersistentClient, plik na dysku)
  - generowanie: Ollama (domyslnie qwen3.5:9b)

Zaden fragment notatek nie opuszcza tej maszyny.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# --------------------------------------------------------------------------
# Konfiguracja (wszystko nadpisywalne zmiennymi srodowiskowymi)
# --------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen3.5:9b")

# Katalog bazy wektorowej. Domyslnie ./chroma_db obok tego pliku.
CHROMA_PATH = os.environ.get("CHROMA_PATH", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge_base")

# Prog odciecia: dystans cosinusowy (0 = identyczne, 1 = brak zwiazku, 2 = przeciwne).
#
# HISTORIA TEJ WARTOSCI - warto przeczytac przed zmiana.
#
# 2026-08-04: kalibracja na 38 pytaniach (32 w zakresie + 6 spoza), 58 fragmentow,
#   dala 0.52. Skrypt rekomendowal 0.56, ale przy 0.56 przeciekalo pytanie o haslo
#   do konta bankowego (0.536), a prog byl wtedy JEDYNA obrona przed konfabulacja.
#
# 2026-08-07: podniesione do 0.65. Dwa powody.
#   (1) Zestaw kalibracyjny zawieral wylacznie pytania FAKTOGRAFICZNE ("ile mam VRAM").
#       Takie maja waskie dopasowanie i niskie dystanse (0.33-0.55). Pytania szerokie
#       i konwersacyjne ("co wiesz o mnie") sa rozproszone semantycznie i lezą w 0.52-0.60,
#       wiec prog 0.52 odrzucal je mimo trafnego retrievalu - "co wiesz o mnie" mialo
#       najlepszy fragment o dystansie 0.522, odrzucony o 0.002.
#       Zestaw testowy nie reprezentowal rzeczywistego uzycia.
#   (2) Prompt okazal sie skuteczna DRUGA warstwa obrony. Zmierzone przy 0.65:
#       "jak ugotowac risotto" (0.626) i "jakie jest moje haslo do konta bankowego"
#       przeszly przez prog, a model mimo to odmowil. Prog nie musi juz byc jedyna
#       gwarancja, wiec moze pelnic role zgrubnego filtra.
#
# Wartosc jest wazna DLA TEGO modelu i TEJ bazy. Po istotnym wzroscie liczby notatek
# przelicz ponownie, ale najpierw DOPISZ do questions.csv pytania konwersacyjne -
# inaczej kalibracja znowu zoptymalizuje sie pod jeden typ pytan.
MAX_DISTANCE = float(os.environ.get("MAX_DISTANCE", "0.65"))

# Liczba fragmentow pobieranych z bazy.
#
# Zmierzone na 32 pytaniach w zakresie (poprawnosc merytoryczna, ocena reczna):
#   k=4, GATE_ON_BEST=False -> 26/32 (81%)  wersja wyjsciowa
#   k=6, GATE_ON_BEST=False -> 26/32 (81%)  bez zmian: prog odcinal nadmiarowe fragmenty
#   k=4, GATE_ON_BEST=True  -> 27/32 (84%)  odzyskane 1 pytanie
#   k=6, GATE_ON_BEST=True  -> 30/32 (94%)  odzyskane 4 pytania  <-- ustawione
#
# Zadna z tych zmian osobno nie dawala pelnego efektu. Dopiero zlozenie obu:
# bramka przepuszcza komplet fragmentow, a szostka sprawia, ze wsrod nich jest
# wlasciwa sekcja pliku. Roznice czasowe miedzy konfiguracjami (4.7-5.9 s mediany)
# mieszcza sie w szumie pomiarowym - dwa przebiegi tej samej konfiguracji daly
# 6.07 s i 4.67 s.
TOP_K = int(os.environ.get("TOP_K", "6"))

REFUSAL = "Nie znalazlem tego w notatkach."

# Polityka odciecia. Dwa tryby:
#
#   True  (domyslnie) - prog decyduje TYLKO o tym, czy pytanie miesci sie w bazie
#         wiedzy, i rozstrzyga to NAJLEPSZE trafienie. Jesli przejdzie, model
#         dostaje wszystkie k fragmentow.
#   False - kazdy fragment filtrowany osobno (zachowanie pierwotne).
#
# Powod zmiany: przy najlepszym dystansie bliskim progu przez filtr przechodzil
# jeden fragment, i niekoniecznie ten wlasciwy w obrebie pliku. Model dostawal
# sekcje "Co to jest" zamiast "Napotkane problemy" i uczciwie odmawial mimo
# poprawnego trafienia w plik. Podnoszenie k tego NIE naprawialo - dodatkowe
# fragmenty i tak odpadaly na progu (zmierzone: k=4 i k=6 daly identyczny wynik).
#
# Gwarancja odmowy zostaje: pytanie spoza zakresu ma najlepszy dystans powyzej
# progu, wiec model nadal nie jest wywolywany.
GATE_ON_BEST = os.environ.get("GATE_ON_BEST", "1") not in ("0", "false", "False")

# Hybrid search: laczenie wyszukiwania wektorowego z BM25 metoda RRF.
# Wektory lapia znaczenie, BM25 lapie rzadkie ciagi znakow (COOIS, F00003,
# 80070057, nazwiska). Wylaczenie: USE_BM25=0
USE_BM25 = os.environ.get("USE_BM25", "1") not in ("0", "false", "False")

# Okno kontekstu modelu generujacego. Domyslne 16384 w qwen3.5 zajmuje w VRAM
# tyle, ze czesc warstw laduje na CPU (sprawdz: `ollama ps` -> kolumna PROCESSOR).
# Nasz realny prompt: k=4 fragmenty x ~350 slow + system + pytanie ~= 3000 tokenow.
# 6144 daje zapas, a zwalnia VRAM na pelniejszy offload modelu.
# Jesli podniesiesz TOP_K powyzej 6, podnies tez te wartosc.
NUM_CTX = int(os.environ.get("NUM_CTX", "6144"))

# Twardy limit generowanych tokenow - zabezpieczenie przed rozbiegnieta generacja.
# UWAGA: przy WLACZONYM thinkingu zbyt niski limit ucina blok <think> w polowie,
# a po jego wycieciu zostaje pusta odpowiedz. Dlatego trzymamy zapas.
NUM_PREDICT = int(os.environ.get("NUM_PREDICT", "1200"))

# Wylaczenie trybu rozumowania. Modele Qwen3 generuja blok <think>, ktory i tak
# wycinamy - a placimy za niego pelnym czasem generowania.
# Ollama przyjmuje "think": false na poziomie requestu (NIE w "options").
# Jesli model albo wersja Ollamy tego nie wspiera, kod cofa sie automatycznie.
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "1") not in ("0", "false", "False")

# Poziom rozumowania. Ollama przyjmuje w polu "think" nie tylko true/false,
# ale takze poziomy: "low", "medium", "high", "max". To nie jest przelacznik,
# tylko suwak - a my przez caly czas uzywalismy go jak przelacznika.
#
# Dlaczego to wazne: wylaczenie thinkingu zbilo czas z 24,9 s do 9,5 s, ale
# odbieralo modelowi rozumowanie tam, gdzie faktycznie sie przydaje - przy
# ocenie, czy fakt jest trwaly, albo przy zderzaniu sprzecznych twierdzen.
# Poziom posredni pozwala placic za rozumowanie tylko tam, gdzie ma zwrot.
#
# UWAGA na NUM_PREDICT: przy wlaczonym rozumowaniu blok <think> zjada limit
# tokenow. Zbyt niski NUM_PREDICT ucina go w polowie i po wycieciu zostaje
# pusta odpowiedz - dokladnie ten blad mielismy przy 600. Kod podnosi limit
# automatycznie, gdy think nie jest wylaczony.
#
# Wartosci: "false" (bez rozumowania), "true", "low", "medium", "high", "max".
POZIOMY_THINK = ("low", "medium", "high", "max")
THINK = os.environ.get("THINK", "false" if DISABLE_THINKING else "true").strip().lower()


def _think_wartosc(poziom: str | bool | None):
    """Zamienia ustawienie na to, co Ollama przyjmuje w polu "think"."""
    if poziom is None:
        poziom = THINK
    if isinstance(poziom, bool):
        return poziom
    p = str(poziom).strip().lower()
    if p in ("false", "0", "nie", "off"):
        return False
    if p in ("true", "1", "tak", "on"):
        return True
    if p in POZIOMY_THINK:
        return p
    return False

EMBED_TIMEOUT = int(os.environ.get("EMBED_TIMEOUT", "300"))
CHAT_TIMEOUT = int(os.environ.get("CHAT_TIMEOUT", "600"))


# --------------------------------------------------------------------------
# Embeddingi przez Ollama
# --------------------------------------------------------------------------

def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Zwraca liste wektorow dla listy tekstow.

    Uzywa nowego endpointu /api/embed (batch). Jesli Ollama jest starsza
    i zwraca 404, spada na /api/embeddings (jeden tekst na raz).
    """
    if not texts:
        return []

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": texts},
                timeout=EMBED_TIMEOUT,
            )
            if r.status_code == 404:
                return _embed_legacy(texts)
            r.raise_for_status()
            data = r.json()
            vectors = data.get("embeddings")
            if not vectors:
                raise RuntimeError(f"Ollama nie zwrocila embeddingow: {data}")
            return vectors
        except Exception as exc:  # noqa: BLE001 - swiadomy retry na wszystkim
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"Nie udalo sie pobrac embeddingow z Ollama ({OLLAMA_URL}). "
        f"Czy Ollama dziala i czy model '{EMBED_MODEL}' jest pobrany? "
        f"Ostatni blad: {last_err}"
    )


def _embed_legacy(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=EMBED_TIMEOUT,
        )
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------

def get_collection(create: bool = True):
    """Zwraca kolekcje Chroma z metryka cosinusowa.

    UWAGA: domyslna metryka w Chromie to L2. Wymuszamy 'cosine',
    bo progi dystansu w tym projekcie sa liczone dla cosinusa.
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


# --------------------------------------------------------------------------
# BM25 - ranking leksykalny (dopelnienie wyszukiwania wektorowego)
# --------------------------------------------------------------------------
#
# Embeddingi lapia znaczenie, ale gubia rzadkie nazwy wlasne: "COOIS", "F00003",
# "80070057", nazwiska. Dla modelu to prawie szum, bo takich ciagow nie widzial
# w treningu. BM25 dziala odwrotnie - liczy dokladne trafienia slow i premiuje te
# rzadkie. Polaczenie obu (hybrid search) laduje trafnie w obu przypadkach.
#
# Implementacja wlasna, bo rank_bm25 to kolejna zaleznosc dla ~50 linii kodu.

import math
import re as _re

BM25_K1 = 1.5   # nasycenie czestoscia - powyzej tego kolejne wystapienia daja coraz mniej
BM25_B = 0.75   # normalizacja dlugoscia dokumentu

_TOKEN_RE = _re.compile(r"[0-9a-ząćęłńóśźż]+", _re.IGNORECASE)
_DIAKRYTYKI = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


# Lekki stemmer dla polskiego. Bez niego BM25 jest w polskim prawie bezuzyteczny:
# "baza wektorowa" nie trafia w "bazy wektorowej", a "Bialobrzegi" w "Bialobrzegach".
# Pelny stemmer (Morfologik) to kolejna zaleznosc; obcinanie najczestszych koncowek
# zalatwia wiekszosc przypadkow za 20 linii.
#
# Kolejnosc ma znaczenie - najdluzsze koncowki najpierw, inaczej "owego" zostanie
# obciete do "ow" zamiast do rdzenia.
_KONCOWKI = (
    "iejszego", "iejszym", "iejsza", "iejsze",
    "owego", "owemu", "owych", "owymi", "iego", "iemu", "ymi", "imi",
    "ach", "ami", "owi", "owa", "owe", "owy", "ego", "emu", "ych", "ich",
    "cie", "cia", "ciu", "nie", "nia", "niu",
    "em", "ie", "om", "ow", "ej", "ym", "im", "ia", "ie",
    "a", "e", "i", "o", "u", "y",
)


MIN_TOKEN = 4   # krotszych nie ruszamy
MIN_RDZEN = 3   # ponizej tego rdzen przestaje cokolwiek znaczyc


def stem(token: str) -> str:
    """Obcina koncowki fleksyjne, maksymalnie dwie warstwy.

    Dwie warstwy sa konieczne, bo polski laczy sufiks slowotworczy z fleksyjnym:
    'wektorowej' = wektor + ow + ej. Jedno przejscie dalo 'wektorow', a 'wektorowa'
    dawalo 'wektor' - te same slowa nie trafialyby w siebie.

    NIE rusza tokenow z cyframi - dzieki temu 'F00003' i '80070057' zostaja
    nietkniete, a to wlasnie na nich BM25 daje najwiecej wartosci.
    """
    if any(c.isdigit() for c in token):
        return token

    for _ in range(2):
        if len(token) < MIN_TOKEN:
            break
        obciete = False
        for k in _KONCOWKI:
            if token.endswith(k) and len(token) - len(k) >= MIN_RDZEN:
                token = token[: -len(k)]
                obciete = True
                break
        if not obciete:
            break
    return token


def tokenizuj(tekst: str) -> list[str]:
    """Tokeny bez diakrytykow, sprowadzone do rdzenia.

    Diakrytyki usuwane, zeby 'Gdansk' == 'Gdańsk' (czeste przy szybkim pisaniu).
    """
    return [
        stem(t.translate(_DIAKRYTYKI).lower())
        for t in _TOKEN_RE.findall(tekst)
    ]


class BM25:
    """Klasyczny BM25 Okapi. Buduje sie w pamieci z listy dokumentow."""

    def __init__(self, dokumenty: list[str]) -> None:
        self.tokeny = [tokenizuj(d) for d in dokumenty]
        self.n = len(self.tokeny)
        self.dlugosci = [len(t) for t in self.tokeny]
        self.srednia_dlugosc = (sum(self.dlugosci) / self.n) if self.n else 0.0

        # df: w ilu dokumentach wystepuje dany token
        self.df: dict[str, int] = {}
        for toks in self.tokeny:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

        # czestosci w kazdym dokumencie
        self.tf: list[dict[str, int]] = []
        for toks in self.tokeny:
            licznik: dict[str, int] = {}
            for t in toks:
                licznik[t] = licznik.get(t, 0) + 1
            self.tf.append(licznik)

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def punkty(self, zapytanie: str) -> list[float]:
        """Wynik BM25 dla kazdego dokumentu. Wyzszy = lepiej (odwrotnie niz dystans)."""
        q = tokenizuj(zapytanie)
        wyniki = [0.0] * self.n
        for token in q:
            idf = self.idf(token)
            if idf == 0.0:
                continue
            for i in range(self.n):
                f = self.tf[i].get(token, 0)
                if f == 0:
                    continue
                norma = 1 - BM25_B + BM25_B * (self.dlugosci[i] / self.srednia_dlugosc)
                wyniki[i] += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * norma)
        return wyniki


# Indeks BM25 budowany leniwie i przebudowywany, gdy zmieni sie liczba fragmentow.
_bm25_cache: dict | None = None


def reset_cache() -> None:
    """Unieważnia indeks BM25 po dopisaniu notatek.

    Cache i tak przelicza sie sam, gdy zmieni sie liczba fragmentow - ale gdy
    dopisany tekst zostal SCALONY z istniejaca sekcja, liczba fragmentow bywa
    ta sama mimo zmienionej tresci. Wtedy tylko jawny reset ratuje sytuacje.
    """
    global _bm25_cache, _tozsamosc_cache
    _bm25_cache = None
    _tozsamosc_cache = None      # notatka o tozsamosci mogla sie wlasnie zmienic


def liczba_fragmentow() -> int:
    try:
        return get_collection().count()
    except Exception:  # noqa: BLE001
        return -1


def _bm25_index() -> dict:
    """Zwraca {bm25, ids, documents, metadatas}. Odswieza sie po zmianie bazy."""
    global _bm25_cache
    collection = get_collection()
    liczba = collection.count()

    if _bm25_cache is not None and _bm25_cache["liczba"] == liczba:
        return _bm25_cache

    dane = collection.get(include=["documents", "metadatas"])
    dokumenty = dane.get("documents") or []
    _bm25_cache = {
        "liczba": liczba,
        "bm25": BM25(dokumenty),
        "ids": dane.get("ids") or [],
        "documents": dokumenty,
        "metadatas": dane.get("metadatas") or [],
    }
    return _bm25_cache


@dataclass
class Hit:
    text: str
    distance: float
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("rel_path") or self.metadata.get("source") or "?")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "distance": round(self.distance, 4),
            "source": self.source,
            "heading": self.metadata.get("heading", ""),
            "chunk_index": self.metadata.get("chunk_index"),
            "category": self.metadata.get("category", ""),
        }


def search(
    query: str,
    k: int = TOP_K,
    max_distance: float | None = MAX_DISTANCE,
    category: str | None = None,
) -> tuple[list[Hit], list[Hit]]:
    """Zwraca (trafienia_po_progu, wszystkie_trafienia).

    Drugi element przydaje sie do kalibracji progu - widzisz co odrzucil.
    """
    vector = embed([query])[0]
    collection = get_collection()

    where = {"category": category} if category else None
    res = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    all_hits: list[Hit] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        all_hits.append(Hit(text=doc, distance=float(dist), metadata=dict(meta or {})))

    if USE_BM25:
        all_hits = polacz_rrf(all_hits, bm25_hits(query, k=max(k * 3, 20)), k=k)

    return select_context(all_hits, max_distance), all_hits


def bm25_hits(query: str, k: int = 20) -> list[Hit]:
    """Najlepsze trafienia wedlug samego BM25, bez wektorow."""
    idx = _bm25_index()
    punkty = idx["bm25"].punkty(query)
    if not punkty:
        return []
    kolejnosc = sorted(range(len(punkty)), key=lambda i: punkty[i], reverse=True)[:k]
    wynik = []
    for i in kolejnosc:
        if punkty[i] <= 0:
            break
        wynik.append(
            Hit(
                text=idx["documents"][i],
                # BM25 nie ma dystansu - wstawiamy wartosc powyzej progu, zeby
                # przypadkiem nie zostala uzyta do decyzji o zakresie
                distance=999.0,
                metadata=dict(idx["metadatas"][i] or {}),
            )
        )
    return wynik


RRF_K = 60  # stala tlumiaca z oryginalnej pracy o Reciprocal Rank Fusion


def polacz_rrf(wektorowe: list[Hit], leksykalne: list[Hit], k: int) -> list[Hit]:
    """Laczy dwa rankingi metoda Reciprocal Rank Fusion.

    Kazdy fragment dostaje punkty 1/(RRF_K + pozycja) z kazdej listy, w ktorej
    wystapil. Fragment wysoko w obu rankingach wygrywa z takim, ktory jest
    pierwszy w jednym i nieobecny w drugim.

    Zaleta RRF: nie trzeba skalowac ani wazyc wynikow z roznych systemow -
    liczy sie wylacznie pozycja, a te sa porownywalne z definicji.

    WAZNE: fragmentom z BM25 zachowujemy dystans wektorowy, jesli byl znany.
    Prog odciecia dziala na dystansie, wiec fragment znany tylko z BM25
    (distance=999) nie moze przypadkiem otworzyc bramki.
    """
    punkty: dict[str, float] = {}
    obiekty: dict[str, Hit] = {}

    def klucz(h: Hit) -> str:
        return f"{h.source}::{h.metadata.get('chunk_index')}"

    for lista in (wektorowe, leksykalne):
        for pozycja, h in enumerate(lista):
            kl = klucz(h)
            punkty[kl] = punkty.get(kl, 0.0) + 1.0 / (RRF_K + pozycja + 1)
            # zawsze wolimy wersje ze znanym dystansem wektorowym
            if kl not in obiekty or (h.distance < obiekty[kl].distance):
                obiekty[kl] = h

    najlepsze = sorted(punkty, key=lambda kl: punkty[kl], reverse=True)[:k]
    return [obiekty[kl] for kl in najlepsze]


def select_context(all_hits: list[Hit], max_distance: float | None) -> list[Hit]:
    """Decyduje, ktore fragmenty trafia do promptu.

    GATE_ON_BEST=True : najlepsze trafienie rozstrzyga o zakresie; jesli przejdzie,
                        model dostaje komplet k fragmentow.
    GATE_ON_BEST=False: kazdy fragment filtrowany osobno.
    """
    if not all_hits or max_distance is None:
        return all_hits
    if GATE_ON_BEST:
        return all_hits if all_hits[0].distance <= max_distance else []
    return [h for h in all_hits if h.distance <= max_distance]


# --------------------------------------------------------------------------
# Prompt + generowanie
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""Jestes asystentem osobistej bazy wiedzy. Odpowiadasz WYLACZNIE na podstawie
fragmentow notatek podanych w sekcji KONTEKST.

Zasady, ktorych nie wolno zlamac:
1. Nie korzystasz z wiedzy spoza KONTEKSTU. Nawet jesli znasz odpowiedz - nie uzywasz jej.
2. Po kazdym twierdzeniu podajesz numer fragmentu w nawiasie kwadratowym: [1], [2].
   Numery sa widoczne w KONTEKSCIE przy kazdym fragmencie. Gdy twierdzenie opiera sie
   na kilku fragmentach, piszesz [1][3]. NIE wpisujesz nazw plikow - sam numer.
   Zdanie bez numeru czytelnik zobaczy jako Twoj wlasny komentarz, nie fakt z notatek,
   wiec nie zostawiaj bez numeru niczego, co pochodzi z notatek.
3. Zanim napiszesz, ze czegos nie ma, sprawdz KAZDY podany fragment po kolei.
   Odpowiedz moze byc w ktorymkolwiek z nich, takze czesciowa. Czesciowa odpowiedz
   z podaniem zrodla jest lepsza niz odmowa.
4. Dopiero gdy ZADEN fragment nie zawiera odpowiedzi, piszesz doslownie: {REFUSAL}
   Nie zgadujesz, nie parafrazujesz pytania, nie uzupelniasz braku wlasna wiedza.
   Gdy odmawiasz, nie podajesz zadnego zrodla.
5. Jesli fragmenty sobie przecza, mowisz o tym wprost i cytujesz oba zrodla.
5a. CZYTASZ RAMKE FRAGMENTU, NIE TYLKO TRESC. Naglowek sekcji mowi, jaki status ma
   to, co pod nim stoi. Naglowki w rodzaju "Propozycja", "Do rozwazenia", "Swiadomie
   odlozone", "Co jest propozycja, a nie moja decyzja", "Nastepne kroki", "Warte
   sprawdzenia" oznaczaja POMYSL, ktory NIE zostal wdrozony. Opisujesz go jako plan
   albo rozwazana opcje - nigdy jako stan faktyczny.
   ZLE:  "Surowy zrzut trafia do folderu raw/, przetwarzanie co 30 minut."
   DOBRZE: "W notatkach jest propozycja folderu raw/ z cyklicznym przetwarzaniem,
           ale zapisana jako pomysl do rozwazenia, nie jako wdrozone rozwiazanie."
   To samo dotyczy czasu: fragment moze opisywac stan sprzed miesiecy.
6. Odpowiadasz zwiezle, w jezyku pytania. Nie opisujesz swojego toku rozumowania.
   Mozesz uzywac markdowna: **pogrubienie**, listy zaczynane myslnikiem, `kod`.
   Nie uzywasz naglowkow (#) - odpowiedz to kilka akapitow, nie dokument.
7. Piszesz wylacznie alfabetem lacinskim. Nie uzywasz cyrylicy.
8. Notatki sa pisane w PIERWSZEJ OSOBIE przez uzytkownika. Ty nie jestes uzytkownikiem.
   Zwracasz sie do niego w drugiej osobie: "Twoje", "wybrales", "mieszkasz".
   Nigdy nie piszesz o uzytkowniku "jestem", "mieszkam", "moje", "moim celem".
"""

USER_TEMPLATE = """KONTEKST:
{context}

PYTANIE: {question}

Odpowiedz zgodnie z zasadami. Pamietaj o cytowaniu zrodel."""


# Naglowki oznaczajace tresc NIEWDROZONA. Rozpoznawane mechanicznie, a nie
# pozostawione ocenie modelu.
#
# Powod: regula w prompcie ("sekcje typu Propozycja opisuj jako plan") nie
# zadzialala. Model przeczytal sekcje "Co jest propozycja, a nie moja decyzja"
# i zameldowal folder raw/ oraz przetwarzanie co 30 minut jako stan faktyczny.
# To ta sama lekcja co przy cyrylicy i przy pierwszej osobie: instrukcja tekstowa
# nie jest gwarancja. Znacznik doklejony do KAZDEGO fragmentu z osobna dziala
# lepiej niz jedna zasada na poczatku promptu, bo stoi w miejscu uzycia danych.
_POMYSL_RE = _re.compile(
    r"propozycj|do rozwazenia|do rozwazeni|swiadomie odlozon|nastepne kroki|"
    r"warte sprawdzenia|pomysl|plan(y|ow)?\b|kiedys|w przyszlosci|rozwazam|"
    r"nie moja decyzj|zamiast budowania",
    _re.IGNORECASE,
)


def czy_pomysl(heading: str) -> bool:
    """Czy naglowek sekcji zapowiada pomysl, a nie wdrozony stan."""
    return bool(_POMYSL_RE.search(bez_ogonkow_core(heading or "")))


def build_context(hits: list[Hit]) -> str:
    """Fragmenty numerowane, bez nazw plikow w widocznym miejscu.

    Wczesniej etykieta zawierala sciezke pliku, wiec model wklejal ja do odpowiedzi
    jako "[zrodlo: meta/o-mnie.md > Czego szukam]" - dlugie, powtarzalne i rozbijajace
    zdanie. Numer wystarczy modelowi do wskazania fragmentu, a mapowanie numer -> plik
    interfejs zna z pola `przypisy` i pokazuje pod odpowiedzia.

    Dystans zostaje w kontekscie, bo pomaga modelowi wazyc sprzeczne fragmenty.
    """
    parts = []
    for i, h in enumerate(hits, 1):
        naglowek = h.metadata.get("heading") or ""
        head = f" > {naglowek}" if naglowek else ""
        status = ""
        if czy_pomysl(naglowek):
            status = (
                "\n[STATUS TEGO FRAGMENTU: POMYSL / PLAN - NIE JEST WDROZONY.\n"
                " Nie wolno opisywac tego jako stanu faktycznego. Jesli uzywasz tej\n"
                " tresci, piszesz \"jest pomysl, zeby...\" albo \"rozwazasz...\", nigdy\n"
                " \"tak to dziala\".]"
            )
        parts.append(
            f"--- FRAGMENT [{i}] ({h.source}{head}, dystans {h.distance:.3f}) ---"
            f"{status}\n{h.text}"
        )
    return "\n\n".join(parts)


def przypisy(hits: list[Hit]) -> list[dict]:
    """Mapowanie numer -> plik, sekcja, dystans i podglad tresci.

    Fragmenty znalezione WYLACZNIE przez BM25 nie maja dystansu wektorowego -
    nosza sentinel 999.0, ktory blokuje im otwarcie bramki progu. To wartosc
    techniczna, nie pomiar, wiec do interfejsu idzie None plus flaga `lexykalny`.
    Pokazywanie "999.000" jako dystansu bylo mylace: sugerowalo skrajnie zle
    dopasowanie, podczas gdy fragment trafil tu przez dokladne slowo.
    """
    out = []
    for i, h in enumerate(hits, 1):
        tekst = " ".join(h.text.split())
        tylko_bm25 = h.distance >= 900
        out.append(
            {
                "n": i,
                "source": h.source,
                "heading": h.metadata.get("heading") or "",
                "distance": None if tylko_bm25 else round(h.distance, 4),
                "lexykalny": tylko_bm25,
                "pomysl": czy_pomysl(h.metadata.get("heading") or ""),
                "fragment": tekst[:400] + ("..." if len(tekst) > 400 else ""),
            }
        )
    return out


SYSTEM_HYBRYDA_Z_NOTATKAMI = """Rozmawiasz z uzytkownikiem i masz do dyspozycji fragmenty
jego notatek w sekcji KONTEKST. Notatki sa Twoim materialem, nie kagancem.

Roznica wobec trybu scisle notatkowego: tam wolno Ci bylo powiedziec wylacznie to, co
stoi w notatkach. Tutaj mozesz myslec. Mozesz laczyc fakty z notatek z wlasna wiedza,
wyciagac wnioski, nie zgadzac sie, zwracac uwage na sprzecznosc, dopytac.

Zasady:
1. Fakty wziete z notatek oznaczasz numerem fragmentu: [1], [2]. To jest obowiazkowe.
2. Wlasne wnioski, wiedze ogolna i opinie piszesz BEZ numeru. Interfejs pokaze je jako
   niepoparte notatka i tak ma byc - czytelnik ma widziec, gdzie konczy sie zapis,
   a zaczyna Twoje rozumowanie. Nie udajesz, ze wniosek jest cytatem.
3. Gdy notatki nie dotycza pytania, po prostu odpowiadasz z wlasnej wiedzy. Nie
   informujesz o tym osobnym zdaniem i nie tlumaczysz sie z braku notatek.
4. Nie streszczasz kontekstu na sile. Fragment, ktory nie dotyczy pytania, pomijasz
   w milczeniu - lepiej krotka trafna odpowiedz niz przeglad wszystkiego, co przyszlo.
4a. Naglowek sekcji mowi, jaki status ma jej tresc. "Propozycja", "Do rozwazenia",
   "Swiadomie odlozone", "Nastepne kroki" to POMYSLY, ktore nie zostaly wdrozone.
   Nigdy nie przedstawiasz ich jako stanu faktycznego - piszesz "jest pomysl, zeby...",
   a nie "tak to dziala".
5. Bierzesz pod uwage wczesniejsze wiadomosci w rozmowie. Zdanie w rodzaju "nastepnym
   razem wykorzystam to" odnosi sie do czegos, co padlo wczesniej - odczytujesz to
   z historii, a nie odpowiadasz, ze nie wiesz, o co chodzi.
6. Odpowiadasz zwiezle, po polsku, wylacznie alfabetem lacinskim.
7. Notatki sa pisane przez uzytkownika w pierwszej osobie. Ty nie jestes uzytkownikiem -
   zwracasz sie do niego w drugiej osobie.
"""


def build_messages(
    question: str,
    hits: list[Hit],
    historia: list[dict] | None = None,
    system: str | None = None,
) -> list[dict]:
    # Blok tozsamosci idzie takze tutaj, nie tylko na sciezke modelowa. Wlasnie ta,
    # notatkowa, produkowala "Wykorzystuje Cie Patryk Dabek" - bo notatki sa pisane
    # w pierwszej osobie i model przejmowal glos autora zamiast opowiadac o nim.
    # Historia idzie MIEDZY prompt systemowy a biezace pytanie, jako zwykle tury
    # rozmowy. Wklejanie jej do tresci pytania miesza dwie rzeczy: model traktowalby
    # wtedy poprzednie odpowiedzi jak material do cytowania na rowni z notatkami.
    wiadomosci = [
        {"role": "system", "content": blok_tozsamosci() + (system or SYSTEM_PROMPT)}
    ]
    if historia:
        wiadomosci += [
            {"role": w["role"], "content": w["content"]} for w in historia[-6:]
        ]
    wiadomosci.append(
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=build_context(hits) or "(brak fragmentow)",
                question=question,
            ),
        }
    )
    return wiadomosci


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Usuwa blok <think>...</think>, ktory generuja modele Qwen3.

    Zabezpieczenie: jesli po wycieciu nie zostalo NIC (bo blok byl niedomkniety -
    generowanie ucieto limitem num_predict), zwracamy tekst bez samych znacznikow,
    zamiast pustego stringa. Lepiej pokazac surowe rozumowanie niz nic.
    """
    # 1. usun kompletne bloki <think>...</think>
    cleaned = _THINK_RE.sub("", text).strip()

    # 2. zostal osierocony </think> (brak otwarcia) - wez to, co po ostatnim
    if "</think>" in cleaned:
        tail = cleaned.rsplit("</think>", 1)[1].strip()
        if tail:
            return tail
        cleaned = cleaned.rsplit("</think>", 1)[0].strip()

    # 3. zostal niedomkniety <think> - generowanie ucieto limitem num_predict.
    #    Wez to, co przed nim; jesli nic tam nie ma, oddaj sama tresc bez znacznika,
    #    zeby nigdy nie zwrocic pustego stringa.
    if "<think>" in cleaned:
        head = cleaned.split("<think>", 1)[0].strip()
        if head:
            return head
        cleaned = _THINK_TAG_RE.sub("", cleaned).strip()

    return _THINK_TAG_RE.sub("", cleaned).strip()


def _chat_payload(messages: list[dict], temperature: float, think) -> dict:
    # Rozumowanie zjada tokeny z tego samego limitu co odpowiedz. Przy niskim
    # NUM_PREDICT blok <think> bywa ucinany w polowie, a po jego wycieciu
    # zostaje pusty string - mielismy to przy 600. Dlatego przy wlaczonym
    # rozumowaniu limit idzie w gore, tym bardziej im wyzszy poziom.
    limit = NUM_PREDICT
    if think not in (False, None):
        mnoznik = {"low": 1.5, True: 2.0, "medium": 2.0, "high": 3.0, "max": 4.0}
        limit = int(NUM_PREDICT * mnoznik.get(think, 2.0))

    payload: dict = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": NUM_CTX,
            "num_predict": limit,
        },
    }
    if think is not None:
        payload["think"] = think
    return payload


# Ustawiane raz na proces: czy dana Ollama/model w ogole przyjmuje "think".
_think_supported: bool | None = None


_CYRYLICA_RE = re.compile(r"[Ѐ-ӿ]")


def ma_cyrylice(text: str) -> bool:
    """Wykrywa znaki cyrylicy w odpowiedzi.

    qwen3.5 jako model wielojezyczny sporadycznie wtraca slowo cyrylica w srodku
    polskiego zdania (zmierzone: 2 razy na 32 odpowiedzi). Instrukcja w prompcie
    systemowym tego NIE eliminuje - potrzebna jest kontrola po stronie kodu.
    """
    return bool(_CYRYLICA_RE.search(text))


# Notatki sa pisane w pierwszej osobie ("Szukam automatyzacji", "Odmowilem dwoch rol").
# Model, dostajac je jako kontekst, przejmuje te perspektywe i odpowiada jakby BYL
# uzytkownikiem: "Jestem Patrykiem Dabkiem, mieszkajacym w Katowicach".
#
# Regula w prompcie systemowym (punkt 8) tego nie wystarcza - dziala niekonsekwentnie,
# bo kontekst z notatek jest dluzszy i "glosniejszy" niz jedno zdanie instrukcji.
# Dokladnie ta sama sytuacja co z cyrylica: kontrola musi byc po stronie kodu.
_OSOBA_WZORCE = (
    re.compile(r"\b(moj|moja|moje|moim|mojego|mojej|moich|moimi|mna|mnie)\b"),
    re.compile(r"\b\w+(lem|lam)\b"),          # odmowilem, zbudowalem - 1 os. czasu przeszlego
    re.compile(r"\b(jestem|mam|wiem|mieszkam|szukam)\b"),
)


# Agent mowiacy o SOBIE uzywa pierwszej osoby calkiem poprawnie: "Jestem osobistym
# agentem wiedzy". To nie jest przejecie tozsamosci uzytkownika i nie wolno tego
# ponawiac - inaczej pytanie "czym jestes" zawsze kosztowaloby dwie generacje.
_AUTOPREZENTACJA_RE = re.compile(
    r"\bjestem\b[^.!?]{0,60}\b(agent|narzedzi|asystent|program|model)"
)


def ma_pierwsza_osobe(text: str) -> bool:
    """Czy model pisze jakby byl uzytkownikiem, zamiast mowic do niego.

    Wymagane sa DWA niezalezne sygnaly, nie jeden. Pojedyncze trafienie zdarza sie
    w cytacie albo w zwrocie o samym modelu ("nie mam tej informacji") i ponawianie
    odpowiedzi z tego powodu kosztowaloby wiecej, niz daje.
    """
    t = bez_ogonkow_core(text)
    if _AUTOPREZENTACJA_RE.search(t):
        return False
    return sum(1 for w in _OSOBA_WZORCE if w.search(t)) >= 2


def bez_ogonkow_core(s: str) -> str:
    return s.translate(_DIAKRYTYKI).lower()


def czy_odmowa(odpowiedz: str) -> bool:
    """Czy model odmowil odpowiedzi.

    Porownanie MUSI ignorowac ogonki. REFUSAL jest zapisany bez nich ("znalazlem"),
    a model pisze poprawna polszczyzna ("znalazłem") - naiwne porownanie prefiksow
    raz juz zaniżylo pomiar z 8/8 do 2/8 i o maly wlos nie doprowadzilo do cofniecia
    dobrej zmiany w architekturze.
    """
    return bez_ogonkow_core(odpowiedz).strip().startswith(bez_ogonkow_core(REFUSAL)[:22])


def chat(
    messages: list[dict],
    retries: int = 2,
    temperature: float = 0.1,
    wymus_druga_osobe: bool = False,
    think=None,
) -> str:
    """Wywoluje model. Przy wykryciu cyrylicy ponawia raz z jawnym przypomnieniem.

    `wymus_druga_osobe` wlaczamy TYLKO na sciezce z notatkami. W trybie rozmowy
    pierwsza osoba jest poprawna - tam model mowi sam o sobie.
    """
    odpowiedz = _chat_raw(messages, retries, temperature, poziom_think=think)

    if wymus_druga_osobe and ma_pierwsza_osobe(odpowiedz):
        poprawka = messages + [
            {"role": "assistant", "content": odpowiedz},
            {
                "role": "user",
                "content": (
                    "Napisales te odpowiedz w pierwszej osobie, jakbys byl mna. "
                    "Notatki sa moje i sa pisane z mojej perspektywy, ale to JA "
                    "zadaje pytanie, a Ty na nie odpowiadasz. Napisz to samo jeszcze "
                    "raz, identycznie co do tresci i zrodel, ale zwracajac sie do mnie "
                    "w drugiej osobie: 'Jestes', 'Mieszkasz', 'Szukasz', 'Twoim celem'. "
                    "Nie dodawaj komentarza o poprawce."
                ),
            },
        ]
        druga = _chat_raw(poprawka, retries=0, temperature=temperature, poziom_think=think)
        if druga.strip() and not ma_pierwsza_osobe(druga) and not ma_cyrylice(druga):
            return druga
        # nie udalo sie - pierwsza wersja jest merytorycznie poprawna, tylko zle
        # sformulowana, wiec lepsza niz brak odpowiedzi

    if ma_cyrylice(odpowiedz):
        poprawka = messages + [
            {"role": "assistant", "content": odpowiedz},
            {
                "role": "user",
                "content": (
                    "Twoja odpowiedz zawiera znaki cyrylicy. Napisz ja ponownie, "
                    "identycznie co do tresci i zrodel, ale wylacznie alfabetem "
                    "lacinskim (polskim). Nie dodawaj komentarza o poprawce."
                ),
            },
        ]
        druga = _chat_raw(poprawka, retries=0, temperature=temperature, poziom_think=think)
        if druga.strip() and not ma_cyrylice(druga):
            return druga
        # ponowna proba tez zawiodla - oddajemy pierwsza wersje, lepsza niz nic
    return odpowiedz


def _chat_raw(
    messages: list[dict],
    retries: int = 2,
    temperature: float = 0.1,
    poziom_think=None,
) -> str:
    global _think_supported

    think = _think_wartosc(poziom_think) if _think_supported is not False else None

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=_chat_payload(messages, temperature, think),
                timeout=CHAT_TIMEOUT,
            )
            # Model albo wersja Ollamy nie wspiera parametru "think" - cofamy sie.
            if r.status_code == 400 and think is not None:
                _think_supported = False
                think = None
                continue
            r.raise_for_status()
            if think is False:
                _think_supported = True
            return strip_thinking(r.json()["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(
        f"Ollama nie odpowiedziala ({OLLAMA_URL}, model {CHAT_MODEL}). Ostatni blad: {last_err}"
    )


def zapytanie_z_kontekstem(question: str, historia: list[dict] | None) -> str:
    """Skleja krotkie doprecyzowanie z poprzednim pytaniem, zeby mialo czego szukac.

    "nastepnym razem wykorzystam to" samo w sobie nie ma tresci - wektor takiego
    zdania nie wskazuje na nic. Doklejenie poprzedniej wypowiedzi uzytkownika daje
    wyszukiwaniu punkt zaczepienia.

    To heurystyka, nie przepisywanie zapytania modelem: dziala tylko dla krotkich
    wypowiedzi, bo dlugie pytanie ma dosc wlasnej tresci, a doklejanie do niego
    historii tylko rozmyloby wektor.
    """
    if not historia or len(question) > 60:
        return question
    poprzednie = [w["content"] for w in historia if w.get("role") == "user"]
    if not poprzednie:
        return question
    return f"{poprzednie[-1]} {question}"


def answer(
    question: str,
    k: int = TOP_K,
    max_distance: float | None = MAX_DISTANCE,
    category: str | None = None,
    historia: list[dict] | None = None,
    system: str | None = None,
) -> dict:
    """Pelny przebieg RAG: retrieval -> prompt -> generacja."""
    t0 = time.perf_counter()
    zapytanie = zapytanie_z_kontekstem(question, historia)
    hits, all_hits = search(zapytanie, k=k, max_distance=max_distance, category=category)
    t_retrieval = time.perf_counter() - t0

    if not hits:
        return {
            "question": question,
            "answer": REFUSAL,
            "sources": [],
            "przypisy": [],
            "in_scope": False,
            "hits": [h.to_dict() for h in all_hits],
            "retrieval_s": round(t_retrieval, 3),
            "generation_s": 0.0,
        }

    t1 = time.perf_counter()
    text = chat(
        build_messages(question, hits, historia=historia, system=system),
        wymus_druga_osobe=True,
    )
    t_generation = time.perf_counter() - t1

    if not text.strip():
        text = (
            "(model zwrocil pusta odpowiedz - najczestsza przyczyna to za niski "
            "NUM_PREDICT przy wlaczonym trybie rozumowania)"
        )

    # unikalne zrodla z zachowaniem kolejnosci rankingu
    sources: list[str] = []
    for h in hits:
        if h.source not in sources:
            sources.append(h.source)

    return {
        "question": question,
        "answer": text,
        "sources": sources,
        "przypisy": przypisy(hits),
        "in_scope": True,
        "hits": [h.to_dict() for h in hits],
        "retrieval_s": round(t_retrieval, 3),
        "generation_s": round(t_generation, 3),
    }


# --------------------------------------------------------------------------
# Tryb rozmowy - zwykly czat z modelem, bez notatek
# --------------------------------------------------------------------------

# ==========================================================================
# TOZSAMOSC - wspolna dla wszystkich trybow
# ==========================================================================
#
# Bez tego agent rozpadal sie na dwa byty. Zapytany "kto cie wykorzystuje" na
# sciezce z notatkami mowil "jestem Twoim osobistym agentem wiedzy", a na sciezce
# modelowej - "jestem sztuczna inteligencja stworzona przez Google". Drugie zdanie
# jest w dodatku nieprawdziwe: pod spodem stoi qwen, nie model Google.
#
# Przyczyna: prompt hybrydowy opisywal ZADANIE ("odpowiedz z wiedzy ogolnej"),
# ale nie mowil, KIM jest odpowiadajacy. Model wypelnial luke tym, co pamieta
# z treningu - a modele wytrenowane na danych z internetu maja tam pomieszane
# tozsamosci innych asystentow.
#
# Zrodlem prawdy jest notatka w vaulcie, nie stala w kodzie. Dzieki temu zmiana
# opisu agenta to edycja notatki, a nie zmiana kodu i restart - i ta sama tresc
# obsluguje zarowno pytania "z notatek", jak i tryb modelowy.
TOZSAMOSC_ZRODLO = "czym-jest-ten-agent.md"

# Uzywana, gdy notatki jeszcze nie ma w bazie. Krotka, ale musi wystarczyc,
# zeby model nie zaczal sie przedstawiac jako produkt obcej firmy.
TOZSAMOSC_ZAPASOWA = """Jestes Personal Knowledge Agent - lokalnym narzedziem, ktore
odpowiada na pytania uzytkownika na podstawie jego wlasnych notatek. Dzialasz w calosci
na jego komputerze. Nie jestes produktem Google, OpenAI, Anthropic ani zadnej innej firmy
i nigdy tak sie nie przedstawiasz. Pod spodem pracuje lokalny model jezykowy uruchomiony
przez Ollame, ale Twoja tozsamoscia jest ten agent, nie model."""

_tozsamosc_cache: str | None = None


def tozsamosc() -> str:
    """Opis agenta wczytany z notatki systemowej, z zapasem w kodzie."""
    global _tozsamosc_cache
    if _tozsamosc_cache is not None:
        return _tozsamosc_cache

    tekst = ""
    try:
        r = get_collection(create=False).get(
            where={"source": TOZSAMOSC_ZRODLO},
            include=["documents", "metadatas"],
        )
        pary = list(zip(r.get("metadatas") or [], r.get("documents") or []))
        pary.sort(key=lambda p: (p[0] or {}).get("chunk_index", 0))
        tekst = "\n\n".join(d for _, d in pary if d)
    except Exception:  # noqa: BLE001 - brak notatki nie moze wywrocic odpowiadania
        pass

    _tozsamosc_cache = (tekst.strip() or TOZSAMOSC_ZAPASOWA)[:2500]
    return _tozsamosc_cache


def blok_tozsamosci() -> str:
    return (
        "KIM JESTES (obowiazuje we wszystkich trybach, takze gdy odpowiadasz "
        "z wiedzy ogolnej):\n"
        f"{tozsamosc()}\n\n"
        "Nigdy nie przedstawiasz sie jako asystent Google, OpenAI, Anthropic ani "
        "innej firmy. Zapytany, czym jestes albo kto Cie uzywa, odpowiadasz zgodnie "
        "z powyzszym opisem - niezaleznie od tego, czy odpowiedz pochodzi z notatek, "
        "czy z Twojej wiedzy ogolnej.\n\n"
        "KONTRAKT ZAIMKOW - obowiazuje bezwzglednie:\n"
        "  \"ja\", \"mnie\", \"moje\"  = agent, czyli Ty\n"
        "  \"Ty\", \"Ciebie\", \"Twoje\" = uzytkownik, czyli wlasciciel notatek\n\n"
        "Notatki sa pisane przez uzytkownika w pierwszej osobie. To NIE jest Twoj glos "
        "- czytasz cudzy zapis i opowiadasz o nim wlascicielowi.\n\n"
        "Pytanie \"kto Cie wykorzystuje\" dotyczy CIEBIE, agenta. Odpowiada na nie "
        "uzytkownik jako sprawca:\n"
        "  DOBRZE: \"Wykorzystujesz mnie Ty - do przeszukiwania wlasnych notatek.\"\n"
        "  ZLE:    \"Wykorzystuje Cie Patryk Dabek\" - to zdanie robi z uzytkownika agenta.\n"
        "  ZLE:    \"Jestem Patrykiem Dabkiem\" - to przejecie jego tozsamosci.\n\n"
        "Odpowiadasz tylko na zadane pytanie. Nie doklejasz faktow o uzytkowniku, "
        "ktore akurat znalazly sie w kontekscie, a nie wynikaja z pytania - wzrostu, "
        "miejsca zamieszkania, planow zawodowych.\n\n"
        "CO POTRAFISZ, A CZEGO NIE - nie wolno Ci tego zmyslac:\n"
        "  Potrafisz: przeszukiwac notatki i odpowiadac na ich podstawie.\n"
        "  NIE potrafisz: tworzyc, edytowac, nadpisywac ani usuwac notatek. Nie masz\n"
        "  dostepu do internetu. Nie wysylasz maili, nie umawiasz spotkan, niczego nie\n"
        "  uruchamiasz i nie pamietasz niczego miedzy rozmowami poza tym, co jest\n"
        "  w notatkach.\n\n"
        "Notatki dopisuje SAM uzytkownik, przyciskiem w interfejsie. Tekst trafia na "
        "koniec pliku inbox/surowe-RRRR-MM-DD.md - dopisywany, nigdy nadpisywany.\n\n"
        "Notatki opisuja czynnosci uzytkownika w pierwszej osobie: \"zaktualizowalem\", "
        "\"zapisalem\", \"zmienilem\". To sa JEGO dzialania, nie Twoje. Zapytany, czy cos "
        "zrobiles - zmieniles notatke, zapisales cos, wyslales - odpowiadasz, ze nie "
        "masz takiej mozliwosci. NIGDY nie opisujesz jako wlasnego dzialania czegos, "
        "o czym przeczytales w notatce.\n\n"
    )


CHAT_SYSTEM = """Jestes pomocnym asystentem rozmawiajacym po polsku.

Zasady:
1. Odpowiadasz zwiezle i konkretnie. Bez zbednych wstepow i podsumowan.
2. W tym trybie NIE masz dostepu do notatek uzytkownika. Jesli pytanie wyraznie
   dotyczy jego wlasnych notatek, decyzji albo projektow - powiedz, zeby przelaczyl
   sie na tryb "Notatki", zamiast zgadywac.
3. Gdy czegos nie wiesz albo nie jestes pewien - mowisz to wprost.
4. Piszesz wylacznie alfabetem lacinskim. Nie uzywasz cyrylicy.
"""


def rozmowa(historia: list[dict], temperature: float = 0.4) -> str:
    """Zwykla rozmowa z modelem, bez zadnego kontekstu z notatek.

    historia: lista {"role": "user"|"assistant", "content": "..."} w kolejnosci
    chronologicznej. Prompt systemowy dokladany jest tutaj, nie po stronie klienta.

    Wyzsza temperatura niz w RAG (0.4 zamiast 0.1), bo w swobodnej rozmowie
    sztywnosc szkodzi, a nie ma tu ryzyka przeklamania zrodel.
    """
    wiadomosci = [
        {"role": "system", "content": blok_tozsamosci() + CHAT_SYSTEM}
    ] + list(historia)
    return chat(wiadomosci, temperature=temperature)


HYBRYDA_SYSTEM = """Jestes asystentem osobistej bazy wiedzy uzytkownika.

W notatkach uzytkownika NIE MA odpowiedzi na to pytanie - sprawdzono to przed
przekazaniem Ci go. Odpowiadasz wiec z wlasnej wiedzy ogolnej.

Zasady:
1. Zaczynasz odpowiedz od zdania: "W notatkach tego nie ma, odpowiadam z wiedzy ogolnej."
2. Potem odpowiadasz normalnie, zwiezle i konkretnie.
3. Nie udajesz, ze cytujesz notatki. Nie wymyslasz nazw plikow ani zrodel.
4. Jesli pytanie dotyczy osobistych faktow uzytkownika, ktorych nie mozesz znac
   (jego decyzji, projektow, ustalen) - mowisz, ze tego nie wiesz i warto to zapisac
   jako notatke, zamiast zgadywac.
5. Zwracasz sie do uzytkownika w drugiej osobie. Nie piszesz o nim "jestem", "moje".
6. Piszesz wylacznie alfabetem lacinskim. Nie uzywasz cyrylicy.
7. WYJATEK od zasady 1: jesli pytanie dotyczy Ciebie samego - czym jestes, kto Cie
   uzywa, co potrafisz, jak dzialasz - NIE zaczynasz od "W notatkach tego nie ma".
   Odpowiadasz wprost z opisu w sekcji "KIM JESTES" powyzej. To nie jest wiedza
   ogolna modelu, tylko Twoja wlasna definicja.
"""


def answer_hybrid(
    question: str,
    k: int = TOP_K,
    max_distance: float | None = MAX_DISTANCE,
    category: str | None = None,
    historia: list[dict] | None = None,
) -> dict:
    """Najpierw notatki. Gdy ich brak - wiedza modelu, ale JAWNIE oznaczona.

    Roznica wobec answer(): zamiast odmowy uzytkownik dostaje odpowiedz z wiedzy
    ogolnej, wyraznie opisana jako pochodzaca spoza notatek. Pole `zrodlo_wiedzy`
    mowi, ktora sciezka zadzialala - i to ono, a nie tresc, powinno decydowac
    o tym, czy interfejs pokazuje ostrzezenie.

    Gwarancja braku konfabulacji na notatkach zostaje: gdy prog przepuszcza,
    dziala zwykly RAG z zakazem wychodzenia poza kontekst. Model dostaje
    swobode wylacznie wtedy, gdy jawnie wiadomo, ze notatek nie ma.
    """
    # Inny prompt niz w trybie scislym. Tam notatki sa jedynym dozwolonym zrodlem,
    # tu sa materialem do myslenia - model moze wyciagac wnioski i dopowiadac z wiedzy
    # ogolnej, byle nie podszywal ich pod cytat. Rozroznienie widac w interfejsie:
    # zdania z numerem sa poparte notatka, reszta jest podkreslona.
    wynik = answer(
        question,
        k=k,
        max_distance=max_distance,
        category=category,
        historia=historia,
        system=SYSTEM_HYBRYDA_Z_NOTATKAMI,
    )

    # Dwa niezalezne powody, dla ktorych notatki moga nie wystarczyc:
    #   1. prog nie przepuscil        -> in_scope == False
    #   2. prog przepuscil, ale model uznal, ze w kontekscie nie ma odpowiedzi
    #
    # Sprawdzanie samego in_scope dawalo zielona plakietke "z Twoich notatek"
    # nad zdaniem "Nie znalazlem tego w notatkach" - plakietka klamala.
    if wynik["in_scope"] and not czy_odmowa(wynik["answer"]):
        wynik["zrodlo_wiedzy"] = "notatki"
        return wynik

    t0 = time.perf_counter()
    wiadomosci = [{"role": "system", "content": blok_tozsamosci() + HYBRYDA_SYSTEM}]
    if historia:
        wiadomosci += [{"role": w["role"], "content": w["content"]} for w in historia[-6:]]
    wiadomosci.append({"role": "user", "content": question})
    tekst = chat(wiadomosci, temperature=0.3)
    wynik["answer"] = tekst
    wynik["zrodlo_wiedzy"] = "model"
    wynik["generation_s"] = round(time.perf_counter() - t0, 3)
    return wynik


# --------------------------------------------------------------------------
# Stan modeli - swiatla i przelaczanie
# --------------------------------------------------------------------------
#
# Powod istnienia tego bloku jest pomiarowy, nie kosmetyczny. Przy 8 GB VRAM
# bge-m3 (664 MB) i qwen3.5:9b (6,4 GB) nie mieszcza sie wygodnie razem - gdy oba
# sa rezydentne, Ollama zaczyna je przerzucac miedzy VRAM a RAM przy kazdym
# zapytaniu i mediana odpowiedzi skacze z ~6 s do 29 s. Do tej pory ratunkiem bylo
# `ollama stop` w konsoli. Teraz widac to na ekranie i da sie klikniac.

def stan_modeli() -> dict:
    """Trzy stany dla kazdego modelu: zaladowany / dostepny / brak.

    zielone  - model siedzi w pamieci, odpowie natychmiast
    zolte    - model jest pobrany, ale trzeba go wczytac (pierwsze pytanie wolniejsze)
    czerwone - modelu nie ma, trzeba `ollama pull`
    """
    out: dict = {"ollama": "ok", "modele": []}
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        pobrane = {m.get("name", "") for m in tags.get("models", [])}
    except Exception as exc:  # noqa: BLE001
        return {"ollama": f"blad: {exc}", "modele": []}

    zaladowane: dict[str, dict] = {}
    try:
        ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json()
        for m in ps.get("models", []):
            zaladowane[m.get("name", "")] = m
    except Exception:  # noqa: BLE001 - starsze Ollamy nie maja /api/ps
        pass

    def dopasuj(nazwa: str, zbior) -> str | None:
        """Ollama zwraca 'bge-m3:latest' tam, gdzie w configu jest 'bge-m3'."""
        baza = nazwa.split(":")[0]
        for n in zbior:
            if n == nazwa or n.split(":")[0] == baza:
                return n
        return None

    for rola, nazwa in (("embedding", EMBED_MODEL), ("generowanie", CHAT_MODEL)):
        pelna = dopasuj(nazwa, pobrane)
        w_pamieci = dopasuj(nazwa, zaladowane.keys()) if pelna else None
        info = zaladowane.get(w_pamieci or "", {})

        if not pelna:
            swiatlo = "czerwone"
        elif w_pamieci:
            swiatlo = "zielone"
        else:
            swiatlo = "zolte"

        vram = info.get("size_vram") or 0
        out["modele"].append(
            {
                "rola": rola,
                "nazwa": nazwa,
                "pelna_nazwa": pelna or nazwa,
                "swiatlo": swiatlo,
                "vram_mb": round(vram / 1024 / 1024) if vram else 0,
                "wygasa": info.get("expires_at", ""),
            }
        )

    suma = sum(m["vram_mb"] for m in out["modele"])
    out["vram_mb_razem"] = suma
    # Prog ostrzegawczy dobrany pod 8 GB karty - powyzej ~6.5 GB zaczyna sie
    # przerzucanie warstw i czasy odpowiedzi rosna kilkukrotnie.
    out["ostrzezenie_vram"] = suma > 6500
    return out


def przelacz_model(nazwa: str, wlacz: bool) -> dict:
    """Wczytuje model do pamieci albo go z niej zwalnia.

    Ollama nie ma osobnego endpointu do zaladowania i zwolnienia - robi sie to
    pustym zapytaniem z parametrem keep_alive. Zero zwalnia natychmiast.
    """
    czy_embed = nazwa.split(":")[0] == EMBED_MODEL.split(":")[0]
    keep = "30m" if wlacz else 0
    try:
        if czy_embed:
            r = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": nazwa, "input": [""] if wlacz else [], "keep_alive": keep},
                timeout=180,
            )
        else:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": nazwa, "prompt": "", "keep_alive": keep, "stream": False},
                timeout=180,
            )
        r.raise_for_status()
        return {"ok": True, "nazwa": nazwa, "wlaczony": wlacz}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "nazwa": nazwa, "blad": str(exc)}


def health() -> dict:
    """Szybki test: czy Ollama zyje, czy modele sa, ile chunkow w bazie."""
    out: dict = {"ollama_url": OLLAMA_URL, "chroma_path": CHROMA_PATH}
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json()
        names = [m.get("name", "") for m in tags.get("models", [])]
        out["ollama"] = "ok"
        out["models"] = names
        out["embed_model_present"] = any(n.split(":")[0] == EMBED_MODEL.split(":")[0] for n in names)
        out["chat_model_present"] = any(n.split(":")[0] == CHAT_MODEL.split(":")[0] for n in names)
    except Exception as exc:  # noqa: BLE001
        out["ollama"] = f"blad: {exc}"
    try:
        out["chunks"] = get_collection().count()
    except Exception as exc:  # noqa: BLE001
        out["chunks"] = f"blad: {exc}"
    out["num_ctx"] = NUM_CTX
    out["num_predict"] = NUM_PREDICT
    out["disable_thinking"] = DISABLE_THINKING
    out["use_bm25"] = USE_BM25
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(health(), indent=2, ensure_ascii=False))
