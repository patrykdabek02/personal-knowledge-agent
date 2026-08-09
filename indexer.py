"""
indexer.py - buduje lokalny indeks wektorowy z folderu notatek (.md / .txt).

Uzycie:
    python indexer.py --path "C:\\Users\\patry\\Notatki"
    python indexer.py --path "..." --rebuild        # pelna przebudowa od zera
    python indexer.py --path "..." --dry-run        # pokaz co by zrobil, nic nie zapisuj

Cechy:
  - chunkowanie swiadome markdowna (najpierw po naglowkach, potem oknem slow z overlapem)
  - kazdy chunk dostaje prefiks "tytul > naglowek", co wyraznie poprawia trafnosc wyszukiwania
  - inkrementalny re-index: przetwarza tylko pliki, ktore sie zmienily (hash SHA-256 tresci)
  - usuniete pliki sa czyszczone z bazy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import core

# --------------------------------------------------------------------------
# Parametry chunkowania
# --------------------------------------------------------------------------

CHUNK_WORDS = 350          # docelowy rozmiar fragmentu w slowach
CHUNK_OVERLAP = 60         # zakladka miedzy fragmentami (zeby nie ciac mysli w pol)
MIN_CHUNK_CHARS = 80       # ponizej tego fragment jest smieciem, pomijamy
EMBED_BATCH = 32           # ile fragmentow naraz wysylamy do Ollama

EXTENSIONS = {".md", ".txt", ".markdown"}
SKIP_DIRS = {".git", ".obsidian", ".trash", "node_modules", "__pycache__", ".venv", "chroma_db"}

MANIFEST_NAME = "index_manifest.json"


# --------------------------------------------------------------------------
# Parsowanie plikow
# --------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Lekki parser YAML frontmatter (bez zaleznosci od pyyaml).

    Obsluguje plaskie pary klucz: wartosc - w zupelnosci wystarczy do metadanych.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    raw = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].replace('"', "").replace("'", "").strip()
        if value:
            fm[key.strip()] = value
    return fm, body


def split_sections(body: str) -> list[tuple[str, str]]:
    """Dzieli tresc na sekcje wg naglowkow markdown. Zwraca [(naglowek, tekst)]."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    in_code_block = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            buf.append(line)
            continue
        m = HEADING_RE.match(line) if not in_code_block else None
        if m:
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))

    sections = [(h, t) for h, t in sections if t.strip()]
    return merge_short_sections(sections)


def merge_short_sections(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Doklej sekcje krotsze niz MIN_CHUNK_CHARS do sasiedniej.

    Bez tego krotkie a kluczowe sekcje ("## Decyzja\\n\\nWybralem Chroma.")
    wypadaja z indeksu i agent nie ma czego znalezc.
    """
    if not sections:
        return []

    merged: list[list[str]] = []
    for heading, text in sections:
        if len(text.strip()) < MIN_CHUNK_CHARS and merged:
            # doklej do poprzedniej sekcji, zachowujac naglowek w tresci
            prefix = f"{heading}: " if heading else ""
            merged[-1][1] = f"{merged[-1][1]}\n\n{prefix}{text.strip()}"
        else:
            merged.append([heading, text.strip()])

    # jesli pierwsza sekcja byla za krotka i nie miala do czego sie dokleic,
    # doklej ja do nastepnej
    if len(merged) > 1 and len(merged[0][1]) < MIN_CHUNK_CHARS:
        head, text = merged.pop(0)
        prefix = f"{head}: " if head else ""
        merged[0][1] = f"{prefix}{text}\n\n{merged[0][1]}"

    return [(h, t) for h, t in merged if t.strip()]


def window_split(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Dzieli dlugi tekst na okna slow z zakladka."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if not piece:
            break
        out.append(" ".join(piece))
        if i + size >= len(words):
            break
    return out


# --------------------------------------------------------------------------
# Contextual Retrieval
# --------------------------------------------------------------------------
#
# Technika z artykulu Anthropic (wrzesien 2024). Problem, ktory rozwiazuje,
# wystapil u nas doslownie: fragment "Widelki 7 500 - 9 000, podloga 7 000"
# nie mowi, DLA JAKIEJ ROLI te kwoty obowiazuja. Agent dwa razy podal je jako
# widelki dla stazu part-time, cytujac zrodlo - retrieval trafil, cytowanie bylo
# uczciwe, a rada nie do zastosowania.
#
# Rozwiazanie: przed embedowaniem model dopisuje do fragmentu jedno-dwa zdania
# osadzajace go w calej notatce. Tresc notatki sie NIE zmienia - zmienia sie to,
# co trafia do bazy.
#
# Zmierzone przez Anthropic: samo kontekstowanie -35% nieudanych trafien,
# razem z BM25 -49%, z rerankerem -67%. My mamy juz BM25.
#
# Uwaga na koszt: Anthropic uzywa cache'u promptow, zeby nie placic za caly
# dokument przy kazdym fragmencie. Ollama tego nie ma, ale przy notatkach
# rzedu kilku kB i ~124 fragmentach to kilka minut jednorazowo.

KONTEKST_PROMPT = """<dokument>
{dokument}
</dokument>

Oto fragment tego dokumentu, ktory chcemy osadzic w kontekscie calosci:
<fragment>
{fragment}
</fragment>

Napisz jedno lub dwa krotkie zdania, ktore umiejscawiaja ten fragment w calym
dokumencie - tak, zeby wyszukiwarka mogla go trafnie znalezc, a czytajacy
wiedzial, CZEGO on dotyczy i W JAKIM ZAKRESIE obowiazuje.

Zawrzyj zakres stosowania, jesli wynika z dokumentu: jakiego rodzaju roli,
sytuacji albo okresu dotycza podane liczby i ustalenia.

NAJWAZNIEJSZE OGRANICZENIE: uzywasz WYLACZNIE informacji, ktore stoja w tym
dokumencie. Nie dodajesz typow zatrudnienia, okresow, form wspolpracy ani
warunkow, ktorych w nim nie ma. Jesli dokument nie mowi, czego dotycza liczby -
piszesz o czym jest fragment i na tym konczysz, zamiast zgadywac zakres.

Dopisanie zakresu, ktorego w dokumencie nie ma, jest gorsze niz brak kontekstu:
fragment zacznie byc znajdowany przy pytaniach, ktorych NIE dotyczy.

Odpowiedz WYLACZNIE tym kontekstem, bez wstepu, bez powtarzania fragmentu.
Jedno lub dwa zdania. Po polsku, alfabetem lacinskim."""


def zbuduj_kontekst(dokument: str, fragment: str) -> str:
    """Jedno-dwa zdania osadzajace fragment w calym dokumencie."""
    # Dokument obcinamy, bo przy dlugiej notatce prompt zjadalby cale okno
    # i wypychal z niego sam fragment.
    if len(dokument) > 12000:
        dokument = dokument[:12000] + "\n[...]"
    try:
        kontekst = core.chat(
            [{"role": "user",
              "content": KONTEKST_PROMPT.format(dokument=dokument, fragment=fragment)}],
            temperature=0.0,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - brak kontekstu jest lepszy niz brak indeksu
        print(f"    [kontekst nieudany: {exc}]")
        return ""
    # Model bywa gadatliwy mimo instrukcji; trzy zdania wystarcza az nadto.
    return " ".join(kontekst.split())[:400]


def build_chunks(rel_path: str, text: str, kontekstowo: bool = False) -> list[dict]:
    """Zamienia plik w liste fragmentow gotowych do embeddingu.

    Przy kontekstowo=True kazdy fragment dostaje dopisane zdanie osadzajace go
    w calej notatce. Idzie ono do pola `document`, czyli do embeddingu ORAZ do
    indeksu BM25 - u Anthropic to sa dwie osobne techniki (Contextual Embeddings
    i Contextual BM25), u nas wychodza za darmo razem, bo BM25 czyta dokumenty
    z tej samej kolekcji.

    Oryginalna tresc ladu je w metadanych, zeby dalo sie ja pokazac w przypisach
    bez doklejonego kontekstu.
    """
    fm, body = split_frontmatter(text)
    path = Path(rel_path)
    title = fm.get("title") or path.stem
    category = path.parts[0] if len(path.parts) > 1 else "root"

    chunks: list[dict] = []
    index = 0
    for heading, section in split_sections(body):
        for piece in window_split(section):
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            # Prefiks kontekstowy - fragment "wie", z czego pochodzi.
            prefix = f"{title} > {heading}" if heading else title
            tresc = piece.strip()

            kontekst = ""
            if kontekstowo:
                print(f"    kontekst {index + 1}...", end="\r", flush=True)
                kontekst = zbuduj_kontekst(body, tresc)

            document = f"{prefix}\n\n" + (f"{kontekst}\n\n{tresc}" if kontekst else tresc)

            chunks.append(
                {
                    "id": f"{rel_path}::{index}",
                    "document": document,
                    "metadata": {
                        "rel_path": rel_path,
                        "source": path.name,
                        "title": title,
                        "heading": heading,
                        "category": category,
                        "chunk_index": index,
                        "tags": fm.get("tags", ""),
                        "kontekst": kontekst,
                        "tekst_oryginalny": tresc[:1500],
                    },
                }
            )
            index += 1
    return chunks


# --------------------------------------------------------------------------
# Skanowanie i indeksowanie
# --------------------------------------------------------------------------

def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Indeksuje notatki do lokalnej bazy Chroma.")
    ap.add_argument("--path", required=True, help="folder z notatkami (.md / .txt)")
    ap.add_argument("--rebuild", action="store_true", help="wyczysc kolekcje i zbuduj od zera")
    ap.add_argument("--dry-run", action="store_true", help="policz fragmenty, nic nie zapisuj")
    ap.add_argument("--kontekst", action="store_true",
                    help="Contextual Retrieval: model dopisuje kontekst do kazdego fragmentu "
                         "przed embedowaniem (wolniejsze, wymaga --rebuild)")
    args = ap.parse_args()

    if args.kontekst and not args.rebuild:
        # Bez przebudowy powstalaby baza mieszana: czesc fragmentow z kontekstem,
        # czesc bez. Dystanse przestalyby byc porownywalne miedzy soba, a prog
        # odciecia straciłby sens.
        print("BLAD: --kontekst wymaga --rebuild (zmienia sie tresc embedowana).")
        return 1

    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"BLAD: '{root}' nie jest folderem.")
        return 1

    manifest_path = Path(core.CHROMA_PATH).parent / MANIFEST_NAME
    manifest = {} if args.rebuild else load_manifest(manifest_path)

    if args.dry_run:
        collection = None
    else:
        collection = core.get_collection()
        if args.rebuild:
            import chromadb

            client = chromadb.PersistentClient(path=core.CHROMA_PATH)
            try:
                client.delete_collection(core.COLLECTION_NAME)
            except Exception:  # noqa: BLE001 - kolekcja mogla nie istniec
                pass
            collection = core.get_collection()
            print("Kolekcja wyczyszczona (--rebuild).")

    files = list(iter_files(root))
    print(f"Znaleziono {len(files)} plikow w {root}\n")

    stats = {"nowe": 0, "zmienione": 0, "bez_zmian": 0, "puste": 0, "fragmenty": 0}
    seen: set[str] = set()
    started = time.perf_counter()

    for path in files:
        rel = path.relative_to(root).as_posix()
        seen.add(rel)
        # utf-8-sig, a nie utf-8: Notatnik i PowerShell zapisuja UTF-8 z BOM.
        # Przy zwyklym utf-8 BOM zostaje jako ﻿ na poczatku pliku i psuje
        # wykrywanie frontmattera (text.startswith("---") zwraca False).
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig", errors="replace")

        digest = file_hash(text)
        previous = manifest.get(rel, {}).get("hash")
        if previous == digest and not args.rebuild:
            stats["bez_zmian"] += 1
            continue

        chunks = build_chunks(rel, text, kontekstowo=args.kontekst)
        if not chunks:
            stats["puste"] += 1
            print(f"  pominiety (brak tresci): {rel}")
            continue

        status = "zmieniony" if previous else "nowy"
        stats["zmienione" if previous else "nowe"] += 1
        stats["fragmenty"] += len(chunks)
        print(f"  [{status}] {rel} -> {len(chunks)} fragmentow")

        if args.dry_run:
            continue

        # Kasujemy stare fragmenty tego pliku (liczba chunkow mogla sie zmienic).
        if previous:
            try:
                collection.delete(where={"rel_path": rel})
            except Exception as exc:  # noqa: BLE001
                print(f"    ostrzezenie: nie udalo sie usunac starych fragmentow: {exc}")

        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            vectors = core.embed([c["document"] for c in batch])
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["document"] for c in batch],
                embeddings=vectors,
                metadatas=[c["metadata"] for c in batch],
            )

        manifest[rel] = {"hash": digest, "chunks": len(chunks), "indexed_at": int(time.time())}

    # Pliki usuniete z dysku - czyscimy z bazy.
    removed = [rel for rel in manifest if rel not in seen]
    for rel in removed:
        print(f"  [usuniety] {rel}")
        if not args.dry_run:
            try:
                collection.delete(where={"rel_path": rel})
            except Exception as exc:  # noqa: BLE001
                print(f"    ostrzezenie: {exc}")
        manifest.pop(rel, None)

    elapsed = time.perf_counter() - started

    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print(f"  nowe pliki        : {stats['nowe']}")
    print(f"  zmienione pliki   : {stats['zmienione']}")
    print(f"  bez zmian         : {stats['bez_zmian']}")
    print(f"  puste / pominiete : {stats['puste']}")
    print(f"  usuniete z bazy   : {len(removed)}")
    print(f"  nowe fragmenty    : {stats['fragmenty']}")
    print(f"  czas              : {elapsed:.1f} s")

    if not args.dry_run:
        save_manifest(manifest_path, manifest)
        total = collection.count()
        print(f"  fragmentow w bazie: {total}")

        if total:
            sample = collection.get(limit=2, include=["metadatas", "documents"])
            print("\nPrzykladowe metadane:")
            for meta, doc in zip(sample["metadatas"], sample["documents"]):
                print(f"  {json.dumps(meta, ensure_ascii=False)}")
                print(f"    tekst: {doc[:120].replace(chr(10), ' ')}...")
    else:
        print("\n(--dry-run: nic nie zapisano)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
