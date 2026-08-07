# Personal Knowledge Agent — instrukcja krok po kroku

Lokalny agent RAG nad Twoimi osobistymi notatkami. **Zero danych na zewnątrz**: embeddingi, baza wektorowa i generowanie odpowiedzi działają w całości na Twojej maszynie.

Czas realizacji: **jeden wieczór na Etapy 0–4** (działający retrieval), **drugi wieczór na Etapy 5–8** (n8n + zmierzone wyniki + repo).

---

## Architektura

```
                    ┌──────────────────────────────────────────┐
                    │  TWÓJ FOLDER NOTATEK (.md)               │
                    │  C:\Users\patryk\Notatki\                │
                    └────────────────┬─────────────────────────┘
                                     │
                        indexer.py   │  (jednorazowo + przy zmianach)
                                     ▼
            ┌────────────────────────────────────────────┐
            │  chunkowanie po nagłówkach + okno 350 słów  │
            │  embedding: Ollama / bge-m3  (localhost)    │
            │  zapis: Chroma PersistentClient (plik)      │
            └────────────────┬───────────────────────────┘
                             ▼
                   ┌───────────────────┐
                   │   chroma_db/      │  ← baza wektorowa na dysku
                   └─────────┬─────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌────────────────┐                    ┌──────────────────────────┐
│  ask.py (CLI)  │                    │ search_api.py :8000      │
│  do testów     │                    │ FastAPI, tylko 127.0.0.1 │
└────────────────┘                    └───────────┬──────────────┘
                                                  │ HTTP
                                                  ▼
                                      ┌───────────────────────────┐
                                      │  n8n workflow :5678       │
                                      │  webhook → search → IF →  │
                                      │  prompt → Ollama → JSON   │
                                      └───────────────────────────┘
```

**Podział odpowiedzialności jest celowy:** Python trzyma logikę (chunkowanie, embedding, retrieval), n8n trzyma orkiestrację i interfejs. Dzięki temu możesz testować retrieval bez uruchamiania n8n — a to jest ta część, którą będziesz debugować najczęściej.

---

## Trzy decyzje techniczne i uzasadnienie

### 1. Model embeddingowy: `bge-m3`, nie `nomic-embed-text`

To zmiana względem pierwotnego planu i **najważniejsza pojedyncza decyzja w tym projekcie**.

`nomic-embed-text` jest trenowany głównie na angielskim. Twoje notatki będą po polsku. `bge-m3` jest zbudowany na XLM-RoBERTa, obsługuje 100+ języków i został właśnie po to wybrany w repo `obsidian-second-brain` jako domyślny — ich pomiary na vaultcie ~2350 notatek pokazały, że zapytania nieangielskie poszły z ~0% do 63% recall@5 po przejściu na model wielojęzyczny.

| | nomic-embed-text | bge-m3 |
|---|---|---|
| wymiary | 768 | 1024 |
| rozmiar w Ollama | ~274 MB | ~1,2 GB |
| kontekst | 8192 | 8192 |
| polski | słaby | dobry |

**Konsekwencja:** zmiana modelu embeddingowego później = pełna przebudowa indeksu (inna liczba wymiarów). Wybierz teraz i się nie rozmyślaj.

### 2. Metryka: cosinus, nie L2

Chroma domyślnie używa L2. Wymuszamy `{"hnsw:space": "cosine"}` przy tworzeniu kolekcji, bo wszystkie progi w tym projekcie liczone są dla dystansu cosinusowego (0 = identyczne, 1 = ortogonalne, 2 = przeciwne). Jest to już zrobione w `core.py` — musisz tylko wiedzieć, dlaczego.

### 3. Chunk: sekcja markdown → okno 350 słów, overlap 60, z prefiksem kontekstowym

Nie tniemy na ślepo co N słów. Najpierw dzielimy po nagłówkach `#`, dopiero potem długie sekcje na okna. Każdy fragment dostaje prefiks `Tytuł notatki > Nagłówek sekcji`, więc fragment "wie", z czego pochodzi — i to trafia do embeddingu. To jedna z tańszych rzeczy, które realnie podnoszą trafność.

---

## ETAP 0 — Środowisko

### 0.1 Python

```powershell
python --version
```

Potrzebujesz 3.10+. Jeśli nie masz — zainstaluj z python.org, zaznaczając "Add Python to PATH".

### 0.2 Folder projektu i wirtualne środowisko

```powershell
mkdir C:\Users\patryk\personal-knowledge-agent
cd C:\Users\patryk\personal-knowledge-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jeśli PowerShell zablokuje skrypt aktywacji:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Skopiuj tutaj wszystkie pliki z tej paczki (`core.py`, `indexer.py`, `search_api.py`, `ask.py`, `evaluate.py`, `requirements.txt`, `.gitignore`, `questions.example.csv`, `n8n_workflow.json`).

### 0.3 Zależności

```powershell
pip install -r requirements.txt
```

Uwaga: `chromadb` ciągnie sporo zależności (~300 MB), pierwsza instalacja potrwa kilka minut. Wewnątrz venv **nie potrzebujesz** `--break-system-packages` — ta flaga jest tylko dla instalacji systemowej.

### 0.4 Model embeddingowy

```powershell
ollama pull bge-m3
ollama list
```

Powinieneś zobaczyć `bge-m3` i `qwen3.5:9b`.

### 0.5 Test środowiska

```powershell
python core.py
```

Oczekiwany wynik:

```json
{
  "ollama_url": "http://localhost:11434",
  "chroma_path": "...\\chroma_db",
  "ollama": "ok",
  "models": ["bge-m3:latest", "qwen3.5:9b"],
  "embed_model_present": true,
  "chat_model_present": true,
  "chunks": 0
}
```

**Nie idź dalej, dopóki `ollama` nie pokaże `ok` a oba modele `true`.**

---

## ETAP 1 — Folder notatek i konwencja zapisu

### 1.1 Struktura

Jeśli masz już vault Obsidiana — użyj go, pomiń tworzenie folderów. Jeśli zaczynasz od zera:

```
C:\Users\patryk\Notatki\
├── projekty\          ← projekty osobiste, decyzje techniczne
├── decyzje\           ← "postanowiłem X, bo Y" — najcenniejsza kategoria
├── zrodla\            ← notatki z artykułów, książek, filmów
├── dziennik\          ← notatki dzienne / tygodniowe
├── ludzie\            ← kto, kontekst, ustalenia
├── cele\              ← cele i przeglądy
└── meta\              ← konwencje, backup, jak działa mój system
```

Nazwa folderu pierwszego poziomu ląduje w metadanych jako `category` — dzięki temu możesz później filtrować wyszukiwanie do jednej kategorii (`ask.py --category decyzje`).

### 1.2 Konwencja notatki (to jest ważniejsze niż kod)

Piszesz notatki **dla przyszłego agenta, nie dla siebie za tydzień**. Wzorzec zapożyczony z `obsidian-second-brain` (ich "AI-first vault rule"), sprowadzony do tego, co realnie działa w RAG:

```markdown
---
title: Wybór bazy wektorowej do Personal Knowledge Agent
tags: rag, chroma, decyzja
data: 2026-08-03
---

## Kontekst

Buduję lokalnego agenta RAG nad własnymi notatkami. Potrzebuję bazy wektorowej,
która działa bez Dockera i bez konta w chmurze.

## Decyzja

Wybrałem **Chroma**, nie Qdrant. Stan na 2026-08.

## Dlaczego

- Chroma to zwykła biblioteka Pythona z `PersistentClient` — zero infrastruktury.
- Qdrant wymaga uruchomionego kontenera, co dokłada warstwę do debugowania.
- Przy skali kilku tysięcy fragmentów różnica w wydajności jest bez znaczenia.

## Czego to NIE rozstrzyga

Qdrant zostaje jako świadomie odłożony następny krok, gdy baza przekroczy
~50 tys. fragmentów albo gdy będę potrzebował filtrowania po metadanych na serio.
```

Cztery zasady, które faktycznie wpływają na jakość odpowiedzi:

1. **Jeden temat = jedna notatka.** Fragment wyciągnięty z notatki-worka jest bezużyteczny.
2. **Nagłówki `##` co 200–400 słów.** To po nich tnie chunker — dobre nagłówki to dobre chunki.
3. **Znacznik czasu przy twierdzeniach, które mogą się zdezaktualizować** ("stan na 2026-08"). Bez tego agent zacytuje Ci decyzję sprzed roku jako aktualną.
4. **Pisz pełnymi zdaniami, nie skrótami.** "Chroma bo prostsze" nie zaembeduje się sensownie. "Wybrałem Chroma, bo nie wymaga Dockera" — tak.

### 1.3 Ile notatek na start

Minimum **15–20 plików**, żeby test 20 pytań miał sens. Jeśli masz mniej — nie zaczynaj od budowania pipeline'u, zacznij od pisania notatek. Agent nad pustym vaultem to zabawka.

---

## ETAP 2 — Indeksowanie

### 2.1 Próba na sucho

```powershell
python indexer.py --path "C:\Users\patryk\Notatki" --dry-run
```

Pokaże, ile plików znajdzie i ile fragmentów z nich zrobi — bez zapisu i bez wywołań Ollama. Szybkie, do sprawdzenia czy ścieżka jest dobra.

**Na co patrzeć:**
- pliki z 1 fragmentem → za krótkie albo brak nagłówków
- pliki z 50+ fragmentami → notatka-worek, rozbij ją
- "pominięty (brak treści)" → sam frontmatter bez tekstu

### 2.2 Właściwe indeksowanie

```powershell
python indexer.py --path "C:\Users\patryk\Notatki"
```

Pierwsze uruchomienie: ~1–3 sekundy na plik (embedding jest liczony lokalnie). 50 notatek ≈ 1–2 minuty.

Na końcu dostaniesz:

```
============================================================
PODSUMOWANIE
============================================================
  nowe pliki        : 34
  zmienione pliki   : 0
  bez zmian         : 0
  puste / pominiete : 2
  usuniete z bazy   : 0
  nowe fragmenty    : 187
  czas              : 94.3 s
  fragmentow w bazie: 187

Przykladowe metadane:
  {"rel_path": "decyzje/baza-wektorowa.md", "source": "baza-wektorowa.md", ...}
```

### 2.3 Re-indeks

Uruchamiaj **ten sam skrypt** po każdej sesji pisania notatek:

```powershell
python indexer.py --path "C:\Users\patryk\Notatki"
```

Skrypt liczy SHA-256 treści każdego pliku i przetwarza **tylko to, co się zmieniło**. Pliki usunięte z dysku czyści z bazy. Odpowiada to na pytanie z pierwotnego planu ("pełny rebuild czy upsert") — masz oba: domyślnie inkrementalny, `--rebuild` gdy chcesz od zera (np. po zmianie modelu embeddingowego lub parametrów chunkowania).

---

## ETAP 3 — Pierwszy test retrievalu

**Zanim podłączysz LLM, sprawdź czy wyszukiwanie w ogóle działa.** 90% złych odpowiedzi RAG to nie wina modelu, tylko retrievalu.

```powershell
python ask.py --search-only "dlaczego wybrałem Chroma"
```

```
Zapytanie: dlaczego wybrałem Chroma
Prog: 0.55   Trafien po progu: 3/4

OK  0.312  decyzje/baza-wektorowa.md  > Dlaczego
      Wybór bazy wektorowej > Dlaczego  Chroma to zwykła biblioteka Pythona...

OK  0.408  projekty/personal-knowledge-agent.md  > Stack
      ...
  x 0.601  dziennik/2026-07.md  > -
      ...
```

**Test krytyczny — parafraza.** Zadaj pytanie, które nie zawiera ani jednego słowa z notatki:

```powershell
python ask.py --search-only "czym trzymam wektory na dysku"
```

Jeśli to znajduje właściwą notatkę — embeddingi działają. Jeśli nie znajduje nic sensownego, a wyszukiwanie po dosłownych słowach działa — masz problem z modelem embeddingowym (sprawdź, czy na pewno używasz `bge-m3`).

### 3.1 Pierwsza pełna odpowiedź

```powershell
python ask.py "Jaką bazę wektorową wybrałem i dlaczego akurat tę?"
```

```
======================================================================
Wybrałeś Chroma, a nie Qdrant [źródło: decyzje/baza-wektorowa.md].
Powody: Chroma to zwykła biblioteka Pythona z PersistentClient, więc nie
wymaga żadnej infrastruktury, podczas gdy Qdrant wymagałby uruchomionego
kontenera [źródło: decyzje/baza-wektorowa.md].
======================================================================

Zrodla:
  - decyzje/baza-wektorowa.md

[retrieval 0.41s | generacja 6.82s | w zakresie: tak]
```

---

## ETAP 4 — Kalibracja progu podobieństwa

To jest etap, który w planie był "otwartym pytaniem". Odpowiedź: **nie zgaduj progu, zmierz go.**

### 4.1 Przygotuj pytania testowe

```powershell
copy questions.example.csv questions.csv
notepad questions.csv
```

Zamień przykłady na **swoje** pytania. Format:

```csv
pytanie,oczekiwane_zrodlo,uwagi
Jaką bazę wektorową wybrałem i dlaczego?,decyzje/baza-wektorowa.md,
Jaka jest stolica Australii?,POZA_ZAKRESEM,czysta wiedza ogólna
```

Zasady:
- **20 pytań w zakresie** — na każde znasz odpowiedź i wiesz, z którego pliku pochodzi
- **5 pytań POZA_ZAKRESEM** — takich, na które agent *musi* odmówić

Te 5 pytań spoza zakresu to nie formalność. Bez nich zmierzysz tylko, jak często agent trafia — a nie jak często **pewnie zmyśla**, co jest znacznie groźniejszym błędem w bazie wiedzy.

Formułuj pytania **tak, jak naprawdę zapytasz za trzy miesiące** — nie kopiuj nagłówków z notatek. Test na własnych nagłówkach zawsze wyjdzie świetnie i niczego Ci nie powie.

### 4.2 Kalibracja

```powershell
python evaluate.py --calibrate --questions questions.csv
```

To leci przez sam retrieval (bez LLM), więc trwa kilkanaście sekund:

```
============================================================
ROZKLAD NAJLEPSZEGO DYSTANSU
============================================================
  w zakresie   (n=20): min 0.198  mediana 0.371  max 0.559
  poza zakresem(n= 5): min 0.612  mediana 0.734  max 0.891

============================================================
PRZEMIATANIE PROGU
============================================================
  prog   trafne_w_zakresie  poprawne_odmowy  laczna_dokladnosc
  ...
  0.56   20/20              5/5              100%  <-
  0.58   20/20              5/5              100%
  0.60   20/20              5/5              100%
  0.62   20/20              4/5               96%
  ...

============================================================
REKOMENDACJA: MAX_DISTANCE = 0.56  (dokladnosc 100%)
============================================================
```

### 4.3 Ustaw próg

```powershell
$env:MAX_DISTANCE="0.56"
```

...albo na stałe w `core.py` (linia `MAX_DISTANCE = ...`). **Zaktualizuj też wartość `max_distance` w węźle "Szukaj w Chroma" w n8n** — inaczej CLI i workflow będą się zachowywać inaczej i zmarnujesz wieczór na szukanie, dlaczego.

### 4.4 Gdy zakresy się nakładają

Jeśli `max` w zakresie > `min` poza zakresem, skrypt Cię ostrzeże. To normalne i nie znaczy, że coś zepsułeś. Znaczy, że **nie istnieje próg dzielący idealnie** i musisz wybrać stronę, po której wolisz się mylić:

- **niższy próg** → więcej "nie znalazłem" na pytania, na które odpowiedź jednak jest
- **wyższy próg** → więcej odpowiedzi zbudowanych na słabo pasujących fragmentach

W osobistej bazie wiedzy wybieraj **niższy**. Fałszywe "nie wiem" kosztuje Cię 10 sekund, żeby sprawdzić ręcznie. Pewnie brzmiąca konfabulacja z cytowaniem nieistniejącego źródła kosztuje Cię zaufanie do całego narzędzia.

---

## ETAP 5 — Search API

n8n potrzebuje czegoś, do czego uderzy HTTP-em. Uruchom w **osobnym oknie PowerShell** (musi działać cały czas):

```powershell
cd C:\Users\patryk\personal-knowledge-agent
.\.venv\Scripts\Activate.ps1
uvicorn search_api:app --host 127.0.0.1 --port 8000
```

Test w drugim oknie:

```powershell
curl http://localhost:8000/health
```

```powershell
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d "{\"query\":\"baza wektorowa\",\"k\":4,\"max_distance\":0.56}"
```

`--host 127.0.0.1` jest celowe: API nie jest widoczne w sieci lokalnej, tylko z tej maszyny.

Masz też `POST /ask`, które robi cały RAG w jednym strzale — wygodne do testów, ale w n8n używamy `/search`, żeby budowanie promptu i wywołanie modelu były widoczne jako osobne węzły workflow.

---

## ETAP 6 — Workflow n8n

### 6.1 Import

1. n8n → **Workflows** → **Import from File**
2. Wskaż `n8n_workflow.json`
3. Zapisz i **aktywuj** (przełącznik Active w prawym górnym rogu)

Dostajesz sześć węzłów plus notatkę:

```
Webhook → Szukaj w Chroma → Czy w zakresie? ─┬─ true → Buduj prompt → Ollama → Formatuj ─┐
                                              └─ false → Odmowa ───────────────────────────┴→ Odpowiedz
```

### 6.2 Jeśli n8n działa w Dockerze

`localhost` wewnątrz kontenera to sam kontener, nie Twój Windows. W węzłach **Szukaj w Chroma** i **Ollama: qwen3.5** zamień:

```
http://localhost:8000/search   →   http://host.docker.internal:8000/search
http://localhost:11434/api/chat →  http://host.docker.internal:11434/api/chat
```

Jeśli n8n zainstalowałeś przez npm lub jako aplikację desktop — zostaw `localhost`.

### 6.3 Test

```powershell
curl -X POST http://localhost:5678/webhook/pka-ask -H "Content-Type: application/json" -d "{\"question\":\"Jaka baze wektorowa wybralem?\"}"
```

Odpowiedź:

```json
{
  "question": "Jaka baze wektorowa wybralem?",
  "answer": "Wybrałeś Chroma... [źródło: decyzje/baza-wektorowa.md]",
  "sources": ["decyzje/baza-wektorowa.md"],
  "in_scope": true,
  "best_distance": 0.312
}
```

Podczas testów w edytorze n8n używaj `/webhook-test/pka-ask` i klikaj "Listen for test event". Adres `/webhook/pka-ask` działa dopiero po aktywacji workflow.

### 6.4 Co daje gałąź "Odmowa"

Gdy `search_api` nie znajdzie nic poniżej progu, zwraca `in_scope: false`. Węzeł IF kieruje wtedy do gałęzi odmowy — **model w ogóle nie jest wywoływany**. To ważne z dwóch powodów: nie da się zmyślić odpowiedzi, której się nie generuje, a pytania spoza zakresu wracają w ułamku sekundy zamiast po 8 sekundach. Etap 4 z pierwotnego planu ("obsługa błędów") jest tym samym załatwiony strukturalnie, nie promptem.

Retry na węzłach HTTP (3 próby, odstęp 2–3 s) jest już włączony — ten sam wzorzec co w Inbox Triage Agent.

---

## ETAP 7 — Pomiar

To odróżnia projekt od tutoriala. Nie piszesz "działa", tylko podajesz liczbę.

```powershell
python evaluate.py --questions questions.csv --out results.csv
```

25 pytań × ~8 s ≈ 3–4 minuty.

```
============================================================
WYNIKI
============================================================
  pytan w zakresie              : 20
  poprawne zrodlo na 1. miejscu : 16/20 (80%)
  poprawne zrodlo w top-4       : 19/20 (95%)
  falszywe odmowy (w zakresie)  : 1/20 (5%)

  pytan poza zakresem           : 5
  poprawnie odmowil             : 5/5 (100%)

  sredni czas odpowiedzi        : 7.84 s
  mediana                       : 7.10 s
  najwolniejsze                 : 14.20 s

Szczegoly zapisane do: results.csv
```

### 7.1 Ocena merytoryczna

Trafność źródła policzył skrypt. **Merytoryczną poprawność musisz ocenić sam** — otwórz `results.csv`, przejrzyj kolumnę `odpowiedz` i wypełnij `ocena_reczna` (1/0). To Twoja trzecia metryka i jedyna, której nie da się zautomatyzować bez drugiego modelu jako sędziego.

### 7.2 Jak czytać wyniki

| Objaw | Prawdopodobna przyczyna | Co zrobić |
|---|---|---|
| top-4 wysokie, top-1 niskie | ranking działa, kolejność nie | podnieś `k` do 6 w prompt'cie |
| top-4 niskie (<70%) | problem z retrievalem, nie z LLM | krótsze notatki, więcej nagłówków, sprawdź `--dry-run` |
| dużo fałszywych odmów | próg za ostry | wróć do `--calibrate`, podnieś próg |
| poprawne źródło, zła odpowiedź | fragment za krótki / bez kontekstu | zwiększ `CHUNK_WORDS` do 500 i przebuduj |
| agent zmyśla mimo odmowy w API | prompt przegrywa z modelem | sprawdź, czy gałąź IF na pewno działa |

**Realistyczne oczekiwania na starcie:** źródło w top-4 na poziomie 85–95%, top-1 70–85%, odmowy poza zakresem 100% (to musi być 100% — jeśli nie jest, próg jest za luźny).

---

## ETAP 8 — Repo i README

```powershell
git init
git add .gitignore core.py indexer.py search_api.py ask.py evaluate.py requirements.txt n8n_workflow.json questions.example.csv README.md
git commit -m "Personal Knowledge Agent - lokalny RAG nad notatkami"
```

**`.gitignore` z tej paczki blokuje `chroma_db/`, `questions.csv`, `results.csv` i typowe nazwy folderów z notatkami.** Sprawdź przed pierwszym pushem:

```powershell
git status --short
```

Jeśli widzisz cokolwiek z treścią swoich notatek — zatrzymaj się i popraw `.gitignore`. `results.csv` zawiera pełne odpowiedzi agenta, czyli fragmenty Twoich notatek; `questions.example.csv` jest bezpieczny, `questions.csv` już nie.

Struktura README (ten sam format co Inbox Triage Agent):

1. **Problem** — jedno zdanie, po co to powstało
2. **Architektura** — diagram + uzasadnienie trzech decyzji (bge-m3, cosinus, chunking)
3. **Wyniki** — tabela z Etapu 7, konkretne liczby, data pomiaru, rozmiar bazy
4. **Kalibracja progu** — pokaż wykres/tabelę przemiatania. To jest najciekawszy fragment dla czytelnika, bo pokazuje metodę, nie tylko efekt
5. **Napotkane problemy** — pisz na bieżąco, nie z pamięci na końcu
6. **Ograniczenia** — czego to nie robi (brak wielojęzycznego cross-lingual, brak reranking, jedna kolekcja)
7. **Następne kroki** — Qdrant, reranker, UI

---

## Utrzymanie

**Po każdej sesji pisania notatek:**

```powershell
cd C:\Users\patryk\personal-knowledge-agent
.\.venv\Scripts\Activate.ps1
python indexer.py --path "C:\Users\patryk\Notatki"
```

**Automat (Harmonogram zadań Windows), codziennie o 23:00:**

Utwórz `reindex.bat`:

```bat
@echo off
cd /d C:\Users\patryk\personal-knowledge-agent
call .venv\Scripts\activate.bat
python indexer.py --path "C:\Users\patryk\Notatki" >> reindex.log 2>&1
```

Następnie: Harmonogram zadań → Utwórz zadanie podstawowe → codziennie → wskaż `reindex.bat`.

**Raz na kwartał** przepuść ponownie `evaluate.py`. Baza rośnie, a wraz z nią rośnie szansa, że fragmenty z różnych notatek zaczną ze sobą konkurować — próg skalibrowany na 187 fragmentach niekoniecznie jest dobry przy 2000.

---

## Troubleshooting

**`Nie udalo sie pobrac embeddingow z Ollama`**
Ollama nie działa albo nie ma modelu. `ollama list`, potem `ollama serve` w osobnym oknie.

**Indeksowanie trwa wieczność**
bge-m3 na CPU to ~1–3 s na fragment. 1000 fragmentów = ~30 min przy pierwszym uruchomieniu, kolejne przebiegi są inkrementalne. Jeśli masz GPU NVIDIA, Ollama użyje go automatycznie.

**Wszystkie dystanse ~0.9, nic nie przechodzi progu**
Prawie na pewno indeks zbudowany innym modelem niż ten, którym pytasz. `python indexer.py --path "..." --rebuild`.

**`chunks: 0` mimo udanego indeksowania**
`CHROMA_PATH` różni się między uruchomieniami — pewnie odpalasz skrypty z innego katalogu roboczego. Zawsze uruchamiaj z folderu projektu albo ustaw `$env:CHROMA_PATH` na ścieżkę bezwzględną.

**n8n: `ECONNREFUSED 127.0.0.1:8000`**
Search API nie działa (Etap 5) albo n8n siedzi w Dockerze i potrzebuje `host.docker.internal` (6.2).

**Odpowiedzi zawierają `<think>...`**
`strip_thinking()` w `core.py` i regex w węźle "Formatuj odpowiedź" to czyszczą. Jeśli przecieka — model zwrócił niedomknięty blok, bo output został ucięty; zwiększ limit tokenów albo skróć kontekst (mniejsze `k`).

**Agent cytuje źródło, którego nie ma w `sources`**
Model wymyślił nazwę pliku. Objaw zbyt luźnego progu albo za długiego kontekstu — zejdź z `k` na 3.

---

## Kryteria "gotowe"

- [ ] `python core.py` pokazuje oba modele i > 0 fragmentów
- [ ] `ask.py --search-only` znajduje właściwą notatkę **na parafrazę**, nie tylko na dosłowne słowa
- [ ] Próg skalibrowany na danych, nie zgadnięty — i ta sama wartość w `core.py` oraz w n8n
- [ ] Workflow n8n odpowiada przez webhook z listą źródeł w JSON
- [ ] 25 pytań przepuszczone, `results.csv` istnieje, liczby wpisane do README
- [ ] Odmowy poza zakresem = 100%
- [ ] `git status` przed pushem nie pokazuje ani jednej notatki

---

## Co dalej (świadomie poza zakresem tej wersji)

- **Reranker** (`bge-reranker-v2-m3`) — pobierz top-20 z Chroma, przesortuj modelem cross-encoder, weź top-4. Największy pojedynczy skok jakości po tym, co już masz.
- **Qdrant** — gdy przekroczysz ~50 tys. fragmentów albo zaczniesz potrzebować poważnego filtrowania po metadanych.
- **Hybrid search** — BM25 + wektory. Ratuje zapytania o rzadkie nazwy własne, gdzie same embeddingi wypadają słabo.
- **Prosty UI** — jednoplikowy HTML z `fetch()` do webhooka n8n. Świadomie na końcu: dopiero gdy kontrakt odpowiedzi jest stabilny.
