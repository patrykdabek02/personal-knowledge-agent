# Personal Knowledge Agent

Lokalny agent RAG odpowiadający na pytania na podstawie własnych notatek, z cytowaniem pliku źródłowego. **Cały pipeline działa offline** — embeddingi, baza wektorowa i generowanie odpowiedzi. Żaden fragment notatek nie opuszcza maszyny.

**Wyniki na 74 pytaniach testowych** (2026-08-09, 28 notatek, 132 fragmenty, próg 0,56): poprawne źródło na 1. miejscu w 77%, w top-6 w 98%, odmowa na pytania spoza bazy wiedzy w 86%, fałszywe odmowy 2%. Mediana czasu odpowiedzi 7,0 s na RTX 4060 Laptop (8 GB VRAM).

> **Jak czytać te liczby.** Nie są porównywalne z poprzednim pomiarem i to jest celowe.
> Zestaw testowy urósł z 38 do 74 pytań, ale ważniejsza jest zmiana jego składu:
> pytania spoza zakresu to już nie tylko „jak ugotować risotto", lecz bełkot,
> pojedyncze słowa i — najtrudniejsze — pytania zbudowane wyłącznie ze słów obecnych
> w notatkach, o fakt, którego tam nie ma („Ile zapłaciłem za licencję Chroma?").
> Stary zestaw dawał 100% odmów, bo nie zawierał niczego trudnego. Na nowym ta sama
> konfiguracja dawała **9%**. Spadek ze 100% na 86% oznacza uczciwszy pomiar,
> nie regresję.
>
> **Co dała bramka zakresu** (jeden dzień, ten sam zestaw 74 pytań):
>
> | | przed | po |
> |---|---|---|
> | odmowy poza zakresem | 2/22 (9%) | 19/22 (86%) |
> | fałszywe odmowy | 2/52 | 1/52 (2%) |
> | przejęcie tożsamości użytkownika | niewykrywane | wykrywane, 0 wycieków |
> | trafność źródła (top-1 / top-6) | 77% / 98% | bez zmian |
>
> Retrieval nie został ruszony — cały zysk pochodzi z decyzji o zakresie i z kontroli
> przeniesionych do kodu.
>
> **Czego te liczby NIE mówią:** nic o merytorycznej poprawności treści. Kolumna
> `ocena_reczna` wymaga przejrzenia odpowiedzi ręcznie i nie została jeszcze wypełniona.
> Znane braki: pytania jednosłowne i pytania z fałszywą przesłanką nadal bywają
> odpowiadane zamiast odrzucone; jedna notatka nie wchodzi do top-6 mimo trafnej treści.

> **Stan dokumentacji.** Tabele w dalszej części pochodzą z pomiaru z 2026-08-04
> (14 notatek, 58 fragmentów, próg 0,52) i opisują konfigurację, dla której je
> faktycznie zmierzono. Od tego czasu doszło sporo:
>
> - **wyszukiwanie hybrydowe** BM25 + wektory łączone metodą RRF, z lekkim stemmerem
>   dla polskiego (bez niego „baza wektorowa" nie trafia w „bazy wektorowej")
> - **Contextual Retrieval** — model dopisuje do fragmentu zdanie osadzające go
>   w całej notatce przed embedowaniem (`indexer.py --kontekst`)
> - **interfejs webowy** z czterema trybami, przypisami `[1]` i oznaczaniem zdań
>   bez pokrycia w notatkach
> - **`przetworz_inbox.py`** — rozbija surowe zapiski na pojedyncze fakty
>   i proponuje, do której notatki i sekcji trafią
> - **zbieranie wpadek** — zła odpowiedź jednym kliknięciem trafia do zestawu testowego
> - **bramka zakresu** — sam próg dystansu okazał się niewystarczający, bo bge-m3
>   jest anizotropowy i dwa niepowiązane teksty leżą w okolicy 0,60, nie 1,0.
>   Doszedł wymóg, by najlepsze trafienie **odstawało** od mediany pozostałych,
>   z wyjątkiem dla rzadkich i długich słów (nazwy własne, kody błędów)
> - **`graf.py`** — graf podobieństwa fragmentów, notatek albo tagów; wykrywa
>   duplikaty i sieroty, czyli notatki, których retrieval nie ma jak zaczepić
> - **`wizualizacja.py`** — rzut PCA prawdziwych wektorów z bazy
>
> Próg wynosi 0,56, baza urosła do 28 notatek i 132 fragmentów.
>
> **Wniosek przekrojowy z tych zmian:** wszystko, co miało być gwarancją, musiało
> ostatecznie trafić do kodu, nie do promptu. Cyrylica, pierwsza osoba, status sekcji
> („propozycja" kontra stan faktyczny), weryfikacja cytatów, limit liczby faktów,
> zakaz zakładania zbędnych plików — każde z tych ograniczeń najpierw próbowałem
> opisać w instrukcji dla modelu i za każdym razem trzymało tylko czasem. Co gorsza,
> dopisywanie kolejnych reguł wypychało wcześniejsze: jeden z przebiegów cofnął zysk
> poprzedniego wyłącznie przez dołożenie trzech zdań do promptu.

---

## Problem

Notatki, które piszę od lat, są bezużyteczne w momencie, w którym ich potrzebuję — bo nie pamiętam, że je napisałem. Wyszukiwanie pełnotekstowe wymaga, żebym trafił w te same słowa, których użyłem pół roku temu. Rozwiązania chmurowe odpadają, bo notatki zawierają decyzje zawodowe i finansowe.

Potrzebowałem czegoś, co odpowie na pytanie zadane własnymi słowami, wskaże plik źródłowy i **powie „nie wiem", zamiast zmyślać**.

---

## Architektura

```
        notatki .md
             │
   indexer.py │ chunkowanie po nagłówkach → okno 350 słów, overlap 60
             ▼
   Ollama / bge-m3  ──►  Chroma (PersistentClient, plik na dysku)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        ask.py (CLI)              search_api.py :8000 (FastAPI)
                                              │
                                              ▼
                                  n8n :5678 — webhook → wyszukiwanie
                                  → IF → prompt → Ollama → JSON
```

**Python trzyma logikę, n8n trzyma orkiestrację.** Podział celowy: retrieval można testować bez uruchamiania n8n, a to jest warstwa, którą debuguje się najczęściej — 90% złych odpowiedzi RAG pochodzi z wyszukiwania, nie z modelu.

### Stack

| Warstwa | Wybór |
|---|---|
| Orkiestracja | n8n 2.29.10 (webhook, węzeł warunkowy, retry) |
| Embeddingi | Ollama + `bge-m3` (1024 wymiary) |
| Baza wektorowa | Chroma, `PersistentClient`, metryka cosinusowa |
| Generowanie | Ollama + `qwen3.5:9b`, `think: false` |
| Indeksowanie i ewaluacja | Python 3.14 |

---

## Trzy decyzje projektowe

### 1. `bge-m3` zamiast `nomic-embed-text`

Wszystkie notatki są po polsku. `nomic-embed-text` jest trenowany głównie na angielskim; `bge-m3` stoi na XLM-RoBERTa i obsługuje ponad 100 języków. Autor repozytorium `obsidian-second-brain` zmierzył na vaulcie ~2350 notatek, że zapytania nieangielskie idą z ~0% do 63% recall@5 po przejściu na model wielojęzyczny.

Decyzja podjęta **przed** pierwszym indeksowaniem, bo zmiana modelu oznacza inną liczbę wymiarów i wymusza pełną przebudowę indeksu.

### 2. Odmowa jako gałąź przepływu, nie instrukcja w promptcie

Gdy żaden fragment nie przejdzie progu podobieństwa, węzeł warunkowy w n8n **odcina wywołanie modelu**. Nie da się zmyślić odpowiedzi, której się nie generuje.

Efekt uboczny jest mierzalny: pytania spoza zakresu wracają w 0,6 s zamiast po kilku sekundach, bo model nie jest w ogóle ładowany do zadania.

### 3. Chunkowanie świadome markdowna, z scalaniem krótkich sekcji

Podział najpierw po nagłówkach, potem okno 350 słów z zakładką 60. Każdy fragment dostaje prefiks `Tytuł notatki > Nagłówek`, który trafia do embeddingu.

Sekcje krótsze niż 80 znaków są **doklejane do sąsiednich, nie usuwane** — patrz problem 2 poniżej.

---

## Wyniki

Zestaw testowy: **32 pytania w zakresie** (znana odpowiedź i plik źródłowy) plus **6 pytań spoza zakresu**, na które agent musi odmówić. Pytania sformułowane opisowo, bez powtarzania słów z nagłówków notatek — test na własnych nagłówkach zawsze wypada świetnie i niczego nie mierzy.

Baza: 14 notatek, 58 fragmentów. Pomiar 2026-08-04.

| Metryka | Wynik |
|---|---|
| **Merytorycznie poprawna odpowiedź** | **28/32 (88%)** |
| **Konfabulacje** | **0/32 (0%)** |
| Poprawne źródło na 1. miejscu | 28/32 (88%) |
| Poprawne źródło w top-6 | 30/32 (94%) |
| Poprawne odmowy poza zakresem | 6/6 (100%) |
| Mediana czasu odpowiedzi | 5,90 s |

Trafność źródła liczy skrypt automatycznie. Poprawność merytoryczną oceniono ręcznie, czytając wszystkie 32 odpowiedzi w każdej z czterech testowanych konfiguracji.

### Jak powstał wynik: cztery konfiguracje

Wersja wyjściowa dawała 81%. Poprawa do 88% wymagała dwóch zmian, z których żadna osobno nie wystarczała — i z których **każda coś zepsuła**:

| Konfiguracja | Poprawne odpowiedzi | Mediana |
|---|---|---|
| `k=4`, filtrowanie każdego fragmentu | 26/32 (81%) | 3,4 s |
| `k=6`, filtrowanie każdego fragmentu | 26/32 (81%) | 4,6 s |
| `k=4`, bramka na najlepszym trafieniu | 27/32 (84%) | 4,7 s |
| **`k=6`, bramka na najlepszym trafieniu** | **28/32 (88%)** | 5,9 s |

**Poprawa nie była monotoniczna.** Każda zmiana naprawiała jedne odpowiedzi i psuła inne:

| Zmiana | Naprawiła | Zepsuła |
|---|---|---|
| bramka przy `k=4` | pytanie o problemy z agentem do maili, pytanie o kwotę wynagrodzenia | pytanie o największe przyspieszenie — model odwrócił kolejność czynników |
| `k=6` + bramka | pytanie o odrzucenie BI, pytanie o zawężenie zakresu projektu | pytanie o najsłabszą statystykę — odpowiedź stała się tautologiczna, bez p-value |

Netto **+2 odpowiedzi**, nie +4. Gdyby mierzyć tylko te pytania, które miały się poprawić, wynik wyszedłby na 94% — i byłby zawyżony. To jest praktyczny argument za ocenianiem **całego** zestawu po każdej zmianie, a nie tylko przypadków, pod które zmiana była projektowana.

**Dlaczego pojedyncze zmiany nie wystarczały.** Podniesienie `k` bez zmiany polityki odcięcia nie dało nic, bo próg i tak odrzucał nadmiarowe fragmenty — model dostawał tyle samo kontekstu co przy `k=4`. Sama bramka przy `k=4` naprawiła jedno pytanie, bo w czwórce najbliższych nadal nie było właściwej sekcji pliku.

**Dlaczego więcej kontekstu potrafi szkodzić.** Przy sześciu fragmentach model częściej streszcza ogólnie zamiast cytować konkret — stąd „najsłabszą statystyką jest statystyka pod główną tezę" zamiast wcześniejszego „Spearman rho blisko −1, p ≈ 0,083". Dodatkowy kontekst rozprasza tak samo, jak jego brak zubaża.

**Na czym polega bramka.** Pierwotnie próg filtrował każdy fragment osobno. Nowa polityka używa progu wyłącznie do rozstrzygnięcia, czy pytanie w ogóle mieści się w bazie wiedzy — decyduje o tym najlepsze trafienie. Gdy przejdzie, model dostaje komplet `k` fragmentów.

Gwarancja odmowy pozostaje nienaruszona: pytanie spoza zakresu ma najlepszy dystans powyżej progu, więc model nadal nie jest wywoływany.

### Objaw, który postawił diagnozę

```
[17] Co poszlo nie tak przy pierwszym agencie do maili?
     top1 trafiony : TAK   dystans: 0.5108
     ODPOWIEDŹ     : Nie znalazłem tego w notatkach.
```

Retrieval zwrócił właściwy plik **na pierwszym miejscu**, a model mimo to odmówił. Przy dystansie bliskim progu przechodził jeden fragment — sekcja „Co to jest" zamiast „Napotkane problemy". Model, któremu prompt zabrania zgadywania, uczciwie odmawiał.

To nie był błąd modelu ani retrievalu, tylko nadmierna ostrożność wynikająca ze złożenia wąskiego progu z surowym promptem. Po zmianie to samo pytanie odpowiada poprawnie i cytuje właściwą sekcję.

### Rozkład pozostałych błędów

| Typ | Liczba | Przyczyna |
|---|---|---|
| Odmowa przewidziana przez kalibrację | 2 | dystanse 0,532 i 0,549 przy progu 0,52 |
| Odpowiedź zbyt ogólna | 1 | model streścił sekcję zamiast zacytować liczby |
| Odwrócona kolejność faktów | 1 | model podał drugi co do wielkości czynnik jako największy |
| **Konfabulacja** | **0** | — |

**Wszystkie błędy to odmowy albo niepełne odpowiedzi. Ani jednego wymyślonego faktu, ani jednego zmyślonego źródła.** Kierunek błędu jest dokładnie ten, który wybrano świadomie przy kalibracji progu — i tutaj jest na to dowód liczbowy.

### Ograniczenie pomiaru czasu

Dwa przebiegi tej samej konfiguracji dały mediany **6,07 s** i **4,67 s** — 30% różnicy bez żadnej zmiany w kodzie. Przy 38 pytaniach szum pomiarowy jest tego rzędu, więc **różnice czasowe poniżej ~30% są nieodróżnialne od szumu** i tabela powyżej nie uprawnia do wniosku, że `k=6` jest wolniejsze od `k=4`.

Metryki jakościowe są odporne na ten problem — przy `temperature=0.1` nie drgnęły w żadnym z przebiegów tej samej konfiguracji.

Osobne znalezisko operacyjne: gdy `bge-m3` (664 MB) i `qwen3.5:9b` (6,4 GB) są rezydentne jednocześnie w 8 GB VRAM, Ollama zaczyna je przerzucać przy każdym zapytaniu i mediana skacze do **29 s**. `ollama stop` przed pomiarem stał się częścią procedury.

### Kalibracja progu podobieństwa

Próg odcięcia nie został zgadnięty, tylko zmierzony. Skrypt przepuszcza wszystkie pytania przez sam retrieval (bez modelu, kilkanaście sekund) i przemiata progi od 0,20 do 1,00:

```
ROZKŁAD NAJLEPSZEGO DYSTANSU
  w zakresie    (n=32): min 0.336  mediana 0.415  max 0.549
  poza zakresem (n= 6): min 0.536  mediana 0.616  max 0.717

  próg   w zakresie   odmowy   dokładność
  0.50    27/32        6/6        87%
  0.52    30/32        6/6        95%
  0.54    31/32        5/6        95%
  0.56    32/32        5/6        97%   ← optimum skryptu
```

**Wybrałem 0,52, mimo że skrypt rekomendował 0,56.** Skrypt maksymalizuje surową dokładność i traktuje oba typy błędu jednakowo. W praktyce nie są równoważne: fałszywe „nie wiem" kosztuje 10 sekund na ręczne sprawdzenie, a pewnie brzmiąca konfabulacja z cytowaniem nieistniejącego źródła kosztuje zaufanie do całego narzędzia.

Dodatkowo pytanie, które przecieka przy progu 0,56, to *„Jakie jest moje hasło do konta bankowego?"* (dystans 0,536) — najgorszy możliwy przypadek do przepuszczenia.

Próg 0,54 odpada, bo jest zdominowany przez 0,52: traci jedną poprawną odmowę, zyskując jedno pytanie.

---

## Napotkane problemy

Pełny opis z kodem i pomiarami: [PROBLEMY.md](PROBLEMY.md). Skrótowo cztery najciekawsze:

### BOM wyłączał parsowanie metadanych

Windows zapisuje UTF-8 z BOM. Przy odczycie jako `utf-8` te trzy bajty stają się znakiem `﻿`, przez co `text.startswith("---")` zawsze zwracało `False` i frontmatter nie był parsowany. Błąd nie rzucał wyjątku — degradował jakość po cichu. Poprawka: `utf-8-sig`.

### Krótkie sekcje znikały z indeksu

Filtr odrzucający fragmenty krótsze niż 80 znaków usunął sekcję `## Decyzja` zawierającą zdanie *„Wybrałem Chroma, nie Qdrant"* — czyli dosłownie odpowiedź, po którą ta notatka powstała.

Korelacja między długością fragmentu a jego wartością bywa **odwrotna**: w dobrze pisanych notatkach kluczowa decyzja jest krótka. Krótkie sekcje trzeba scalać, nie usuwać.

### Notatka o teście przechwyciła test

W notatce o wyborze modelu embeddingowego zapisałem dosłowną treść zapytania testowego („czym trzymam wektory na dysku"). Ta notatka stała się najlepszym dopasowaniem do tego zapytania z dystansem **0,356** — lepszym, niż ma dziś prawidłowa odpowiedź (0,464) — mimo że odpowiedzi nie zawierała.

Samozanieczyszczenie bazy: notatka opisująca system stała się jego częścią i zaburzyła pomiar. Po usunięciu cytatu top-1 wzrosło z 84% na 88%.

Wniosek szerszy: **niski dystans nie jest miarą trafności.** Dopasowanie niemal dosłowne wygrywa z dopasowaniem znaczeniowym, nawet gdy nie niesie odpowiedzi.

### Generowanie: 99 s → 9,5 s bez zmiany modelu

| Krok | Zmiana | Czas | PROCESSOR |
|---|---|---|---|
| wyjściowo | domyślne | 99,0 s | 19%/81% CPU/GPU |
| 1 | `num_ctx` 16384 → 6144 | 24,9 s | 12%/88% |
| 2 | `think: false` | 9,5 s | 12%/88% |

Odruchowa diagnoza („model nie mieści się w 8 GB VRAM, trzeba mniejszy") była trafna, ale nieistotna. Główny koszt siedział w bloku rozumowania `<think>`, który i tak był wycinany przed pokazaniem odpowiedzi — czyli w tokenach, których użytkownik nigdy nie widział.

---

## Znane ograniczenia

**Metryka zakłada jedno poprawne źródło.** Wraz ze wzrostem bazy ta sama informacja żyje w kilku notatkach, więc „złe źródło" nie zawsze znaczy „zła odpowiedź". Przy większym zestawie warto przejść na listę akceptowalnych źródeł.

**Dwa pytania w zakresie dostają „nie wiem"** (dystanse 0,532 i 0,549 przy progu 0,52). To świadomy koszt wyboru bezpieczniejszego progu, raportowany zamiast ukrywany.

**Próg jest ważny dla tej wielkości bazy.** Skalibrowany na 58 fragmentach niekoniecznie będzie dobry przy 2000 — wymaga rekalibracji wraz ze wzrostem.

**Model bywa niestabilny językowo.** `qwen3.5:9b` dwukrotnie na 32 odpowiedzi wtrącił słowo cyrylicą w środku polskiego zdania („занижение" zamiast „zaniżenie"). Instrukcja w prompcie systemowym **tego nie wyeliminowała** — dopiero wykrywanie po stronie kodu z automatycznym ponowieniem zapytania zbiło to do zera. Wniosek ogólny: gwarancji formatu nie da się oprzeć na instrukcji tekstowej dla modelu.

**Ta sama informacja w dwóch notatkach psuje atrybucję źródeł.** Przy `k=4` pytanie o problemy z pierwszym agentem odpowiadało poprawnie, ale cytowało notatkę o CV — bo tam ten sam błąd opisano jako przykład do listu motywacyjnego. Poprawność treści i poprawność cytowania to dwie różne rzeczy.

**Ocena merytoryczna jest ręczna.** Trafność źródła liczy skrypt; poprawność treści oceniana jest przez czytanie odpowiedzi i wpisanie 1/0 w kolumnie `ocena_reczna` w `results.csv`. Automatyzacja wymagałaby drugiego modelu jako sędziego, co przy 81% wyjściowej dokładności wprowadziłoby własny błąd pomiaru.

**Ewaluacja jest jednorazowym pomiarem, nie testem regresji.** Uruchamiana ręcznie po większych zmianach. Przy większej bazie warto ją włączyć do skryptu uruchamianego po każdym re-indeksie.

---

## Uruchomienie

```bash
pip install -r requirements.txt
ollama pull bge-m3
python indexer.py --path "ŚCIEŻKA/DO/NOTATEK"
python ask.py "twoje pytanie"
```

Pełna instrukcja krok po kroku: [INSTRUKCJA.md](INSTRUKCJA.md).

Workflow n8n: import `n8n_workflow.json`, uruchom `uvicorn search_api:app --host 127.0.0.1 --port 8000`.

### Struktura

| Plik | Rola |
|---|---|
| `core.py` | konfiguracja, embeddingi, Chroma, prompt, wywołanie modelu |
| `indexer.py` | chunkowanie i inkrementalne indeksowanie (hash SHA-256 treści) |
| `search_api.py` | lokalne API dla n8n, nasłuchuje tylko na 127.0.0.1 |
| `ask.py` | CLI, tryb `--search-only` do diagnostyki retrievalu |
| `evaluate.py` | kalibracja progu i pełna ewaluacja |
| `n8n_workflow.json` | workflow do importu |

---

## Następne kroki

**1. Uzupełnić dwie notatki**, których pytania nie przeszły progu (dystanse 0,532 i 0,549). Po poprawkach z sekcji „Jak powstał wynik" to są **jedyne dwa pozostałe błędy** poza przypadkiem granicznym. Informacja jest w tych notatkach opisana pośrednio — to problem treści, którego nie naprawi żaden parametr.

**2. Rozdzielić zduplikowaną treść.** Ten sam fakt opisany w dwóch notatkach powoduje, że model cytuje wtórne źródło zamiast pierwotnego. Warto przejrzeć bazę pod tym kątem, gdy urośnie.

**3. Reranker** (`bge-reranker-v2-m3`) — pobrać top-20 z Chromy, przesortować modelem cross-encoder, wziąć top-4. Największy spodziewany skok jakości, ale też największy koszt: dodatkowy model w VRAM, którego przy 8 GB brakuje.

**4. ~~Hybrid search BM25 + wektory~~ — ZROBIONE.** Ratuje zapytania o rzadkie nazwy własne, gdzie same embeddingi wypadają słabo: numer błędu SAP, kod węzła, nazwisko. Implementacja własna (BM25 Okapi, k1=1,5, b=0,75), wyniki łączone metodą **RRF** — sumowanie odwrotności pozycji w obu rankingach, dzięki czemu nie trzeba skalować niewspółmiernych punktacji.

Wymagało to lekkiego stemmera dla polskiego. Bez niego BM25 jest tu prawie bezużyteczny: „baza wektorowa" nie trafia w „bazy wektorowej", a „Katowice" w „Katowicach". Obcinanie końcówek musi być **dwuwarstwowe**, bo polski skleja sufiks słowotwórczy z fleksyjnym (`wektor + ow + ej`) — jedno przejście dawało niezgodne rdzenie dla tego samego słowa. Tokeny zawierające cyfry są z tego wyłączone, żeby nie uszkodzić dokładnie tych ciągów, dla których BM25 został dodany.

**Do zmierzenia:** wpływ na dokładność. Spodziewany zysk dotyczy pytań o rzadkie tokeny, ale nie został jeszcze potwierdzony pełną ewaluacją.

**5. Qdrant** — dopiero powyżej ~50 tys. fragmentów albo gdy pojawi się potrzeba poważnego filtrowania po metadanych. Chroma jest świadomie wybrana jako prostsza na tym etapie.

**6. Lista akceptowalnych źródeł** w zestawie testowym zamiast jednego oczekiwanego pliku — usuwa znane ograniczenie metryki.
