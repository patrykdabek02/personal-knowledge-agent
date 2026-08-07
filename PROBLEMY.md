# Napotkane problemy

Sekcja do wklejenia do README. Wszystkie problemy z fazy budowy pipeline'u (2026-08-04),
z objawem, przyczyną, rozwiązaniem i wnioskiem. Liczby są zmierzone, nie szacowane.

**Sprzęt:** RTX 4060 Laptop, 8188 MiB VRAM · Windows 11 · Python 3.14 · Ollama · n8n 2.29.10
**Modele:** `bge-m3` (embeddingi, 1024 wymiary) · `qwen3.5:9b` (generowanie)

---

## 1. BOM na początku pliku wyłączał parsowanie frontmattera

**Objaw.** Notatki zapisane w Notatniku i przez PowerShell (`Set-Content -Encoding UTF8`) traciły metadane — `title` z frontmattera nie trafiał do indeksu, a chunki dostawały nazwę pliku zamiast tytułu notatki.

**Przyczyna.** Windows zapisuje UTF-8 **z BOM** (`EF BB BF`). Przy odczycie z `encoding="utf-8"` te trzy bajty stają się znakiem `\ufeff` na początku stringa, więc warunek wykrywający frontmatter:

```python
if not text.startswith("---"):
    return {}, text
```

zawsze zwracał `False`. Frontmatter nie był parsowany i lądował w treści jako zwykły tekst.

**Rozwiązanie.** Odczyt jako `utf-8-sig` — ten kodek zjada BOM, jeśli jest, i działa normalnie, jeśli go nie ma:

```python
text = path.read_text(encoding="utf-8-sig")
```

**Wniosek.** Przy pipeline'ach czytających pliki tworzone na Windowsie `utf-8-sig` powinien być domyślnym wyborem, nie `utf-8`. Ten błąd nie rzuca wyjątku — degraduje jakość po cichu, co czyni go groźniejszym niż crash.

**Uwaga poboczna.** `Get-Content` w PowerShell 5.1 domyślnie czyta jako ANSI, więc poprawny plik UTF-8 wyświetla się jako krzaki (`Ĺ‚` zamiast `ł`). To artefakt wyświetlania, nie uszkodzenie pliku — Python czytał go poprawnie. Do podglądu: `Get-Content plik.md -Encoding UTF8`.

---

## 2. Krótkie sekcje były po cichu wyrzucane z indeksu

**Objaw.** Notatka z decyzją o wyborze bazy wektorowej dała 2 fragmenty zamiast 3. Zdanie **„Wybrałem Chroma, nie Qdrant"** — czyli dosłownie odpowiedź na pytanie, po które ta notatka powstała — w ogóle nie trafiło do indeksu.

**Przyczyna.** Filtr odrzucający fragmenty krótsze niż `MIN_CHUNK_CHARS = 80` znaków. Sekcja `## Decyzja` miała 44 znaki. Filtr powstał, żeby odsiewać puste sekcje i śmieci, a odsiewał również najgęstsze informacyjnie fragmenty — bo w dobrze pisanych notatkach kluczowa decyzja jest właśnie krótka.

**Rozwiązanie.** Zamiast wyrzucać, doklejać do sąsiedniej sekcji, zachowując nagłówek w treści:

```python
def merge_short_sections(sections):
    merged = []
    for heading, text in sections:
        if len(text.strip()) < MIN_CHUNK_CHARS and merged:
            prefix = f"{heading}: " if heading else ""
            merged[-1][1] = f"{merged[-1][1]}\n\n{prefix}{text.strip()}"
        else:
            merged.append([heading, text.strip()])
    return merged
```

Efekt — fragment „Kontekst" zawiera teraz również `Decyzja: Wybrałem Chroma, nie Qdrant. Stan na 2026-08.`

**Wniosek.** Filtry długości w chunkowaniu są niebezpieczne, bo korelacja między długością fragmentu a jego wartością bywa **odwrotna**. Krótkie sekcje należy scalać, nie usuwać. Błąd wyszedł wyłącznie dlatego, że test sprawdzał obecność konkretnego zdania w indeksie, a nie samą liczbę fragmentów.

---

## 3. Pusta odpowiedź po ograniczeniu liczby tokenów

**Objaw.** Po dodaniu limitu `num_predict: 600` agent zwrócił pustą odpowiedź — poprawne źródło, poprawny czas, zero treści.

**Przyczyna.** Model Qwen generuje blok rozumowania `<think>...</think>`, który wycinamy przed pokazaniem odpowiedzi. Limit 600 tokenów ucinał generowanie **w środku tego bloku**, więc znacznik zamykający nigdy nie powstawał. Funkcja czyszcząca traktowała niedomknięty blok jako całą treść i zwracała pusty string:

```python
if "<think>" in cleaned and "</think>" not in cleaned:
    cleaned = cleaned.split("<think>")[0]   # -> "" gdy blok zaczynał odpowiedź
```

**Rozwiązanie dwutorowe.**

Po pierwsze — wyłączyć rozumowanie u źródła zamiast wycinać po fakcie. Ollama przyjmuje `think: false` na poziomie zapytania (nie w `options`), z automatycznym wycofaniem się przy HTTP 400, gdyby model tego nie wspierał.

Po drugie — przepisać funkcję czyszczącą tak, żeby **nigdy** nie zwracała pustego stringa: obsługa niedomkniętego `<think>`, osieroconego `</think>`, wielu bloków, oraz ostateczny fallback usuwający same znaczniki. Sześć przypadków brzegowych pokrytych testem.

**Wniosek.** Funkcja post-processingu, która potrafi zwrócić pustkę, jest gorsza od braku post-processingu — usuwa sygnał zamiast szumu. Każde przetwarzanie wyjścia modelu powinno mieć gwarantowany fallback do czegoś niepustego.

To był zarazem najbardziej pouczający błąd w całym projekcie: pusta odpowiedź była **objawem**, ale zarazem **dowodem**, że thinking jest włączony i że to on odpowiada za czas generowania. Bez tego błędu optymalizacja poszłaby w kierunku kwantyzacji modelu, czyli w ślepą uliczkę.

---

## 4. Generowanie odpowiedzi: 99 s → 9,5 s

**Objaw.** Pierwsza działająca odpowiedź RAG powstawała 99 sekund. Przy planowanej ewaluacji 25 pytań oznaczałoby to ponad 40 minut na jeden przebieg testu.

**Diagnoza.** `ollama ps` pokazał dwie rzeczy:

```
NAME          SIZE      PROCESSOR          CONTEXT
qwen3.5:9b    6.7 GB    19%/81% CPU/GPU    16384
```

19% warstw modelu liczyło się na CPU. Ponieważ każdy generowany token przechodzi przez wszystkie warstwy, ta mniejszość dyktowała tempo całości. Powód: przy 8188 MiB VRAM model (6,7 GB) plus cache KV dla okna 16384 tokenów nie mieściły się razem — a realny prompt w tym systemie ma ~3000 tokenów, więc płaciliśmy pamięcią za kontekst, którego nie używamy.

**Zmiany i pomiary.**

| Krok | Zmiana | Czas generowania | PROCESSOR |
|---|---|---|---|
| stan wyjściowy | domyślne ustawienia | **99,0 s** | 19%/81% CPU/GPU |
| 1 | `num_ctx: 16384 → 6144` | **24,9 s** | 12%/88% CPU/GPU |
| 2 | `think: false` | **9,5 s** | 12%/88% CPU/GPU |

Ograniczenie kontekstu dało 4×, wyłączenie rozumowania kolejne 2,6×. Łącznie **10,4× szybciej**, bez zmiany modelu i bez utraty jakości odpowiedzi — treść po optymalizacji jest równie poprawna i tak samo ocytowana.

**Wniosek.** Odruchowa diagnoza („model nie mieści się w VRAM, trzeba mniejszy") była trafna, ale nieistotna. Główny koszt siedział w tokenach, których użytkownik nigdy nie widział. Pozostałe 12% na CPU odpowiada dziś za ułamek czasu i ściganie ich przez zejście na Q4 dałoby może 6 s przy realnym ryzyku dla jakości — czyli optymalizację warto było zatrzymać właśnie tutaj.

**Czego nie zmieniono świadomie.** `bge-m3` (664 MB) mieści się w całości na GPU, ale wygasa z pamięci między zapytaniami — stąd 4,17 s przy pierwszym pytaniu po przerwie i poniżej sekundy przy kolejnych. Trzymanie go na stałe przez `OLLAMA_KEEP_ALIVE` kosztowałoby VRAM, którego przy 8 GB brakuje modelowi generującemu. Zostawione domyślnie.

---

## 5. Decyzja: `bge-m3` zamiast `nomic-embed-text`

To nie był błąd, tylko zmiana pierwotnego planu — ale najważniejsza pojedyncza decyzja w projekcie, więc warta odnotowania.

Pierwotny plan zakładał `nomic-embed-text`. Model ten jest trenowany głównie na angielskim, a wszystkie notatki w tym systemie są po polsku. `bge-m3` jest zbudowany na XLM-RoBERTa i obsługuje ponad 100 języków.

| | nomic-embed-text | bge-m3 |
|---|---|---|
| wymiary | 768 | 1024 |
| rozmiar w Ollama | ~274 MB | ~664 MB |
| polski | słaby | dobry |

Weryfikacja praktyczna: zapytanie **„czym trzymam wektory na dysku"** — nie zawierające ani słowa „Chroma", ani „baza wektorowa" — poprawnie trafiło w notatkę o wyborze bazy wektorowej. To jest test, o który w RAG naprawdę chodzi: dopasowanie po znaczeniu, nie po słowach kluczowych.

**Konsekwencja operacyjna.** Zmiana modelu embeddingowego oznacza inną liczbę wymiarów, czyli wymusza pełną przebudowę indeksu (`indexer.py --rebuild`). Decyzja jest kosztowna do odwrócenia i dlatego została podjęta przed pierwszym indeksowaniem, a nie po.

---

## 6. Uwagi środowiskowe (Windows)

Drobne, ale kosztowały czas:

- **Ścieżki.** Nazwa użytkownika w systemie (`patry`) różniła się od zakładanej w instrukcji. `mkdir C:\Users\patryk\...` próbowało utworzyć nowy katalog bezpośrednio w `C:\Users`, co jest lokalizacją chronioną i kończy się `PermissionDenied`. Rozwiązanie: konsekwentnie `$env:USERPROFILE` zamiast ścieżek wpisywanych na sztywno.

- **Wklejanie wieloliniowe.** Wielolinijkowe polecenia PowerShell gubiły przy wklejeniu pierwszą część, przez co cmdlety uruchamiały się bez argumentów i wpadały w tryb dopytywania (`Supply values for the following parameters:`). Rozwiązanie: polecenia jednoliniowe sklejane średnikami, a dłuższe treści plików tworzone w edytorze zamiast przez here-stringi.

---

### `localhost` i `127.0.0.1` nie sa zamienne na Windowsie

**Objaw.** Webhook n8n zwracal pusta odpowiedz (`""`) z kodem 200, bez zadnego bledu. To samo API
odpytane z PowerShella (`Invoke-RestMethod http://127.0.0.1:8000/health`) dzialalo bez zarzutu.

**Przyczyna.** Wezel HTTP w n8n wolal `http://localhost:8000/search`. Na Windowsie `localhost`
rozwiazuje sie **najpierw do IPv6** (`::1`), a `uvicorn --host 127.0.0.1` nasluchuje wylacznie
na IPv4 i takie polaczenie odrzuca. PowerShell dzialal, bo `Invoke-RestMethod` po nieudanej
probie IPv6 sam ponawia na IPv4 - wezel HTTP w n8n tego nie robi.

Pusta odpowiedz zamiast bledu brala sie stad, ze workflow przerywal sie na wezle wyszukiwania,
wiec wezel odpowiadajacy nigdy sie nie wykonywal.

**Rozwiazanie.** Jawne `http://127.0.0.1:8000/search` i `http://127.0.0.1:11434/api/chat`
w wezlach HTTP. Alternatywa - uruchamianie API z `--host 0.0.0.0` - zostala odrzucona,
bo wystawialaby je na cala siec lokalna, co kloci sie z zalozeniem pelnej lokalnosci.

**Wniosek.** Diagnoza zajela kilka podejsc, bo objaw (pusta odpowiedz, status 200) nie wskazywal
na warstwe sieciowa. Rozstrzygnela dopiero zakladka **Executions** w n8n, ktora pokazala
konkretny wezel i komunikat "The service refused the connection". Przy debugowaniu przeplywu
n8n log wykonan jest pierwszym miejscem do sprawdzenia, nie ostatnim.

**Pulapka przy poprawianiu.** Za pierwszym razem `127.0.0.1` trafilo w miejsce numeru portu
zamiast nazwy hosta - powstalo `http://localhost:127.0.0.1/search`. Adres wygladal na zmieniony,
wiec temat wydawal sie zalatwiony, a n8n dalej pukal w nieistniejacy port. Warto czyscic cale
pole i wpisywac od zera zamiast edytowac fragment.

---

## Podsumowanie: co się sprawdziło metodycznie

**Testowanie retrievalu przed podłączeniem LLM.** Wszystkie trzy błędy merytoryczne (BOM, gubione sekcje, wybór modelu embeddingowego) dotyczyły warstwy wyszukiwania, nie generowania. Gdyby pipeline był testowany dopiero jako całość, objawiłyby się jako „model czasem odpowiada bez sensu" i diagnoza zajęłaby wielokrotnie więcej czasu.

**Testy sprawdzające treść, nie liczby.** Test „czy zdanie *Wybrałem Chroma* jest w indeksie" wykrył problem, którego test „czy powstały jakieś fragmenty" by przepuścił.

**Odmowa jako gałąź przepływu, nie instrukcja w promptcie.** Węzeł warunkowy odcina wywołanie modelu, gdy żaden fragment nie przeszedł progu podobieństwa. Nie da się zmyślić odpowiedzi, której się nie generuje — a pytania spoza zakresu wracają w 0,0 s zamiast po kilku sekundach.

**Mierzenie zamiast zgadywania.** Próg podobieństwa i parametry wydajności zostały ustawione na podstawie pomiarów (`ollama ps`, czasy z każdego przebiegu), nie na podstawie wartości „które zwykle działają".
