# Jak z tego realnie korzystać

## Codziennie: dwa kliknięcia

**`nowa-notatka.bat`** — pyta o folder i nazwę, tworzy plik ze szkieletem (frontmatter + nagłówki), otwiera w Notatniku, a po zamknięciu sam aktualizuje indeks. Enter przy folderze = `inbox`, czyli zero decyzji przy zapisie.

**`pytaj.bat`** — najpierw dociąga zmiany w notatkach, potem otwiera tryb pytań. Pusta linia kończy.

Przypnij oba do paska zadań: prawy przycisk na pliku → **Przypnij do menu Start**. Wtedy cały cykl to dwa kliknięcia zamiast otwierania terminala i pamiętania o `indexer.py`.

## Kiedy to zacznie być użyteczne

Przy 14 notatkach agent nie ma przewagi nad Twoją pamięcią — pamiętasz, co pisałeś w zeszłym tygodniu. Wartość pojawia się przy **50–100 notatkach** i rośnie z czasem, bo dopiero wtedy zaczynasz zapominać własne decyzje i ich uzasadnienia.

To narzędzie spłaca się po miesiącach, nie po tygodniu. Największym ryzykiem nie jest jakość wyszukiwania, tylko to, że przestaniesz pisać notatki.

## Co warto zapisywać

Kryterium jest jedno: **czy za pół roku będę tego szukał i czy będę pamiętał**.

Dobrze się sprawdza:

- decyzje z uzasadnieniem — zwłaszcza to, co **odrzuciłeś** i dlaczego
- ustalenia z rozmów: kto, co, do kiedy
- rzeczy, które sprawiły kłopot i jak zostały rozwiązane (błąd, przyczyna, obejście)
- wnioski z projektów, książek, artykułów — własnymi słowami, nie streszczenie
- liczby i progi, które kiedyś ustaliłeś pomiarem

Słabo się sprawdza: listy zadań (mają swoje miejsce), notatki jednozdaniowe bez kontekstu, kopiowanie cudzych tekstów bez własnego komentarza.

## Typowe pytania, na które to odpowiada

- „Dlaczego wtedy wybrałem X zamiast Y?"
- „Co ustaliłem z tą osobą w sprawie Z?"
- „Miałem już ten błąd — jak go rozwiązałem?"
- „Jaki próg / jaką kwotę / jaki parametr wtedy ustaliłem?"
- „Co wyszło z tamtego projektu i czego się nauczyłem?"

## Automatyczne indeksowanie w tle (opcjonalnie)

Jeśli wolisz nie pamiętać nawet o `pytaj.bat`, ustaw indeksowanie co noc.

Utwórz `reindex.bat`:

```bat
@echo off
cd /d "%~dp0"
python indexer.py --path "%USERPROFILE%\Notatki" >> reindex.log 2>&1
```

Potem: **Harmonogram zadań** → Utwórz zadanie podstawowe → codziennie o 23:00 → wskaż `reindex.bat`. Indeksowanie jest inkrementalne, więc przy braku zmian trwa poniżej sekundy.

## Dostęp z zewnątrz (webhook n8n)

Workflow n8n wystawia agenta pod adresem `http://localhost:5678/webhook/pka-ask`. Przydaje się, gdy chcesz zapytać z innego skryptu, z prostego formularza HTML albo podpiąć to pod skrót klawiszowy.

Wymaga dwóch działających procesów:

```powershell
Start-Process powershell -ArgumentList '-NoExit','-Command','cd C:\Users\patry\personal-knowledge-agent; python -m uvicorn search_api:app --host 127.0.0.1 --port 8000'
```

oraz uruchomionego n8n z opublikowanym workflow.

Do codziennego użytku `pytaj.bat` jest wygodniejszy — nie wymaga trzymania niczego w tle.

## Utrzymanie

**Raz na kwartał** przepuść ponownie ewaluację i kalibrację progu:

```powershell
python evaluate.py --calibrate --questions questions.csv
```

Próg 0,52 skalibrowano na 58 fragmentach. Przy kilkuset fragmentach rozkład dystansów się zmienia i wartość wymaga przeliczenia.

**Gdy odpowiedzi zaczną się psuć**, sprawdź w tej kolejności:

1. `python ask.py --search-only "pytanie"` — czy retrieval znajduje właściwą notatkę
2. jeśli tak, a odpowiedź jest zła → problem w prompcie albo w modelu
3. jeśli nie → problem w notatkach: brak treści, zbyt ogólny opis, albo ta sama informacja rozproszona po kilku plikach

Ta druga sytuacja jest najczęstsza i naprawia się pisaniem, nie kodem.
