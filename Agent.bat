@echo off
chcp 65001 >nul
title Personal Knowledge Agent
color 0B

rem ============================================================
rem  Skrot na pulpit - uruchamia calego agenta jednym kliknieciem
rem
rem  ZAWSZE restartuje serwer. Wczesniej bylo menu z wyborem
rem  [O]twórz / [R]estart / [A]nuluj, ale "choice" w nawiasowym
rem  bloku "if" gubi errorlevel i skrypt zawsze spadal do gałęzi
rem  "otworz przegladarke" - czyli zostawial stary kod w pamieci.
rem  Restart trwa kilka sekund i jest zawsze poprawny, wiec wybor
rem  byl niepotrzebna komplikacja.
rem ============================================================

rem %~dp0 to folder, w ktorym lezy TEN plik - dziala niezaleznie od nazwy
rem uzytkownika i od tego, gdzie stoi skrot. Wczesniej byla tu sciezka na sztywno.
set "PROJEKT=%~dp0"
if not exist "%PROJEKT%search_api.py" set "PROJEKT=%USERPROFILE%\personal-knowledge-agent"
set "NOTATKI=%USERPROFILE%\Notatki"
set "ADRES=http://127.0.0.1:8000"
set "PORT=8000"

cd /d "%PROJEKT%" 2>nul
if errorlevel 1 (
    echo.
    echo   BLAD: nie znaleziono folderu projektu:
    echo   %PROJEKT%
    echo.
    pause
    exit /b 1
)

cls
echo.
echo   ============================================================
echo     PERSONAL KNOWLEDGE AGENT
echo   ============================================================
echo.

rem --- 1. Zatrzymaj to, co juz stoi na porcie ---
rem     UWAGA: w wyniku netstat port stoi PRZED slowem LISTENING,
rem     dlatego filtrujemy dwuetapowo zamiast jednym wyrazeniem.
echo   [1/4] Zwalniam port %PORT%...
set "COKOLWIEK="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo         zatrzymuje stary serwer, PID %%p
    taskkill /PID %%p /F >nul 2>&1
    set "COKOLWIEK=tak"
)
if not defined COKOLWIEK echo         port byl wolny
timeout /t 2 >nul

rem --- 2. Czy Ollama odpowiada? ---
echo   [2/4] Sprawdzam Ollama...
curl -s -o nul -m 5 http://127.0.0.1:11434/api/tags 2>nul
if errorlevel 1 (
    echo         nie odpowiada - uruchamiam w tle...
    start "Ollama" /min ollama serve
    timeout /t 5 >nul
) else (
    echo         OK
)

rem --- 3. Aktualizacja indeksu ---
echo   [3/4] Aktualizuje indeks notatek...
if not exist "%NOTATKI%" (
    echo         UWAGA: brak folderu %NOTATKI% - baza bedzie pusta.
) else (
    python indexer.py --path "%NOTATKI%" 2>nul | findstr /C:"nowe pliki" /C:"zmienione pliki" /C:"fragmentow w bazie"
)

rem --- 4. Serwer + przegladarka ---
echo   [4/4] Startuje serwer...
echo.
echo   ------------------------------------------------------------
echo     Adres:  %ADRES%
echo.
echo     Tryby:  Notatki+model - domyslny; notatki, a gdy ich brak -
echo                             wiedza modelu, zawsze z plakietka
echo             Tylko notatki - rygorystyczny, odmawia zamiast zmyslac
echo             Rozmowa       - zwykly czat z modelem, bez notatek
echo             Szukanie      - sam retrieval, do diagnozy
echo.
echo     ZAMKNIECIE TEGO OKNA WYLACZA AGENTA.
echo   ------------------------------------------------------------
echo.

rem Przegladarka dopiero po chwili - inaczej otwiera sie, zanim
rem uvicorn zdazy zajac port, i pokazuje blad polaczenia.
start "" /min cmd /c "timeout /t 4 >nul & start %ADRES%"

python -m uvicorn search_api:app --host 127.0.0.1 --port %PORT%

echo.
echo   Serwer zatrzymany.
pause
