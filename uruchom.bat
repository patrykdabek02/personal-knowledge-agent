@echo off
chcp 65001 >nul
title Personal Knowledge Agent - serwer
cd /d "%~dp0"

echo.
echo ================================================================
echo   PERSONAL KNOWLEDGE AGENT
echo ================================================================
echo.
echo   Aktualizuje indeks...

python indexer.py --path "%USERPROFILE%\Notatki" 2>nul | findstr /C:"nowe pliki" /C:"zmienione pliki" /C:"fragmentow w bazie"

echo.
echo   Startuje serwer i otwieram przegladarke...
echo   Zamkniecie tego okna wylacza agenta.
echo.
echo ================================================================
echo.

start "" http://127.0.0.1:8000
python -m uvicorn search_api:app --host 127.0.0.1 --port 8000

pause
