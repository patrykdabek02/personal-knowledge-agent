@echo off
chcp 65001 >nul
title Personal Knowledge Agent
cd /d "%~dp0"

echo.
echo ================================================================
echo   PERSONAL KNOWLEDGE AGENT
echo ================================================================
echo.
echo   Aktualizuje indeks...

python indexer.py --path "%USERPROFILE%\Notatki" 2>nul | findstr /C:"nowe pliki" /C:"zmienione pliki" /C:"fragmentow w bazie"

echo.
echo   Pytaj o cokolwiek z notatek. Pusta linia konczy.
echo ================================================================

python ask.py

echo.
pause
