@echo off
chcp 65001 >nul
title Nowa notatka
cd /d "%~dp0"

set "VAULT=%USERPROFILE%\Notatki"

echo.
echo ================================================================
echo   NOWA NOTATKA
echo ================================================================
echo.
echo   Foldery: decyzje  projekty  zrodla  dziennik  ludzie  cele  meta  inbox
echo   (Enter = inbox, posortujesz pozniej)
echo.
set /p FOLDER="   Folder: "
if "%FOLDER%"=="" set "FOLDER=inbox"

echo.
set /p NAZWA="   Nazwa pliku (bez .md): "
if "%NAZWA%"=="" (
    echo   Nazwa jest wymagana.
    pause
    exit /b 1
)

set "PLIK=%VAULT%\%FOLDER%\%NAZWA%.md"

if not exist "%VAULT%\%FOLDER%" mkdir "%VAULT%\%FOLDER%"

if exist "%PLIK%" (
    echo.
    echo   Plik juz istnieje - otwieram do edycji.
) else (
    call :SZKIELET > "%PLIK%"
)

notepad "%PLIK%"

echo.
echo   Aktualizuje indeks...
python indexer.py --path "%VAULT%" 2>nul | findstr /C:"nowe pliki" /C:"zmienione pliki" /C:"fragmentow w bazie"
echo.
pause
exit /b 0

:SZKIELET
echo ---
echo title: %NAZWA%
echo tags:
echo data: %date%
echo ---
echo.
echo ## Kontekst
echo.
echo.
echo ## Decyzja
echo.
echo.
echo ## Dlaczego
echo.
echo.
echo REM --- przypomnienie ---
echo REM Jeden temat = jedna notatka. Naglowki ## co 200-400 slow.
echo REM Znacznik czasu przy tym, co moze sie zdezaktualizowac ("stan na %date%").
echo REM Pelne zdania, nie skroty - skrot sie nie zaembeduje.
exit /b 0
