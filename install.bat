@echo off
title Installazione Dipendenze - Yuki Assistant
color 0A

echo [1/3] Aggiornamento pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installazione librerie da requirements.txt...
pip install -r requirements.txt

echo.
echo [3/3] Installazione completata!
echo.
pause
