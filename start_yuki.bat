@echo off
REM Attiva l'ambiente virtuale se presente
IF EXIST venv\Scripts\activate (
    call venv\Scripts\activate
)

REM Aggiorna pip alla versione più recente
python -m pip install --upgrade pip

REM Avvia Yuki via CLI (grazie al setup.py)
echo Avvio di Yuki...
yuki --debug

pause
