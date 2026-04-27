@echo off
echo ======================================================
echo  Syllable-Constrained Neural Machine Translator (GUI)
echo ======================================================
echo Activating nlp_venv...

if exist .\nlp_venv\Scripts\activate.bat (
    call .\nlp_venv\Scripts\activate.bat
    echo Environment Activated Successfully.
    echo Launching GUI...
    echo.
    python translator_gui.py
) else (
    echo [ERROR] Virtual Environment 'nlp_venv' was not found!
    echo Please ensure the folder exists in this directory.
    pause
)
pause
