@echo off
echo ======================================================
echo  Syllable-Constrained Neural Machine Translator (Vibe)
echo ======================================================
echo Activating nlp_venv...

:: Activate the virtual environment if it exists
if exist .\nlp_venv\Scripts\activate.bat (
    call .\nlp_venv\Scripts\activate.bat
    echo Environment Activated Successfully.
    echo Running Translator...
    echo.
    python semantic_translator.py
) else (
    echo [ERROR] Virtual Environment 'nlp_venv' was not found!
    echo Please ensure the folder exists in this directory.
    pause
)
pause
