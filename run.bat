@echo off
cd /d "%~dp0"
if not exist venv python -m venv venv
call venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
if not exist .env (
    copy .env.example .env >nul
    echo Created .env - paste your GEMINI_API_KEY into it, save, then run this file again.
    notepad .env
    exit /b
)
streamlit run app.py
