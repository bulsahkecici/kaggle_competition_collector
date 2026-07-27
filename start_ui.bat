@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Sanal ortam bulunamadi: .venv\Scripts\python.exe
    echo Once README dosyasindaki kurulum adimlarini tamamlayin.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 pause
