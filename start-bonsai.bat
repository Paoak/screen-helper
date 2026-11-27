@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe main.py --config config.bonsai.json
pause