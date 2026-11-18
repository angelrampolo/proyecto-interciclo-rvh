@echo off
cd /d "%~dp0"
python servir_dashboard.py
if errorlevel 1 py servir_dashboard.py
pause
