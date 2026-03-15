@echo off
if "%~1"=="" (
  set exe=fan.exe
) else (
  set exe=%~1
)
python export.py %exe% export_minimal --force
pause