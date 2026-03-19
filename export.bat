@echo off
setlocal

set script_dir=%~dp0

set input_exe=%~1
if "%input_exe%"=="" set input_exe=fan.exe

set base=%~n1
if "%base%"=="" set base=fan

set /p outname=Output folder name [%base%]:
if "%outname%"=="" set outname=%base%

set outdir=%script_dir%%outname%

python "%script_dir%export.py" "%input_exe%" "%outdir%" --force

rename "%outdir%\%~nx1" "%outname%.exe"

pause