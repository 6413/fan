@echo off
setlocal

set script_dir=%~dp0

set input_exe=%~1
if "%input_exe%"=="" set input_exe=fan.exe

set default_name=%~n1
if "%default_name%"=="" set default_name=fan

set /p newname=New exe name [%default_name%]:

if "%newname%"=="" set newname=%default_name%

set outdir=%script_dir%export_minimal

python "%script_dir%export.py" "%input_exe%" "%outdir%" --force

rename "%outdir%\%~nx1" "%newname%.exe"

pause