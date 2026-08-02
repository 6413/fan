@echo off
setlocal

set script_dir=%~dp0

set input_exe=%~1
if "%input_exe%"=="" set input_exe=fan.exe
set input_exe_name=%~nx1
if "%input_exe_name%"=="" set input_exe_name=fan.exe

set base=%~n1
if "%base%"=="" set base=fan

set /p outname=Output folder name [%base%]:
if "%outname%"=="" set outname=%base%

set outdir=%script_dir%%outname%

set "export_args="
shift
:collect_export_args
if "%~1"=="" goto run_export
set "export_args=%export_args% "%~1""
shift
goto collect_export_args

:run_export
python "%script_dir%export.py" "%input_exe%" "%outdir%" --force %export_args%

rename "%outdir%\%input_exe_name%" "%outname%.exe"

pause
