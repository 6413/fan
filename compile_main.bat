@echo off
setlocal enabledelayedexpansion

if not exist build mkdir build
cd build

echo Running CMake...
cmake .. -G Ninja %*
if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

echo Building project...
cmake --build .
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Moving executable...
if exist a.exe (
    move a.exe ..
    echo Build completed successfully!
) else (
    echo Warning: a.exe not found after build
    exit /b 1
)

cd ..