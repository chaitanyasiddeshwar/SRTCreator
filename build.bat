@echo off
rem Builds srt.exe: whisper.cpp (CUDA) + minimized FFmpeg audio path.
rem Discovers MSVC like Axiom (vswhere -> VsDevCmd), then drives CMake.
rem
rem   build.bat            configure + build (Release)
rem   build.bat clean      wipe build\ and rebuild from scratch
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem --- locate MSVC (VS 2022 Build Tools or full VS) ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [build] vswhere.exe not found - install VS Build Tools with the C++ x64 toolset.
    exit /b 1
)
set "VSPATH="
for /f "tokens=*" %%i in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VSPATH=%%i"
if "%VSPATH%"=="" (
    echo [build] No installation with the MSVC C++ x64 toolset found.
    exit /b 1
)
echo [build] MSVC: %VSPATH%
call "%VSPATH%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 (
    echo [build] VsDevCmd.bat failed.
    exit /b 1
)

rem Ninja (bundled with VS's CMake component) drives nvcc directly, avoiding the
rem "No CUDA toolset found" failure of the Visual Studio generator when CUDA's
rem MSBuild integration isn't installed into VS.
set "NINJADIR=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "%NINJADIR%\ninja.exe" set "PATH=%NINJADIR%;%PATH%"

rem --- CUDA toolkit ---
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo [build] nvcc not found at %CUDA_PATH%. Set CUDA_PATH to your CUDA Toolkit.
    exit /b 1
)
echo [build] CUDA: %CUDA_PATH%

rem --- FFmpeg dev libs must be provided (see README) ---
if not exist "%~dp0third_party\ffmpeg\include" (
    echo [build] Missing FFmpeg dev libs at third_party\ffmpeg\ ^(need include\, lib\, bin\^).
    echo [build] See README.md / CLAUDE.md section 4.
    exit /b 1
)

rem --- clean ---
if /i "%~1"=="clean" (
    echo [build] clean: removing build\
    if exist build rmdir /s /q build
)

if not exist build mkdir build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo [build] cmake configure FAILED
    exit /b 1
)
cmake --build build
if errorlevel 1 (
    echo [build] build FAILED
    exit /b 1
)
echo [build] OK: %~dp0build\srt.exe
exit /b 0
