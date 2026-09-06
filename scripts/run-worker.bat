@echo off
REM Run a Hydra worker (Windows)

setlocal enabledelayedexpansion

cd /d "%~dp0\..\worker"

REM Default values
set NODE_ID=
set COORDINATOR=tcp://localhost:5555
set DEVICE=auto
set DTYPE=float16
set PIPELINE_PORT=6000

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--node-id" (
    set NODE_ID=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-n" (
    set NODE_ID=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--coordinator" (
    set COORDINATOR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-c" (
    set COORDINATOR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-d" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--dtype" (
    set DTYPE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--pipeline-port" (
    set PIPELINE_PORT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-p" (
    set PIPELINE_PORT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help

echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: run-worker.bat [OPTIONS]
echo.
echo Options:
echo   --node-id, -n       Unique worker ID (required)
echo   --coordinator, -c   Coordinator address (default: tcp://localhost:5555)
echo   --device, -d        Device to use: auto, cuda:0, cuda:1, cpu (default: auto)
echo   --dtype             Data type: float16, bfloat16, float32 (default: float16)
echo   --pipeline-port, -p Pipeline port (default: 6000)
echo.
echo Examples:
echo   run-worker.bat --node-id worker-1
echo   run-worker.bat --node-id worker-2 --device cuda:1 --pipeline-port 6001
exit /b 0

:done_parsing

REM Validate required args
if "%NODE_ID%"=="" (
    echo Error: --node-id is required
    echo Run with --help for usage
    exit /b 1
)

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Run setup script first.
    exit /b 1
)

echo Starting Hydra Worker...
echo   Node ID: %NODE_ID%
echo   Coordinator: %COORDINATOR%
echo   Device: %DEVICE%
echo   Dtype: %DTYPE%
echo   Pipeline Port: %PIPELINE_PORT%
echo.
echo Press Ctrl+C to stop
echo.

hydra-worker start --node-id "%NODE_ID%" --coordinator "%COORDINATOR%" --device "%DEVICE%" --dtype "%DTYPE%" --pipeline-port "%PIPELINE_PORT%"
