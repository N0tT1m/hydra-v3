@echo off
REM Run the Hydra coordinator (Windows)

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set CONFIG_FILE=%1
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config.toml

REM Check if coordinator is built
if not exist "build\bin\hydra.exe" (
    echo Coordinator not built. Building now...
    go build -o build\bin\hydra.exe .\cmd\hydra
)

REM Check if config exists
if not exist "%CONFIG_FILE%" (
    echo Config file not found: %CONFIG_FILE%
    echo Using default configuration...
    set CONFIG_FILE=
)

echo Starting Hydra Coordinator...
echo   HTTP API: http://localhost:8080
echo   ZMQ Router: tcp://*:5555
echo   ZMQ Metrics: tcp://*:5556
echo   ZMQ Broadcast: tcp://*:5557
echo.
echo Press Ctrl+C to stop
echo.

if defined CONFIG_FILE (
    build\bin\hydra.exe -config "%CONFIG_FILE%"
) else (
    build\bin\hydra.exe
)
