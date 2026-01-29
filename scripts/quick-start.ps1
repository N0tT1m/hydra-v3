# Quick start script - runs coordinator and one worker (Windows)
# Useful for testing on a single machine

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Hydra V3 Quick Start" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if built
if (-not (Test-Path "$ProjectRoot\build\bin\hydra.exe")) {
    Write-Host "Coordinator not built. Run setup script first:" -ForegroundColor Red
    Write-Host "  .\scripts\setup-windows.ps1" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "$ProjectRoot\worker\venv")) {
    Write-Host "Worker not installed. Run setup script first." -ForegroundColor Red
    exit 1
}

# Start coordinator in new window
Write-Host "Starting coordinator..." -ForegroundColor Green
$coordProcess = Start-Process -FilePath "$ProjectRoot\build\bin\hydra.exe" `
    -WorkingDirectory $ProjectRoot `
    -PassThru `
    -WindowStyle Normal

Start-Sleep -Seconds 2

if ($coordProcess.HasExited) {
    Write-Host "Coordinator failed to start" -ForegroundColor Red
    exit 1
}

Write-Host "Coordinator started (PID: $($coordProcess.Id))" -ForegroundColor Green

# Start worker in new window
Write-Host "Starting worker..." -ForegroundColor Green

$workerScript = @"
Set-Location '$ProjectRoot\worker'
& .\venv\Scripts\Activate.ps1
hydra-worker start --node-id worker-1 --coordinator tcp://localhost:5555
"@

$workerProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", $workerScript `
    -PassThru `
    -WindowStyle Normal

Start-Sleep -Seconds 2

Write-Host "Worker started (PID: $($workerProcess.Id))" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Hydra V3 Running" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Coordinator: http://localhost:8080" -ForegroundColor Cyan
Write-Host "Health check: curl http://localhost:8080/health" -ForegroundColor Cyan
Write-Host "Cluster status: curl http://localhost:8080/api/cluster/status" -ForegroundColor Cyan
Write-Host ""
Write-Host "To load a model:" -ForegroundColor Yellow
Write-Host '  Invoke-RestMethod -Uri "http://localhost:8080/api/models/load" -Method Post -ContentType "application/json" -Body ''{"model_path": "meta-llama/Llama-2-7b-hf", "model_id": "llama-7b"}''' -ForegroundColor White
Write-Host ""
Write-Host "Press Enter to stop both processes..." -ForegroundColor Yellow

Read-Host

Write-Host "Shutting down..." -ForegroundColor Yellow

if (-not $coordProcess.HasExited) {
    Stop-Process -Id $coordProcess.Id -Force -ErrorAction SilentlyContinue
}
if (-not $workerProcess.HasExited) {
    Stop-Process -Id $workerProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "Done." -ForegroundColor Green
