# Run the Hydra coordinator (Windows PowerShell)

param(
    [string]$ConfigFile = "config.toml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Check if coordinator is built
if (-not (Test-Path "build\bin\hydra.exe")) {
    Write-Host "Coordinator not built. Building now..." -ForegroundColor Yellow
    go build -o build\bin\hydra.exe .\cmd\hydra
}

# Check if config exists
if (-not (Test-Path $ConfigFile)) {
    Write-Host "Config file not found: $ConfigFile" -ForegroundColor Yellow
    Write-Host "Using default configuration..." -ForegroundColor Yellow
    $ConfigFile = $null
}

Write-Host "Starting Hydra Coordinator..." -ForegroundColor Green
Write-Host "  HTTP API: http://localhost:8080" -ForegroundColor Cyan
Write-Host "  ZMQ Router: tcp://*:5555" -ForegroundColor Cyan
Write-Host "  ZMQ Metrics: tcp://*:5556" -ForegroundColor Cyan
Write-Host "  ZMQ Broadcast: tcp://*:5557" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

if ($ConfigFile) {
    & .\build\bin\hydra.exe -config $ConfigFile
} else {
    & .\build\bin\hydra.exe
}
