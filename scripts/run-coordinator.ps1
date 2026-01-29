# Run the Hydra coordinator (Windows PowerShell)

param(
    [string]$ConfigFile = "config.toml",
    [switch]$WithLocalWorker,
    [string]$WorkerNodeId = "local-worker",
    [string]$WorkerDevice = "auto"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

# Setup ZeroMQ environment from vcpkg if available
$vcpkgRoot = "C:\vcpkg"
$zmqLibDir = "$vcpkgRoot\installed\x64-windows\lib"

# Check for GCC (required for CGO)
$gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gccCmd) {
    Write-Host "ERROR: GCC not found. Install MinGW-w64:" -ForegroundColor Red
    Write-Host "  choco install mingw -y" -ForegroundColor White
    Write-Host "Then restart PowerShell and try again." -ForegroundColor Yellow
    exit 1
}

if (Test-Path $zmqLibDir) {
    $zmqLibFile = Get-ChildItem -Path $zmqLibDir -Filter "libzmq*.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($zmqLibFile) {
        $zmqLibName = $zmqLibFile.BaseName
        $env:CGO_CFLAGS = "-I$vcpkgRoot\installed\x64-windows\include"
        $env:CGO_LDFLAGS = "-L$zmqLibDir -l$zmqLibName"
        $env:CGO_ENABLED = "1"
        if ($env:PATH -notlike "*$vcpkgRoot\installed\x64-windows\bin*") {
            $env:PATH = "$vcpkgRoot\installed\x64-windows\bin;$env:PATH"
        }
        Write-Host "ZeroMQ environment configured from vcpkg" -ForegroundColor Green
    }
} else {
    Write-Host "ERROR: vcpkg ZeroMQ not found at $zmqLibDir" -ForegroundColor Red
    Write-Host "Run .\scripts\install-zeromq-windows.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Check if coordinator is built
if (-not (Test-Path "build\bin\hydra.exe")) {
    Write-Host "Coordinator not built. Building now..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path "build\bin" | Out-Null
    go build -o build\bin\hydra.exe .\cmd\hydra

    if (-not (Test-Path "build\bin\hydra.exe")) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
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
if ($WithLocalWorker) {
    Write-Host "  Local Worker: $WorkerNodeId ($WorkerDevice)" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Build argument list
$args = @()
if ($ConfigFile) {
    $args += "-config", $ConfigFile
}
if ($WithLocalWorker) {
    $args += "-with-local-worker"
    $args += "-worker-node-id", $WorkerNodeId
    $args += "-worker-device", $WorkerDevice
}

if ($args.Count -gt 0) {
    & .\build\bin\hydra.exe @args
} else {
    & .\build\bin\hydra.exe
}
