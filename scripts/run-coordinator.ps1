# Run the Hydra coordinator (Windows PowerShell)
# Supports both PowerShell-style (-WithLocalWorker) and Unix-style (--with-local-worker) parameters

[CmdletBinding()]
param(
    [string]$ConfigFile = "config.toml",
    [switch]$WithLocalWorker,
    [string]$WorkerNodeId = "local-worker",
    [string]$WorkerDevice = "auto",
    [string]$WorkerDtype = "bfloat16",
    [string]$LoadModel = "",
    [string]$ModelId = "",
    [int]$ModelLayers = 0,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

# Handle Unix-style arguments (--with-local-worker, --load-model, etc.)
# PowerShell doesn't natively support double-dash or hyphenated params
$i = 0
while ($i -lt $RemainingArgs.Count) {
    $arg = $RemainingArgs[$i]
    $matched = $false
    switch -Regex ($arg) {
        '^-{1,2}with-local-worker$' { $WithLocalWorker = $true; $matched = $true }
        '^-{1,2}worker-node-id$' { $i++; $WorkerNodeId = $RemainingArgs[$i]; $matched = $true }
        '^-{1,2}worker-device$' { $i++; $WorkerDevice = $RemainingArgs[$i]; $matched = $true }
        '^-{1,2}worker-dtype$' { $i++; $WorkerDtype = $RemainingArgs[$i]; $matched = $true }
        '^-{1,2}load-model$' { $i++; $LoadModel = $RemainingArgs[$i]; $matched = $true }
        '^-{1,2}model-id$' { $i++; $ModelId = $RemainingArgs[$i]; $matched = $true }
        '^-{1,2}model-layers$' { $i++; $ModelLayers = [int]$RemainingArgs[$i]; $matched = $true }
    }
    # Warn about unrecognized flags (but not their values)
    if (-not $matched -and $arg -match '^-') {
        Write-Host "WARNING: Unrecognized option '$arg' - did you mean --worker-device, --worker-dtype, or --worker-node-id?" -ForegroundColor Yellow
    }
    $i++
}

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
    Write-Host "  Local Worker: $WorkerNodeId ($WorkerDevice, $WorkerDtype)" -ForegroundColor Cyan
}
if ($LoadModel) {
    if ($ModelLayers -eq 0) {
        Write-Host "  Auto-load Model: $LoadModel (auto-detect layers)" -ForegroundColor Cyan
    } else {
        Write-Host "  Auto-load Model: $LoadModel ($ModelLayers layers)" -ForegroundColor Cyan
    }
}
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Build argument list for the Go binary
$binArgs = @()
if ($ConfigFile) {
    $binArgs += "-config", $ConfigFile
}
if ($WithLocalWorker) {
    $binArgs += "-with-local-worker"
    $binArgs += "-worker-node-id", $WorkerNodeId
    $binArgs += "-worker-device", $WorkerDevice
    $binArgs += "-worker-dtype", $WorkerDtype
}
if ($LoadModel) {
    $binArgs += "-load-model", $LoadModel
    if ($ModelId) {
        $binArgs += "-model-id", $ModelId
    }
    $binArgs += "-model-layers", $ModelLayers
}

if ($binArgs.Count -gt 0) {
    & .\build\bin\hydra.exe @binArgs
} else {
    & .\build\bin\hydra.exe
}
