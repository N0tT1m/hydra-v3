# Hydra V3 Setup Script for Windows
# Run as Administrator: powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Hydra V3 Setup - Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check for Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some installations may fail." -ForegroundColor Yellow
}

# Check for Chocolatey
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Green
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Install dependencies via Chocolatey
Write-Host "Installing system dependencies..." -ForegroundColor Green

# Install Go
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Go..." -ForegroundColor Green
    choco install golang -y
}

# Install Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Python..." -ForegroundColor Green
    choco install python311 -y
}

# Install pkg-config and ZeroMQ
Write-Host "Installing ZeroMQ..." -ForegroundColor Green
choco install pkgconfiglite -y

# For ZeroMQ on Windows, we need vcpkg or manual installation
# Using pre-built binaries approach
$zmqUrl = "https://github.com/zeromq/libzmq/releases/download/v4.3.5/zeromq-4.3.5.zip"
$zmqDir = "$env:LOCALAPPDATA\zeromq"

if (-not (Test-Path "$zmqDir\bin\libzmq.dll")) {
    Write-Host "Downloading ZeroMQ..." -ForegroundColor Green

    $zmqZip = "$env:TEMP\zeromq.zip"
    Invoke-WebRequest -Uri $zmqUrl -OutFile $zmqZip

    New-Item -ItemType Directory -Force -Path $zmqDir | Out-Null
    Expand-Archive -Path $zmqZip -DestinationPath $zmqDir -Force
    Remove-Item $zmqZip

    # Add to PATH
    $env:Path += ";$zmqDir\bin"
    [Environment]::SetEnvironmentVariable("Path", $env:Path, "User")

    # Set PKG_CONFIG_PATH
    [Environment]::SetEnvironmentVariable("PKG_CONFIG_PATH", "$zmqDir\lib\pkgconfig", "User")
}

# Refresh environment
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "Go version: $(go version)" -ForegroundColor Green
Write-Host "Python version: $(python --version)" -ForegroundColor Green

# Get project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Build Go coordinator
Write-Host "Building Go coordinator..." -ForegroundColor Green

go mod download
go mod tidy

New-Item -ItemType Directory -Force -Path "build\bin" | Out-Null

$env:CGO_ENABLED = "1"
go build -o build\bin\hydra.exe .\cmd\hydra

if (Test-Path "build\bin\hydra.exe") {
    Write-Host "Coordinator built successfully: build\bin\hydra.exe" -ForegroundColor Green
} else {
    Write-Host "Coordinator build failed" -ForegroundColor Red
    Write-Host "Note: ZeroMQ CGO build on Windows can be tricky." -ForegroundColor Yellow
    Write-Host "Consider using WSL2 for easier setup." -ForegroundColor Yellow
}

# Setup Python virtual environment
Write-Host "Setting up Python virtual environment..." -ForegroundColor Green
Set-Location worker

if (-not (Test-Path "venv")) {
    python -m venv venv
}

& .\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install worker package
Write-Host "Installing Python worker..." -ForegroundColor Green
pip install -e .

# Check for CUDA
$cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvailable -eq "True") {
    Write-Host "CUDA is available" -ForegroundColor Green
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
} else {
    Write-Host "CUDA not detected. Install PyTorch with CUDA support for GPU acceleration:" -ForegroundColor Yellow
    Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor Yellow
}

deactivate
Set-Location $ProjectRoot

# Create default config if not exists
if (-not (Test-Path "config.toml")) {
    Write-Host "Creating default configuration..." -ForegroundColor Green
    Copy-Item "config.example.toml" "config.toml"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the coordinator:"
Write-Host "  .\scripts\run-coordinator.bat"
Write-Host ""
Write-Host "To start a worker:"
Write-Host "  .\scripts\run-worker.bat --node-id worker-1"
Write-Host ""
Write-Host "Or manually:"
Write-Host "  .\build\bin\hydra.exe -config config.toml"
Write-Host "  cd worker; .\venv\Scripts\Activate.ps1; hydra-worker start --node-id worker-1"
Write-Host ""
