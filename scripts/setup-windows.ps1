# Hydra V3 Setup Script for Windows
# Run as Administrator: powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Hydra V3 Setup - Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check for Administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Function to refresh environment variables
function Refresh-Environment {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    # Also refresh ChocolateyInstall
    $chocoPath = [System.Environment]::GetEnvironmentVariable("ChocolateyInstall", "Machine")
    if ($chocoPath) {
        $env:ChocolateyInstall = $chocoPath
        $env:Path = "$chocoPath\bin;" + $env:Path
    }
}

# Check for Chocolatey
$chocoInstalled = $false
$chocoPath = "$env:ProgramData\chocolatey\bin\choco.exe"

if (Test-Path $chocoPath) {
    $chocoInstalled = $true
    Write-Host "Chocolatey already installed." -ForegroundColor Green
} elseif (Get-Command choco -ErrorAction SilentlyContinue) {
    $chocoInstalled = $true
    Write-Host "Chocolatey already installed." -ForegroundColor Green
}

if (-not $chocoInstalled) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Green
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
}

# Refresh environment to get choco in path
Refresh-Environment

# Verify choco is available
if (-not (Test-Path "$env:ProgramData\chocolatey\bin\choco.exe")) {
    Write-Host "ERROR: Chocolatey installation failed or not found." -ForegroundColor Red
    Write-Host "Please install Chocolatey manually: https://chocolatey.org/install" -ForegroundColor Yellow
    exit 1
}

# Use full path to choco
$choco = "$env:ProgramData\chocolatey\bin\choco.exe"

Write-Host "Installing system dependencies..." -ForegroundColor Green

# Install Go
Write-Host "Checking Go..." -ForegroundColor Green
$goInstalled = Get-Command go -ErrorAction SilentlyContinue
if (-not $goInstalled) {
    Write-Host "Installing Go..." -ForegroundColor Green
    & $choco install golang -y --no-progress
    Refresh-Environment
}

# Install Python
Write-Host "Checking Python..." -ForegroundColor Green
$pythonInstalled = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonInstalled) {
    Write-Host "Installing Python 3.11..." -ForegroundColor Green
    & $choco install python311 -y --no-progress
    Refresh-Environment
}

# Install Git if not present
Write-Host "Checking Git..." -ForegroundColor Green
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitInstalled) {
    Write-Host "Installing Git..." -ForegroundColor Green
    & $choco install git -y --no-progress
    Refresh-Environment
}

# Install pkg-config
Write-Host "Installing pkg-config..." -ForegroundColor Green
& $choco install pkgconfiglite -y --no-progress 2>$null
Refresh-Environment

# Refresh one more time
Refresh-Environment

# Verify installations
Write-Host ""
Write-Host "Verifying installations..." -ForegroundColor Green

try {
    $goVersion = & go version 2>&1
    Write-Host "  Go: $goVersion" -ForegroundColor Cyan
} catch {
    Write-Host "  Go: NOT FOUND - please install manually" -ForegroundColor Red
}

try {
    $pythonVersion = & python --version 2>&1
    Write-Host "  Python: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "  Python: NOT FOUND - please install manually" -ForegroundColor Red
}

# Get project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan

# Setup Python virtual environment first (doesn't require ZMQ)
Write-Host ""
Write-Host "Setting up Python virtual environment..." -ForegroundColor Green
Set-Location worker

if (-not (Test-Path "venv")) {
    & python -m venv venv
}

# Activate and install
& .\venv\Scripts\Activate.ps1

# Upgrade pip
& python -m pip install --upgrade pip --quiet

# Install worker package
Write-Host "Installing Python worker dependencies..." -ForegroundColor Green
& pip install -e . --quiet

# Check for CUDA
Write-Host ""
Write-Host "Checking GPU support..." -ForegroundColor Green
try {
    $cudaCheck = & python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>&1
    if ($cudaCheck -eq "CUDA") {
        Write-Host "  CUDA is available" -ForegroundColor Green
        $gpuName = & python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
        Write-Host "  GPU: $gpuName" -ForegroundColor Cyan
    } else {
        Write-Host "  CUDA not detected. Using CPU." -ForegroundColor Yellow
        Write-Host "  For GPU support, install PyTorch with CUDA:" -ForegroundColor Yellow
        Write-Host "    pip install torch --index-url https://download.pytorch.org/whl/cu118" -ForegroundColor White
    }
} catch {
    Write-Host "  Could not detect GPU support" -ForegroundColor Yellow
}

& deactivate
Set-Location $ProjectRoot

# Try to build Go coordinator
Write-Host ""
Write-Host "Attempting to build Go coordinator..." -ForegroundColor Green

New-Item -ItemType Directory -Force -Path "build\bin" | Out-Null

# Check for vcpkg ZeroMQ installation
$vcpkgRoot = "C:\vcpkg"
$zmqLibDir = "$vcpkgRoot\installed\x64-windows\lib"
$zmqFound = $false

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
        Write-Host "  Found ZeroMQ in vcpkg: $zmqLibName" -ForegroundColor Green
        $zmqFound = $true
    }
}

if (-not $zmqFound) {
    Write-Host "  ZeroMQ not found. Run .\scripts\install-zeromq-windows.ps1 first" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  RECOMMENDED: Use WSL2 for the coordinator:" -ForegroundColor Cyan
    Write-Host "    1. Install WSL2: wsl --install" -ForegroundColor White
    Write-Host "    2. In WSL2: ./scripts/setup-linux.sh" -ForegroundColor White
    Write-Host "    3. In WSL2: ./scripts/run-coordinator.sh" -ForegroundColor White
    Write-Host "    4. On Windows: run workers with .\scripts\run-worker.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "  Or install ZeroMQ via vcpkg:" -ForegroundColor Cyan
    Write-Host "    .\scripts\install-zeromq-windows.ps1" -ForegroundColor White
} else {
    try {
        & go mod download 2>&1 | Out-Null
        & go mod tidy 2>&1 | Out-Null
        & go build -o build\bin\hydra.exe .\cmd\hydra 2>&1

        if (Test-Path "build\bin\hydra.exe") {
            Write-Host "  Coordinator built successfully!" -ForegroundColor Green
        } else {
            throw "Build failed"
        }
    } catch {
        Write-Host "  Coordinator build failed" -ForegroundColor Yellow
        Write-Host "  Error: $_" -ForegroundColor Red
    }
}

# Create default config if not exists
if (-not (Test-Path "config.toml")) {
    Write-Host ""
    Write-Host "Creating default configuration..." -ForegroundColor Green
    Copy-Item "config.example.toml" "config.toml"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Python worker is ready to use." -ForegroundColor Green
Write-Host ""
Write-Host "To start a worker:" -ForegroundColor Cyan
Write-Host "  .\scripts\run-worker.ps1 -NodeId worker-1" -ForegroundColor White
Write-Host ""
Write-Host "For the coordinator, we recommend WSL2:" -ForegroundColor Cyan
Write-Host "  wsl --install                         # Install WSL2" -ForegroundColor White
Write-Host "  wsl                                    # Enter WSL2" -ForegroundColor White
Write-Host "  cd /mnt/d/workspace/ai-apps/hydra-v3  # Navigate to project" -ForegroundColor White
Write-Host "  ./scripts/setup-linux.sh              # Setup in Linux" -ForegroundColor White
Write-Host "  ./scripts/run-coordinator.sh          # Run coordinator" -ForegroundColor White
Write-Host ""
Write-Host "The coordinator in WSL2 will be accessible from Windows at localhost:8080" -ForegroundColor Yellow
Write-Host ""
