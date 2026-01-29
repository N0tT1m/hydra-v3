# Install ZeroMQ on Windows using vcpkg
# Run as Administrator: powershell -ExecutionPolicy Bypass -File scripts\install-zeromq-windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ZeroMQ Installation for Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check for Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Running without Administrator privileges. Some steps may fail." -ForegroundColor Yellow
}

# Check for Git
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "ERROR: Git is required. Install from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Check for Visual Studio Build Tools or Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasVS = $false

if (Test-Path $vsWhere) {
    $vsInstalls = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ($vsInstalls) {
        $hasVS = $true
        Write-Host "Found Visual Studio with C++ tools" -ForegroundColor Green
    }
}

if (-not $hasVS) {
    Write-Host ""
    Write-Host "Visual Studio Build Tools not found." -ForegroundColor Yellow
    Write-Host "Installing Visual Studio Build Tools..." -ForegroundColor Green

    # Download and install Build Tools
    $vsUrl = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
    $vsInstaller = "$env:TEMP\vs_BuildTools.exe"

    Write-Host "Downloading Visual Studio Build Tools..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $vsUrl -OutFile $vsInstaller -UseBasicParsing

    Write-Host "Installing (this may take 10-15 minutes)..." -ForegroundColor Cyan
    Start-Process -FilePath $vsInstaller -ArgumentList `
        "--quiet", "--wait", "--norestart", `
        "--add", "Microsoft.VisualStudio.Workload.VCTools", `
        "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", `
        "--add", "Microsoft.VisualStudio.Component.Windows11SDK.22621" `
        -Wait -NoNewWindow

    Remove-Item $vsInstaller -Force -ErrorAction SilentlyContinue
    Write-Host "Visual Studio Build Tools installed" -ForegroundColor Green
}

# Set vcpkg directory
$vcpkgRoot = "C:\vcpkg"

# Install vcpkg if not present
if (-not (Test-Path "$vcpkgRoot\vcpkg.exe")) {
    Write-Host ""
    Write-Host "Installing vcpkg..." -ForegroundColor Green

    if (Test-Path $vcpkgRoot) {
        Remove-Item $vcpkgRoot -Recurse -Force
    }

    git clone https://github.com/Microsoft/vcpkg.git $vcpkgRoot

    Set-Location $vcpkgRoot
    .\bootstrap-vcpkg.bat

    if (-not (Test-Path "$vcpkgRoot\vcpkg.exe")) {
        Write-Host "ERROR: vcpkg installation failed" -ForegroundColor Red
        exit 1
    }

    Write-Host "vcpkg installed successfully" -ForegroundColor Green
} else {
    Write-Host "vcpkg already installed at $vcpkgRoot" -ForegroundColor Green
}

# Install ZeroMQ
Write-Host ""
Write-Host "Installing ZeroMQ (this may take a few minutes)..." -ForegroundColor Green

Set-Location $vcpkgRoot
.\vcpkg install zeromq:x64-windows

# Check for ZeroMQ library - vcpkg uses versioned names like libzmq-mt-4_3_5.lib
$zmqLibs = Get-ChildItem -Path "$vcpkgRoot\installed\x64-windows\lib" -Filter "*zmq*.lib" -ErrorAction SilentlyContinue
if (-not $zmqLibs) {
    Write-Host "ERROR: ZeroMQ installation failed - no library files found" -ForegroundColor Red
    exit 1
}

Write-Host "ZeroMQ installed successfully" -ForegroundColor Green
Write-Host "  Found library: $($zmqLibs[0].Name)" -ForegroundColor Cyan

# Integrate vcpkg
Write-Host ""
Write-Host "Integrating vcpkg with system..." -ForegroundColor Green
.\vcpkg integrate install

# Set up environment variables permanently
Write-Host ""
Write-Host "Setting environment variables..." -ForegroundColor Green

$zmqInclude = "$vcpkgRoot\installed\x64-windows\include"
$zmqLib = "$vcpkgRoot\installed\x64-windows\lib"
$zmqBin = "$vcpkgRoot\installed\x64-windows\bin"

# Find the actual library name (vcpkg uses versioned names like libzmq-mt-4_3_5.lib)
$zmqLibFile = Get-ChildItem -Path $zmqLib -Filter "libzmq*.lib" | Select-Object -First 1
if ($zmqLibFile) {
    # Extract library name without .lib extension for linker
    $zmqLibName = $zmqLibFile.BaseName
    Write-Host "  Using library: $zmqLibName" -ForegroundColor Cyan
} else {
    $zmqLibName = "zmq"
}

# Set for current session
$env:CGO_CFLAGS = "-I$zmqInclude"
$env:CGO_LDFLAGS = "-L$zmqLib -l$zmqLibName"
$env:CGO_ENABLED = "1"

# Add to PATH for current session
if ($env:PATH -notlike "*$zmqBin*") {
    $env:PATH = "$zmqBin;$env:PATH"
}

# Set permanently (User level)
[System.Environment]::SetEnvironmentVariable("CGO_CFLAGS", "-I$zmqInclude", "User")
[System.Environment]::SetEnvironmentVariable("CGO_LDFLAGS", "-L$zmqLib -l$zmqLibName", "User")
[System.Environment]::SetEnvironmentVariable("CGO_ENABLED", "1", "User")

# Add vcpkg bin to PATH permanently
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentPath -notlike "*$zmqBin*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$zmqBin;$currentPath", "User")
}

# Create a helper script to set up environment
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

$envScript = @"
# ZeroMQ Environment Setup for Hydra
# Source this before building: . .\scripts\zmq-env.ps1

`$env:CGO_CFLAGS = "-I$zmqInclude"
`$env:CGO_LDFLAGS = "-L$zmqLib -l$zmqLibName"
`$env:CGO_ENABLED = "1"
`$env:PATH = "$zmqBin;`$env:PATH"

Write-Host "ZeroMQ environment configured" -ForegroundColor Green
"@

Set-Content -Path "$ProjectRoot\scripts\zmq-env.ps1" -Value $envScript

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  ZeroMQ Installation Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "  CGO_CFLAGS = -I$zmqInclude" -ForegroundColor White
Write-Host "  CGO_LDFLAGS = -L$zmqLib -l$zmqLibName" -ForegroundColor White
Write-Host "  PATH includes $zmqBin" -ForegroundColor White
Write-Host ""
Write-Host "Now build the coordinator:" -ForegroundColor Yellow
Write-Host "  cd $ProjectRoot" -ForegroundColor White
Write-Host "  go build -o build\bin\hydra.exe .\cmd\hydra" -ForegroundColor White
Write-Host ""
Write-Host "Or run the full setup:" -ForegroundColor Yellow
Write-Host "  .\scripts\setup-windows.ps1" -ForegroundColor White
Write-Host ""
Write-Host "NOTE: You may need to restart PowerShell for PATH changes to take effect." -ForegroundColor Yellow
Write-Host ""

# Try to build now
Write-Host "Attempting to build coordinator now..." -ForegroundColor Green
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force -Path "build\bin" | Out-Null

try {
    & go build -o build\bin\hydra.exe .\cmd\hydra 2>&1 | Tee-Object -Variable buildOutput

    if (Test-Path "build\bin\hydra.exe") {
        Write-Host ""
        Write-Host "SUCCESS! Coordinator built: build\bin\hydra.exe" -ForegroundColor Green
        Write-Host ""
        Write-Host "To run:" -ForegroundColor Cyan
        Write-Host "  .\scripts\run-coordinator.ps1" -ForegroundColor White
        Write-Host "  # or" -ForegroundColor Gray
        Write-Host "  .\build\bin\hydra.exe" -ForegroundColor White
    } else {
        throw "Build did not produce executable"
    }
} catch {
    Write-Host ""
    Write-Host "Build failed. Error output:" -ForegroundColor Red
    Write-Host $buildOutput -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Try:" -ForegroundColor Cyan
    Write-Host "  1. Close and reopen PowerShell" -ForegroundColor White
    Write-Host "  2. Run: . .\scripts\zmq-env.ps1" -ForegroundColor White
    Write-Host "  3. Run: go build -o build\bin\hydra.exe .\cmd\hydra" -ForegroundColor White
}
