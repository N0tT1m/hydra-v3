# Run a Hydra worker (Windows PowerShell)

param(
    [Parameter(Mandatory=$true)]
    [Alias("n", "node-id")]
    [string]$NodeId,

    [Alias("c", "coordinator")]
    [string]$Coordinator = "tcp://localhost:5555",

    [Alias("d", "device")]
    [string]$Device = "auto",

    [Alias("dtype")]
    [ValidateSet("float16", "bfloat16", "float32", "int8", "int4", "fp8")]
    [string]$Dtype = "bfloat16",

    [Alias("p", "pipeline-port")]
    [int]$PipelinePort = 6000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location "$ProjectRoot\worker"

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & .\venv\Scripts\Activate.ps1
} elseif (Test-Path "..\venv\Scripts\Activate.ps1") {
    & ..\venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found. Run setup script first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Hydra Worker..." -ForegroundColor Green
Write-Host "  Node ID: $NodeId" -ForegroundColor Cyan
Write-Host "  Coordinator: $Coordinator" -ForegroundColor Cyan
Write-Host "  Device: $Device" -ForegroundColor Cyan
Write-Host "  Dtype: $Dtype" -ForegroundColor Cyan
Write-Host "  Pipeline Port: $PipelinePort" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

hydra-worker start `
    --node-id $NodeId `
    --coordinator $Coordinator `
    --device $Device `
    --dtype $Dtype `
    --pipeline-port $PipelinePort
