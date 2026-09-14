# run.ps1 — Script de arranque del servidor Traffic Vision Analytics
# Uso: .\run.ps1
# Detener: Ctrl+C

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] No se encontró el entorno virtual en .venv\" -ForegroundColor Red
    Write-Host "Ejecuta primero: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Traffic Vision Analytics — YOLOv8 RTSP    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Dashboard → http://localhost:8000"           -ForegroundColor Green
Write-Host "  API stats → http://localhost:8000/api/stats" -ForegroundColor Green
Write-Host "  Detener   → Ctrl+C"                         -ForegroundColor Yellow
Write-Host ""

Set-Location $ProjectDir
& $VenvPython main.py
