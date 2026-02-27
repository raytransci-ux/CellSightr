# CellSightr - Hemocytometer Cell Counter
# Windows launch script (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host " ============================================"
Write-Host "   CellSightr - Hemocytometer Cell Counter"
Write-Host " ============================================"
Write-Host ""

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ---------------------------------------------------------------------------
# 1. Find a working Python 3.10+
# ---------------------------------------------------------------------------
$pythonExe = $null

# Try the py launcher first — it resolves to the actual python.exe path
try {
    $candidate = & py -3 -c "import sys; print(sys.executable)" 2>$null
    if ($candidate -and (Test-Path $candidate) -and ($candidate -notmatch "WindowsApps")) {
        $ok = & $candidate -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate }
    }
} catch {}

# Search PATH entries, skipping the Microsoft Store stub
if (-not $pythonExe) {
    $candidates = Get-Command python -All -ErrorAction SilentlyContinue |
                  Where-Object { $_.Source -notmatch "WindowsApps" }
    foreach ($c in $candidates) {
        try {
            $ok = & $c.Source -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonExe = $c.Source; break }
        } catch {}
    }
}

# Check common install directories
if (-not $pythonExe) {
    $dirs = 313,312,311,310 | ForEach-Object {
        "$env:LOCALAPPDATA\Programs\Python\Python$_\python.exe",
        "C:\Python$_\python.exe",
        "C:\Program Files\Python$_\python.exe"
    }
    foreach ($p in $dirs) {
        if (Test-Path $p) {
            try {
                & $p -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>$null
                if ($LASTEXITCODE -eq 0) { $pythonExe = $p; break }
            } catch {}
        }
    }
}

if (-not $pythonExe) {
    Write-Host "ERROR: Python 3.10+ not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "  1. Download Python from https://python.org/downloads"
    Write-Host "  2. During install, tick 'Add Python to PATH'"
    Write-Host "  3. Re-run this script"
    Write-Host ""
    exit 1
}

Write-Host "Python: $pythonExe"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Using existing virtual environment."
    & $venvActivate
} else {
    Write-Host "Creating virtual environment..."
    & $pythonExe -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    & $venvActivate
    Write-Host "Installing dependencies (first run - this takes a minute)..."
    pip install -r (Join-Path $root "webapp\backend\requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install failed. Check your internet connection." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# ---------------------------------------------------------------------------
# 3. Check model weights
# ---------------------------------------------------------------------------
$weights = Join-Path $root "checkpoints\yolo\nano\weights\best.pt"
if (-not (Test-Path $weights)) {
    Write-Host "ERROR: Model weights not found at: $weights" -ForegroundColor Red
    Write-Host "Copy the checkpoints\ folder from the development machine and re-run."
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Launch
# ---------------------------------------------------------------------------
$env:PYTHONPATH = Join-Path $root "webapp\backend"

Write-Host ""
Write-Host "Starting CellSightr on http://localhost:8000"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir (Join-Path $root "webapp\backend")
