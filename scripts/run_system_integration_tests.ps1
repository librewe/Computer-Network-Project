$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Get-PythonCommand {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source) {
        return $pythonCommand.Source
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand -and $pyCommand.Source) {
        return $pyCommand.Source
    }

    return $null
}

Write-Host "============================================================"
Write-Host "Part 3: System Integration and Frontend Tests"
Write-Host "============================================================"
Write-Host ""

try {
    $pythonCmd = Get-PythonCommand
    if (-not $pythonCmd) {
        throw "No Python launcher found"
    }
    $pythonVersion = & $pythonCmd --version
    Write-Host $pythonVersion
} catch {
    Write-Host "Python is not available. Please install it or add it to PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[1/2] Running integration test script..."
$env:PYTHONDONTWRITEBYTECODE = "1"
& $pythonCmd scripts\test_system_integration.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Tests failed. Check the error output above." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[2/2] Tests finished."
Write-Host "If you want to continue with a demo run:"
Write-Host "  cd src-trained"
Write-Host "  streamlit run dashboard.py --server.port 8501"
Write-Host ""
Write-Host "============================================================"
Write-Host "All test steps finished"
Write-Host "============================================================"
