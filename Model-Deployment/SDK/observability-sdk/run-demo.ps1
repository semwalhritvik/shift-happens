$venvPath = Join-Path $PSScriptRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-Not (Test-Path $pythonPath)) {
    Write-Error "Virtual environment not found. Run .\setup.ps1 first."
    exit 1
}

if (-Not $env:GOOGLE_APPLICATION_CREDENTIALS) {
    Write-Error "GOOGLE_APPLICATION_CREDENTIALS is not set. Set it before running and then rerun this task."
    Write-Host "Example command:"
    Write-Host "  $env:GOOGLE_APPLICATION_CREDENTIALS = 'C:\path\to\service-account.json'"
    exit 1
}

Write-Host "Running demo with virtual environment Python..."
& $pythonPath "$(Join-Path $PSScriptRoot 'examples\demo.py')"
