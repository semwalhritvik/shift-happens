Write-Host "Setting up the observability SDK environment..."

$venvPath = Join-Path $PSScriptRoot ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Error "Virtual environment activation script not found. Ensure Python is installed and accessible."
    exit 1
}

Write-Host "Installing Python dependencies..."
& "$venvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$venvPath\Scripts\python.exe" -m pip install -r "$(Join-Path $PSScriptRoot 'requirements.txt')"
& "$venvPath\Scripts\python.exe" -m pip install -e "$PSScriptRoot"

Write-Host "Setup complete. To use the environment, run:"
Write-Host "    .\ .venv\Scripts\Activate.ps1"
Write-Host "Then run:"
Write-Host "    python examples/demo.py"
