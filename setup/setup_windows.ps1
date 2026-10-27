# One-time setup on a Windows training laptop. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
# Needs Python 3.10+ (python.org installer, "Add to PATH" ticked). No admin rights needed.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $exe $($arguments -join ' ')" }
}

if (Get-Command py -ErrorAction SilentlyContinue) { $pyExe = 'py'; $pyArgs = @('-3') }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $pyExe = 'python'; $pyArgs = @() }
else { throw 'Python not found. Install Python 3.10 or newer from python.org (tick "Add python.exe to PATH").' }

Run $pyExe ($pyArgs + @('-c', "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ needed'"))
if (-not (Test-Path .venv)) { Run $pyExe ($pyArgs + @('-m', 'venv', '.venv')) }
$vpy = '.venv\Scripts\python.exe'
Run $vpy @('-m', 'pip', 'install', '--upgrade', 'pip')

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host 'NVIDIA GPU found: installing PyTorch with CUDA support (~3 GB download)...'
    Run $vpy @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128')
} else {
    Write-Host 'No NVIDIA GPU found: installing the CPU version of PyTorch (training will be slow).'
    Run $vpy @('-m', 'pip', 'install', 'torch', 'torchvision')
}
Run $vpy @('-m', 'pip', 'install', '-r', 'requirements.txt')
& $vpy src\check_setup.py
Write-Host "`nNext: .venv\Scripts\activate   then   python src\benchmark.py"
