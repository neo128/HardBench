Param(
  [string]$Venv = ""
)

if ($Venv -ne "") {
  python -m venv $Venv
  $activate = Join-Path $Venv "Scripts\Activate.ps1"
  if (Test-Path $activate) { . $activate }
}

python -m pip install --upgrade pip
pip install .

Write-Host "Installed robot-diagnostic-suite. Try: robot-diag --help"
