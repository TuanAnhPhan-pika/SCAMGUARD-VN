$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if (-not (Test-Path ".venv")) { py -3.12 -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install torch
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Write-Host "Environment ready. Next: .\scripts\download_checkpoints.ps1"

