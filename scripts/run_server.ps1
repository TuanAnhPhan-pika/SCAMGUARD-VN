$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:PYTHONUTF8 = "1"
& ".\.venv\Scripts\python.exe" -m uvicorn server.api:app --host 127.0.0.1 --port 8765

