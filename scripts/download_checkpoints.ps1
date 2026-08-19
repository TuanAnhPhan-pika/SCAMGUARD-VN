$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$target = Join-Path $root "artifacts\checkpoints"
New-Item -ItemType Directory -Force $target | Out-Null
$base = "https://github.com/TuanAnhPhan-pika/SCAMGUARD-VN/releases/download/model-v1"
$files = @("checkpoint_best.pt", "risk_e2e_best.pt")
foreach ($name in $files) {
    $out = Join-Path $target $name
    if (-not (Test-Path $out)) {
        Write-Host "Downloading $name ..."
        Invoke-WebRequest "$base/$name" -OutFile $out
    }
}
& (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "scripts\verify_release.py") --checkpoints

