# Build the Windows binary.
#
#   .\packaging\build.ps1
#
# Output:
#   dist\AIOLocal\AIOLocal.exe
#   dist\AIOLocal-0.1.0-win64.zip
#
# The build is a PyInstaller "onedir" bundle so startup stays fast and the web
# assets are visible for auditing. No model weights are bundled.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing .venv. Create it with: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}

Write-Host "Installing build tools..." -ForegroundColor Cyan
& $python -m pip install --quiet --upgrade "pyinstaller>=6.11,<7"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "Running tests before packaging..." -ForegroundColor Cyan
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; refusing to build" }

Write-Host "Cleaning old build output..." -ForegroundColor Cyan
foreach ($path in @("build", "dist")) {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm --clean "packaging\aiolocal.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$exe = Join-Path $root "dist\AIOLocal\AIOLocal.exe"
if (-not (Test-Path $exe)) { throw "Build finished but $exe is missing" }

Write-Host "Adding licence and notices..." -ForegroundColor Cyan
foreach ($doc in @("README.md", "NOTICE", "LICENSE", "SECURITY.md")) {
    $source = Join-Path $root $doc
    if (Test-Path $source) {
        Copy-Item -Force $source (Join-Path $root "dist\AIOLocal\$doc")
    }
}

$zip = Join-Path $root "dist\AIOLocal-0.1.0-win64.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Write-Host "Creating zip..." -ForegroundColor Cyan
Compress-Archive -Path (Join-Path $root "dist\AIOLocal\*") -DestinationPath $zip

$size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host ""
Write-Host "Built: $exe ($size MB)" -ForegroundColor Green
Write-Host "Zip:   $zip" -ForegroundColor Green
Write-Host ""
Write-Host "Run it with:  .\dist\AIOLocal\AIOLocal.exe"
Write-Host "First launch writes to %LOCALAPPDATA%\AIOLocal and opens your browser."
