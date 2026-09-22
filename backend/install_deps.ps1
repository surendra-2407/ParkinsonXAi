# ============================================================
# install_deps.ps1 — Install all backend dependencies
# Fixes Windows MAX_PATH issue by using a short temp directory.
#
# Usage (from backend/ folder):
#   .\install_deps.ps1
# ============================================================

# Use short temp path to avoid Windows 260-char path limit
$env:TEMP = "C:\tmp"
$env:TMP  = "C:\tmp"
New-Item -ItemType Directory -Force -Path "C:\tmp" | Out-Null

Write-Host "Installing ParkinsonXAI backend dependencies..." -ForegroundColor Cyan
Write-Host "Using TEMP=C:\tmp to avoid Windows path-length errors." -ForegroundColor Yellow

# Upgrade pip first
python.exe -m pip install --upgrade pip --quiet

# Install all packages with --prefer-binary (avoids C compiler requirement)
pip install -r requirements_backend.txt --prefer-binary

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅  All dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n❌  Some packages failed. Check the output above." -ForegroundColor Red
    exit 1
}
