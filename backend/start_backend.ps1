# Start ParkinsonXAI Backend
# Usage: .\start_backend.ps1
# Run this from the backend\ folder

# Fix Windows MAX_PATH issue for any runtime temp writes
$env:TEMP = "C:\tmp"
$env:TMP  = "C:\tmp"
New-Item -ItemType Directory -Force -Path "C:\tmp" | Out-Null

$envFile = ".\.env"
if (-not (Test-Path $envFile)) {
    Write-Error "Missing .env file. Copy .env.example and add your MONGODB_URL."
    exit 1
}

$mongoUrl = (Get-Content $envFile | Where-Object { $_ -match "^MONGODB_URL=" }) -replace "^MONGODB_URL=", ""
if ($mongoUrl -eq "PASTE_YOUR_MONGODB_ATLAS_URL_HERE" -or $mongoUrl -eq "") {
    Write-Warning "MONGODB_URL not set in .env — running without database persistence."
}

Write-Host "Starting ParkinsonXAI Backend..." -ForegroundColor Cyan
Write-Host "API docs available at: http://localhost:8000/docs" -ForegroundColor Green
uvicorn main:app --reload --host 0.0.0.0 --port 8000
