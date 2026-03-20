# test_batch.ps1
# Batch test Restormer model weights
# Usage: .\LLIE\test_batch.ps1 -WeightsDir "path/to/weights" -Opt "path/to/opt.yml"

param (
    [Parameter(Mandatory = $false)]
    [string]$WeightsDir = ".\experiments\LowLight_Restormer_128_2_60k_HTA",

    [Parameter(Mandatory = $false)]
    [string]$Opt = "LLIE/Options/LowLight_Restormer.yml",

    [Parameter(Mandatory = $false)]
    [string]$InputDir = ".\datasets\LOL-v2\Real_captured\Test\Low\",

    [Parameter(Mandatory = $false)]
    [string]$BaseResultDir = ""
)

# If BaseResultDir is not specified, use the parent folder name of WeightsDir
if ([string]::IsNullOrWhiteSpace($BaseResultDir)) {
    $FolderName = Split-Path $WeightsDir -Leaf
    $BaseResultDir = Join-Path ".\results" $FolderName
}

# Check if running from root
if (-not (Test-Path "LLIE/test.py")) {
    Write-Host "Error: LLIE/test.py not found. Please run this script from the project root directory." -ForegroundColor Red
    Write-Host "Current Directory: $(Get-Location)"
    exit 1
}

# Check if weights directory exists
if (-not (Test-Path $WeightsDir)) {
    Write-Host "Error: Weights directory not found at $WeightsDir" -ForegroundColor Red
    exit 1
}

# Find all .pth files in the specified directory
$WeightFiles = Get-ChildItem -Path $WeightsDir -Filter "*.pth" | Sort-Object Name

if ($WeightFiles.Count -eq 0) {
    Write-Host "No .pth files found in $WeightsDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($WeightFiles.Count) weight files to test." -ForegroundColor Green

foreach ($file in $WeightFiles) {
    # Create a unique result directory for each weight
    $WeightName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $CurrentResultDir = Join-Path $BaseResultDir $WeightName
    
    # Create the result directory if it doesn't exist
    if (-not (Test-Path $CurrentResultDir)) {
        New-Item -ItemType Directory -Path $CurrentResultDir -Force | Out-Null
    }

    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host ">>> Starting test for weight: $($file.Name)" -ForegroundColor Green
    Write-Host ">>> Output directory: $CurrentResultDir" -ForegroundColor Yellow
    Write-Host ("=" * 80) -ForegroundColor Cyan

    # Execute test.py
    # Using python directly. Ensure your environment is activated.
    python LLIE/test.py --input_dir "$InputDir" --result_dir "$CurrentResultDir" --weights "$($file.FullName)" --opt "$Opt"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error occurred while testing $($file.Name)" -ForegroundColor Red
    }
}

Write-Host "`nBatch testing completed!" -ForegroundColor Green
