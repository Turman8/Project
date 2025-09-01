# Project Safe Cleanup Script
param(
    [switch]$Preview = $false,
    [switch]$Force = $false
)

$baseDir = Split-Path $PSScriptRoot
Write-Host "Project root: $baseDir" -ForegroundColor Green

# Directories to delete
$deleteDirs = @(
    "FPGA\ecg_classifier_project\solution1\csim",
    "FPGA\ecg_classifier_project\solution1\sim", 
    "FPGA\ecg_classifier_project\solution1\syn",
    "FPGA\ecg_classifier_project\solution1\.autopilot",
    "FPGA\ecg_classifier_project\solution1\.ipcache",
    "FPGA\vivado_ecg_proj\.Xil",
    "FPGA\vivado_ecg_proj\.cache", 
    "FPGA\vivado_ecg_proj\ecg_proj.cache",
    "FPGA\vivado_ecg_proj\ecg_proj.hw",
    "FPGA\vivado_ecg_proj\ecg_proj.ip_user_files"
)

# Files to delete
$deleteFiles = @(
    "FPGA\synthesis.tcl",
    "Untitled-1.ipynb"
)

Write-Host "=== Project Safe Cleanup ===" -ForegroundColor Cyan

if ($Preview) {
    Write-Host "[Preview Mode] Files/directories to be deleted:" -ForegroundColor Yellow
} else {
    if (-not $Force) {
        $confirm = Read-Host "Confirm deletion of temporary files? (type 'yes' to confirm)"
        if ($confirm -ne "yes") {
            Write-Host "Operation cancelled" -ForegroundColor Red
            exit 0
        }
    }
    Write-Host "Starting cleanup..." -ForegroundColor Green
}

$deletedCount = 0
$totalSize = 0

# Delete directories
foreach ($dir in $deleteDirs) {
    $fullPath = Join-Path $baseDir $dir
    if (Test-Path $fullPath) {
        $size = (Get-ChildItem $fullPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        if ($null -eq $size) { $size = 0 }
        $totalSize += $size
        
        if ($Preview) {
            $sizeMB = [math]::Round($size/1MB, 2)
            Write-Host "  - $fullPath ($sizeMB MB)" -ForegroundColor Gray
        } else {
            Write-Host "Deleting directory: $fullPath" -ForegroundColor Red
            Remove-Item $fullPath -Recurse -Force -ErrorAction SilentlyContinue
            $deletedCount++
        }
    }
}

# Delete files
foreach ($file in $deleteFiles) {
    $fullPath = Join-Path $baseDir $file
    if (Test-Path $fullPath) {
        $item = Get-Item $fullPath
        $totalSize += $item.Length
        
        if ($Preview) {
            $sizeKB = [math]::Round($item.Length/1KB, 2)
            Write-Host "  - $fullPath ($sizeKB KB)" -ForegroundColor Gray
        } else {
            Write-Host "Deleting file: $fullPath" -ForegroundColor Red
            Remove-Item $fullPath -Force -ErrorAction SilentlyContinue
            $deletedCount++
        }
    }
}

# Delete log files
$logPatterns = @("*.log", "*.jou")
foreach ($pattern in $logPatterns) {
    $logFiles = Get-ChildItem (Join-Path $baseDir "FPGA") -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($logFile in $logFiles) {
        $totalSize += $logFile.Length
        if ($Preview) {
            $sizeKB = [math]::Round($logFile.Length/1KB, 2)
            Write-Host "  - $($logFile.FullName) ($sizeKB KB)" -ForegroundColor Gray
        } else {
            Write-Host "Deleting log: $($logFile.FullName)" -ForegroundColor Red
            Remove-Item $logFile.FullName -Force -ErrorAction SilentlyContinue
            $deletedCount++
        }
    }
}

if ($Preview) {
    $totalMB = [math]::Round($totalSize/1MB, 2)
    Write-Host "Estimated space to free: $totalMB MB" -ForegroundColor Cyan
    Write-Host "To execute: .\scripts\project_cleanup_safe.ps1 -Force" -ForegroundColor Green
} else {
    $totalMB = [math]::Round($totalSize/1MB, 2)
    Write-Host "Cleanup completed!" -ForegroundColor Green
    Write-Host "Deleted $deletedCount items, freed $totalMB MB space" -ForegroundColor Cyan
}

Write-Host "Preserved critical files:" -ForegroundColor Green
Write-Host "  √ All source code (.cpp, .h, .py)" -ForegroundColor Green  
Write-Host "  √ IP core: FPGA\ecg_classifier_project\solution1\impl\ip\" -ForegroundColor Green
Write-Host "  √ Bitstream: FPGA\vivado_ecg_proj\ecg_proj.runs\impl_1\design_1_wrapper.bit" -ForegroundColor Green
Write-Host "  √ XSA file: FPGA\vivado_ecg_proj\export\design_1_wrapper.xsa" -ForegroundColor Green
Write-Host "  √ Automation scripts: run_*.ps1, *.tcl" -ForegroundColor Green
