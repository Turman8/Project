param(
  [switch]$Force = $false
)

$ErrorActionPreference = 'Stop'
$proj = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent

# 保护清单：这些产物和目录一律保留
$keep = @(
  'FPGA/vivado_ecg_proj/export/design_1_wrapper.xsa',
  'FPGA/vivado_ecg_proj/ecg_proj.runs/impl_1',
  'FPGA/ecg_classifier_project/solution1/impl/ip'
)

# 清理目标（仅生成物/缓存/日志，不含源码与配置）
$targets = @(
  'FPGA/vivado_ecg_proj/.Xil',
  'FPGA/vivado_ecg_proj/.cache',
  'FPGA/vivado_ecg_proj/ecg_proj.cache',
  'FPGA/vivado_ecg_proj/ecg_proj.hw',
  'FPGA/vivado_ecg_proj/ecg_proj.ip_user_files',
  'FPGA/vivado_ecg_proj/ecg_proj.runs/synth_1',
  'FPGA/vivado_ecg_proj/ecg_proj.runs/design_1_*_synth_1',
  'FPGA/vivado_ecg_proj/reports',
  'FPGA/vivado_ecg_proj/vivado.jou',
  'FPGA/vivado_ecg_proj/vivado.log',
  'FPGA/vivado_ecg_proj/*.jou',
  'FPGA/vivado_ecg_proj/*.log',
  'FPGA/ecg_classifier_project/solution1/csim',
  'FPGA/ecg_classifier_project/solution1/sim',
  'FPGA/ecg_classifier_project/solution1/syn',
  'FPGA/ecg_classifier_project/solution1/.autopilot',
  'FPGA/ecg_classifier_project/solution1/.ipcache',
  'FPGA/ecg_classifier_project/solution1/impl/.Xil',
  'FPGA/*.jou',
  'FPGA/*.log',
  'FPGA/vivado_hls.log',
  'outputs/fpga_deployment*',
  '__pycache__',
  '*/__pycache__'
)

Write-Host "[CLEAN] Preview of generated files/folders to delete:" -ForegroundColor Cyan
$toDelete = @()
foreach ($t in $targets) {
  $paths = Get-ChildItem -LiteralPath (Join-Path $proj '.') -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    $_.FullName -like (Join-Path $proj $t).Replace('/', '\')
  }
  foreach ($p in $paths) {
    # 保护检查
    $protected = $false
    foreach ($k in $keep) {
      $kfull = (Join-Path $proj $k).Replace('/', '\')
      if ($p.FullName -like "$kfull*") { $protected = $true; break }
    }
    # 跳过 Vivado 状态文件（.rst, __synthesis_is_complete__ 等）
    if ($p.Name -match '\.(rst|rpt)$' -or $p.Name -match '^__.*__$') { $protected = $true }
    if (-not $protected) { $toDelete += $p.FullName }
  }
}
$toDelete = $toDelete | Sort-Object -Unique
if ($toDelete.Count -eq 0) {
  Write-Host "  (nothing to delete)"
} else {
  $toDelete | ForEach-Object { Write-Host "  - $_" }
}

Write-Host "`n[KEEP] Always keep:" -ForegroundColor Green
$keep | ForEach-Object { Write-Host "  + $_" }

if (-not $Force) {
  Write-Host "`n[INFO] Preview only. Run with -Force to delete."
  exit 0
}

Write-Host "`n[DELETE] Deleting..." -ForegroundColor Yellow
foreach ($path in $toDelete) {
  try {
    if (Test-Path $path) {
      if ((Get-Item $path).PSIsContainer) {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
      } else {
        Remove-Item -LiteralPath $path -Force -ErrorAction Stop
      }
      Write-Host "  Deleted: $path"
    }
  } catch {
    Write-Warning "  Failed: $path -> $($_.Exception.Message)"
  }
}
Write-Host "[DONE] Cleanup completed."
