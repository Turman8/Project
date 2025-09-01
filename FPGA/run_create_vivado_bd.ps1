param(
  [string]$SettingsPath = "D:\Xilinx\Vivado\2019.2\settings64.bat"
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path $SettingsPath)) {
  Write-Error "settings64.bat not found: $SettingsPath"
}

$script = Join-Path $here 'create_vivado_bd.tcl'
if (-not (Test-Path $script)) { Write-Error "Missing $script" }

$cmd = "call `"$SettingsPath`" && vivado -mode batch -source `"$script`""
Write-Host "Running: $cmd"
cmd.exe /c $cmd

if ($LASTEXITCODE -ne 0) { throw "Vivado batch failed: exit $LASTEXITCODE" }

$proj = Join-Path $here 'vivado_ecg_proj'
if (-not (Test-Path $proj)) { throw "Project folder not created: $proj" }

Write-Host "Done. Project at $proj"
