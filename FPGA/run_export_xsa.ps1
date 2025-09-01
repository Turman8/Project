param(
  [string]$SettingsPath = "D:\Xilinx\Vivado\2019.2\settings64.bat"
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$script = Join-Path $here 'export_xsa.tcl'
if (-not (Test-Path $script)) { throw "Missing $script" }

$cmd = "call `"$SettingsPath`" && vivado -mode batch -source `"$script`""
Write-Host "Running: $cmd"
cmd.exe /c $cmd
if ($LASTEXITCODE -ne 0) { throw "Vivado export failed: exit $LASTEXITCODE" }
