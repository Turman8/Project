param(
  [string]$SettingsPath = "D:\Xilinx\Vivado\2019.2\settings64.bat",
  [switch]$Background
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$script = Join-Path $here 'build_bitstream.tcl'
if (-not (Test-Path $script)) { throw "Missing $script" }

$cmd = "call `"$SettingsPath`" && vivado -mode batch -source `"$script`""
Write-Host "Running: $cmd"
if ($Background) {
  Start-Process cmd.exe -ArgumentList "/c $cmd" -WindowStyle Minimized
  Write-Host "Started in background. Check vivado.log under project run dir."
} else {
  cmd.exe /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Vivado build failed: exit $LASTEXITCODE" }
}
