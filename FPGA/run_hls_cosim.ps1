param(
  [string]$SettingsPath,
  [string]$GccDir
)

# 自动配置 Xilinx 环境并运行 HLS：csim -> csynth -> cosim
$ErrorActionPreference = 'Stop'

# 组装候选 settings64.bat 路径（不做递归扫描，避免卡顿）
$settingsCandidates = @()
if ($SettingsPath) { $settingsCandidates += $SettingsPath }
$settingsCandidates += @(
  'C:\Xilinx\Vivado\2019.2\settings64.bat',
  'D:\Xilinx\Vivado\2019.2\settings64.bat'
)

$settings = $settingsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $settings) {
  throw "settings64.bat not found. Please pass -SettingsPath 'C:\\Xilinx\\Vivado\\2019.2\\settings64.bat' or adjust the path."
}

Write-Host ("[ENV] Using Xilinx settings: {0}" -f $settings)
Write-Host "[ENV] HLS tool: vivado_hls (targeting 2019.2)"

# 切换到脚本所在目录（FPGA）以保证相对路径正确
Set-Location -Path $PSScriptRoot

# 基础输入检查
$req = @('hls_source/classifier.cpp','hls_source/weights.cpp','testbench/test.cpp','build.tcl')
$missing = @()
foreach ($f in $req) { if (-not (Test-Path $f)) { $missing += $f } }
if ($missing.Count -gt 0) { throw ("Missing required files: {0}" -f ($missing -join ', ')) }

# 预置 GCC (MinGW) 路径，优先使用用户指定，其次常见安装目录
$gccCandidates = @()
if ($GccDir) { $gccCandidates += $GccDir }
$gccCandidates += @(
  'D:\Xilinx\Vivado\2019.2\mingw64\bin',
  'C:\Xilinx\Vivado\2019.2\mingw64\bin'
)
$gccPath = $gccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($gccPath) {
  Write-Host ("[ENV] Using GCC from: {0}" -f $gccPath)
  $env:PATH = "$gccPath;" + $env:PATH
} else {
  Write-Host "[WARN] GCC path not found in common locations; will rely on system PATH."
}

# 诊断：验证 gcc 可用性（在当前会话环境下）
Write-Host "[CHECK] where gcc && gcc --version"
& cmd.exe /c "where gcc && gcc --version"
if ($LASTEXITCODE -ne 0) {
  throw "GCC not available in PATH. Please install MinGW that ships with Vivado or pass -GccDir."
}

# 运行 HLS 流程（单次 cmd 调用：call settings && vivado_hls -f build.tcl）
Write-Host "[RUN] Running build.tcl (csim -> csynth -> cosim -> export) via cmd"
$cmd = "call `"$settings`" && vivado_hls -f build.tcl"
& cmd.exe /c $cmd
if ($LASTEXITCODE -ne 0) {
  throw ("HLS flow failed (exitcode: {0})" -f $LASTEXITCODE)
}

# 额外健壮性：即使 cmd 返回 0，也检查 HLS 日志是否有 ERROR 关键字
$hlsLog = Join-Path -Path (Get-Location) -ChildPath 'vivado_hls.log'
if (Test-Path $hlsLog) {
  $err = Select-String -Path $hlsLog -Pattern '^ERROR: ' -SimpleMatch -Quiet
  if ($err) {
    Write-Host "[ERR] vivado_hls.log contains ERROR lines. Tail follows:" -ForegroundColor Red
    Get-Content $hlsLog -Tail 50 | ForEach-Object { Write-Host $_ }
    throw "HLS flow encountered errors. See vivado_hls.log above."
  }
}

Write-Host "[OK] HLS flow completed. Artifacts (if exported): ecg_classifier_project/solution1/impl/ip/"
