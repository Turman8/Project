@echo off
REM 项目自动清理脚本 - Windows版本
REM 防止临时文件重新出现

echo 开始清理项目临时文件...

REM 切换到项目目录
cd /d "%~dp0"

REM 清理FPGA目录中的临时脚本
if exist "FPGA\*.ps1" del /f /q "FPGA\*.ps1" >nul 2>&1
if exist "FPGA\create_*.tcl" del /f /q "FPGA\create_*.tcl" >nul 2>&1
if exist "FPGA\*deploy*.tcl" del /f /q "FPGA\*deploy*.tcl" >nul 2>&1
if exist "FPGA\hls_build*.tcl" del /f /q "FPGA\hls_build*.tcl" >nul 2>&1
if exist "FPGA\*test*.tcl" del /f /q "FPGA\*test*.tcl" >nul 2>&1
if exist "FPGA\*synthesize*.tcl" del /f /q "FPGA\*synthesize*.tcl" >nul 2>&1
if exist "FPGA\*.txt" del /f /q "FPGA\*.txt" >nul 2>&1

REM 清理根目录临时文件
if exist "Untitled-*" del /f /q "Untitled-*" >nul 2>&1
if exist "test.py" del /f /q "test.py" >nul 2>&1
if exist "*.log" del /f /q "*.log" >nul 2>&1
if exist "*.jou" del /f /q "*.jou" >nul 2>&1

REM 清理Python缓存
if exist "__pycache__" rmdir /s /q "__pycache__" >nul 2>&1
if exist "*.pyc" del /f /q "*.pyc" >nul 2>&1
if exist "*.pyo" del /f /q "*.pyo" >nul 2>&1

REM 清理清理报告文档
if exist "CLEANUP_*.md" del /f /q "CLEANUP_*.md" >nul 2>&1

echo 清理完成！
