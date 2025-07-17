#!/bin/bash
# 项目自动清理脚本
# 防止临时文件重新出现

echo "开始清理项目临时文件..."

# 清理FPGA目录中的临时脚本
find ./FPGA -name "*.ps1" -delete 2>/dev/null
find ./FPGA -name "create_*.tcl" -delete 2>/dev/null
find ./FPGA -name "*deploy*.tcl" -delete 2>/dev/null
find ./FPGA -name "hls_build*.tcl" -delete 2>/dev/null
find ./FPGA -name "*test*.tcl" -delete 2>/dev/null
find ./FPGA -name "*synthesize*.tcl" -delete 2>/dev/null
find ./FPGA -name "*.txt" -delete 2>/dev/null
find ./FPGA -name "*分析*.md" -delete 2>/dev/null

# 清理根目录临时文件
find . -maxdepth 1 -name "Untitled-*" -delete 2>/dev/null
find . -maxdepth 1 -name "test.py" -delete 2>/dev/null
find . -maxdepth 1 -name "*.log" -delete 2>/dev/null
find . -maxdepth 1 -name "*.jou" -delete 2>/dev/null

# 清理Python缓存
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# 清理清理报告文档
find . -maxdepth 1 -name "CLEANUP_*.md" -delete 2>/dev/null

echo "清理完成！"
