#!/usr/bin/env python3
"""
GitHub上传前的项目清理脚本
清理Vivado、Vitis生成文件和多余文档，保持项目核心代码
"""

import os
import shutil
import glob
from pathlib import Path

def remove_path(path):
    """安全删除文件或目录"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"✅ 删除文件: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✅ 删除目录: {path}")
        else:
            print(f"⚠️ 路径不存在: {path}")
    except Exception as e:
        print(f"❌ 删除失败 {path}: {e}")

def cleanup_vivado_vitis():
    """清理Vivado、Vitis自动生成的文件"""
    print("🧹 开始清理Vivado/Vitis生成文件...")
    
    # Vivado/Vitis生成文件模式
    patterns_to_remove = [
        # HLS生成文件
        "FPGA/hls_project/solution1/",
        "FPGA/hls_project/hls.app",
        
        # Vivado工程文件
        "**/*.xpr",
        "**/*.cache/",
        "**/*.runs/",
        "**/*.sim/",
        "**/*.srcs/",
        "**/*.gen/",
        "**/*.hw/",
        "**/*.ip_user_files/",
        
        # 日志和报告文件
        "**/*.log",
        "**/*.jou",
        "**/*.rpt",
        "**/*.dcp",
        "**/*.bit",
        
        # 临时文件
        "**/.Xil/",
        "**/vivado*.backup.*",
        "**/vitis_*.backup.*",
        "**/*.str",
        "**/*.wcfg",
    ]
    
    project_root = Path(".")
    for pattern in patterns_to_remove:
        for path in project_root.glob(pattern):
            remove_path(str(path))

def cleanup_documents():
    """清理多余的文档文件，只保留一个主要的README"""
    print("📚 开始清理文档文件...")
    
    # 要删除的文档文件
    docs_to_remove = [
        "DEVELOPMENT_WORKFLOW.md",
        "GIT_CONCEPT_EXPLAINED.md", 
        "PROJECT_SAFETY_STATUS.md",
        "FPGA/EMERGENCY_SOLUTION.md",
        "FPGA/ROOT_CAUSE_ANALYSIS.md",
        "PROJECT_REPORT.md",
        "safety_report_*.json",
        "project_config.json"
    ]
    
    for doc_pattern in docs_to_remove:
        for path in glob.glob(doc_pattern):
            remove_path(path)

def cleanup_scripts():
    """清理多余的脚本文件"""
    print("🔧 开始清理多余脚本...")
    
    scripts_to_remove = [
        "cleanup_large_files.ps1",
        "cleanup.bat", 
        "cleanup.sh",
        "dev_launcher.py",
        "project_safety_check.py"
    ]
    
    for script in scripts_to_remove:
        if os.path.exists(script):
            remove_path(script)

def create_main_readme():
    """创建主要的README文档"""
    print("� 创建README.md...")
    
    readme_content = """# ECG心电信号分类器 - FPGA硬件加速项目

## 🎯 项目概述

本项目实现了基于MIT-BIH数据库的心电信号分类系统，达到99.08%的分类准确率，并成功部署到Xilinx Zynq-7020 FPGA平台。

## 🏗️ 项目架构

```
Project/
├── main.py                # 主训练脚本 (Python)
├── export_weights.py      # 权重导出工具
├── data/                  # MIT-BIH心电数据集
├── outputs/              # 训练结果输出
└── FPGA/                 # FPGA实现
    ├── build.tcl         # 主构建脚本
    ├── hls_source/       # HLS C++源代码
    │   ├── classifier.cpp # 主分类器
    │   ├── classifier.h   # 头文件
    │   ├── weights.h      # 神经网络权重
    │   └── params.vh      # 参数定义
    └── testbench/        # 测试平台
        └── testbench.cpp # 测试台
```

## 🎨 技术特点

### 算法性能
- **分类准确率**: 99.08% (MIT-BIH数据库)
- **特征维度**: 46维 (36维db4小波 + 10维时域特征)  
- **分类类别**: 6类心电信号 (N,L,R,A,V,F)
- **网络架构**: 全连接神经网络 (46→256→128→64→6)

### FPGA实现
- **目标平台**: Xilinx Zynq-7020
- **开发工具**: Vivado 2024.1.2 + Vitis HLS
- **数据类型**: 16位定点数设计，零浮点依赖
- **接口协议**: AXI3兼容 (针对Zynq-7000 HP端口)

## 🚀 快速开始

### 环境要求
```bash
Python 3.8+
pandas, numpy, scikit-learn
Xilinx Vivado 2024.1.2
Vitis HLS 2024.1.2
```

### 训练模型
```bash
python main.py
python export_weights.py
```

### FPGA构建
```bash
cd FPGA
vivado -mode batch -source build.tcl
```

## 📊 性能指标

- **FPGA资源使用**: DSP优化，流水线设计
- **处理延迟**: 低延迟实时分类
- **功耗**: 低功耗硬件加速
- **兼容性**: Zynq-7000系列全兼容

## 🏆 技术创新

1. **AXI协议适配**: 解决HLS默认AXI4与Zynq-7000 AXI3兼容性问题
2. **定点化优化**: 手动定点化设计，提升硬件效率  
3. **特征工程**: db4小波+时域特征的混合特征方案
4. **实时处理**: 硬件加速的实时心电分类系统

## 📈 未来计划

- [ ] 支持更多心电数据库
- [ ] 优化FPGA资源使用
- [ ] 添加可视化界面
- [ ] 扩展到其他FPGA平台

## 📜 许可证

本项目采用 MIT 许可证

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---
*基于深度学习的心电信号分类与FPGA硬件加速实现*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ README.md 创建完成")

def update_gitignore():
    """更新.gitignore文件"""
    print("📄 更新.gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Vivado/Vitis生成文件
*.xpr
*.cache/
*.runs/
*.sim/
*.srcs/
*.gen/
*.hw/
*.ip_user_files/
*.log
*.jou
*.rpt
*.dcp
*.bit
*.str
*.wcfg
.Xil/
vivado*.backup.*
vitis*.backup.*

# HLS生成文件  
solution*/
csim/
*.aps
*.directive
*_data.json
hls.app

# 大数据文件
*.dat
*.atr
*.hea
*.xws
*.at_
*.at-

# 系统文件
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# 临时文件
*.tmp
*.temp
*.bak

# VS Code
.vscode/settings.json
.vscode/tasks.json
.vscode/launch.json
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore 更新完成")

def main():
    """主清理函数"""
    print("🚀 开始GitHub上传前的项目清理...")
    print("=" * 50)
    
    # 执行清理步骤
    cleanup_vivado_vitis()
    print()
    cleanup_documents() 
    print()
    cleanup_scripts()
    print()
    create_main_readme()
    print()
    update_gitignore()
    
    print("=" * 50)
    print("🎉 项目清理完成！现在可以安全上传到GitHub了")
    print("\n📋 保留的核心文件:")
    print("   ✅ main.py - 主训练脚本")
    print("   ✅ export_weights.py - 权重导出")
    print("   ✅ FPGA/hls_source/ - 核心HLS源代码") 
    print("   ✅ FPGA/testbench/ - 测试代码")
    print("   ✅ FPGA/build.tcl - 构建脚本")
    print("   ✅ data/ - 数据集")
    print("   ✅ outputs/ - 输出结果")
    print("   ✅ README.md - 项目文档")
    print("   ✅ .gitignore - Git忽略规则")

if __name__ == "__main__":
    main()
