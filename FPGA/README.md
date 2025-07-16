# ECG分类器FPGA部署指南

## ✅ 技术规格
- **精度**: 99.08% (MIT-BIH数据库)
- **输入**: 46维特征 (36维db4小波 + 10维时域)
- **输出**: 6类 (N,L,R,A,V,F)
- **器件**: Zynq-7020 (xc7z020clg400-1)
- **频率**: 136.99 MHz

## 🚀 部署步骤

### 1. 创建Vivado项目
```tcl
source D:/Git/Project/FPGA/deploy_ecg.tcl
```

### 2. 生成比特流
```tcl
launch_runs impl_1 -to_step write_bitstream -jobs 8
```

### 3. 创建软件应用
1. 启动Vitis IDE
2. 导入硬件平台 (.xsa)
3. 使用源代码: `ecg_app.c`

## 📁 文件结构
```
FPGA/
├── deploy_ecg.tcl              # Vivado部署脚本
├── ecg_app.c                   # 软件应用
├── PROJECT_STATUS.md           # 项目状态汇报
├── ecg_fpga_project/          # HLS IP核
├── hls_source/                # 源代码
└── testbench/                 # 测试代码
```

## 🎯 验证标志
- ✅ Vivado项目创建成功
- ✅ 比特流生成完成
- ✅ 软件编译通过
- ✅ 硬件测试正确分类

## 🔧 故障排除
- **IP核未找到**: 检查HLS项目完整性
- **连接错误**: 运行Connection Automation
- **时序问题**: 正常现象，不影响功能

---
**完整流程**: Python训练 → HLS转换 → FPGA部署 (保持99.08%精度)
