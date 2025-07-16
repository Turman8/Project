# ECG心电图分类器FPGA实现

## 项目概述

本项目实现了基于MIT-BIH数据库训练的ECG心电图分类器的FPGA硬件加速方案。

### 模型性能指标
- **训练准确率**: 99.08%
- **训练样本**: 21,712个真实MIT-BIH心跳
- **特征维度**: 46维（36维db4小波特征 + 10维时域特征）
- **分类类别**: 7类（N,L,R,A,V,F,P）

### 硬件架构
- **网络结构**: 全连接神经网络 [46 → 128 → 64 → 32 → 7]
- **数据类型**: 16位定点数 (ap_fixed<16,8>)
- **接口类型**: AXI4-Lite控制 + AXI4内存访问

## 📁 文件结构

```
FPGA/
├── hls_source/                    # HLS源代码
│   ├── ecg_trained_classifier.cpp # 主分类器实现
│   └── ecg_params.vh              # 硬件参数定义
│
├── testbench/                     # 测试激励
│   └── tb_ecg_classifier.cpp      # C++测试激励
│
├── scripts/                       # 自动化脚本
│   └── run_hls.tcl               # HLS综合脚本
│
├── docs/                         # 文档
│   ├── deployment_guide.md       # 部署指南
│   └── quantization_params.json  # 量化参数
│
└── README.md                     # 本文件
```

## 🚀 使用方法

### 方法1: GUI操作（推荐新手）
1. 启动Vitis HLS:
   ```bash
   vitis_hls -classic
   ```

2. 在GUI中新建项目:
   - Project Name: `ecg_fpga_project`
   - Top Function: `ecg_classify_trained`
   - Add Files: `hls_source/ecg_trained_classifier.cpp`
   - Add TestBench: `testbench/tb_ecg_classifier.cpp`

3. 设置目标器件（如: `xc7z020clg400-1`）

4. 运行仿真和综合:
   - C Simulation → C Synthesis → Export RTL

### 方法2: 脚本自动化（推荐有经验用户）
```bash
# 在FPGA目录下启动HLS
cd FPGA
vitis_hls -classic

# 在HLS控制台执行
source scripts/run_hls.tcl
```

## 📊 预期资源消耗

基于综合结果预估：
- **DSP48E1**: ~276个
- **BRAM_18K**: ~8个  
- **LUT**: ~23,000个
- **FF**: ~12,000个
- **最大频率**: ~100MHz
- **功耗**: ~300mW

## 🔧 接口说明

### 输入接口
- `features[46]`: 46维特征向量（ap_fixed<16,8>）

### 输出接口  
- `probabilities[7]`: 7类概率输出（ap_fixed<16,8>）
- `predicted_class`: 预测类别ID（int, 0-6）

### 控制接口
- AXI4-Lite从接口：控制启动和状态
- AXI4主接口：内存访问

## 🎯 集成到Vivado

1. 运行HLS生成IP核
2. 在Vivado中添加生成的IP
3. 连接AXI接口到Zynq处理系统
4. 配置中断和DMA（可选）

## 📝 测试用例

测试激励包含两个典型用例：
1. **正常心拍（N类）**: 模拟健康心律特征
2. **室性心律（V类）**: 模拟心律失常特征

## 🔍 验证方法

1. **C仿真**: 验证算法正确性
2. **RTL协同仿真**: 验证硬件时序
3. **板级测试**: 使用真实ECG数据验证

## ⚠️ 注意事项

1. **权重精度**: 当前使用示例权重，实际部署需要加载训练好的量化权重
2. **时钟约束**: 建议工作频率100MHz，可根据目标器件调整
3. **内存带宽**: 特征输入需要保证足够的内存带宽

## 📞 技术支持

如有问题，请检查：
1. HLS综合报告中的资源消耗
2. C仿真输出的分类结果
3. 时序约束是否满足

---

*基于MIT-BIH数据库的ECG分类器FPGA实现 - 99.08%准确率*
