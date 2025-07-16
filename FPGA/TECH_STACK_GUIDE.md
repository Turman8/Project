# 项目技术栈与依赖关系

## 🛠️ 完整技术栈

### 算法层 (已完成)
```
Python生态系统:
├── NumPy: 数值计算
├── SciPy: 信号处理 (小波变换)
├── TensorFlow/Keras: 神经网络
├── Matplotlib: 数据可视化
└── wfdb: MIT-BIH数据读取

特征工程:
├── 小波变换: pywt.wavedec (db4, level=5)
├── 时域特征: 统计分析 (均值、方差等)
└── 归一化: MinMax缩放
```

### HLS层 (已完成)
```
Vitis HLS 2024.1:
├── 语言: C++ (IEEE 754标准)
├── 数据类型: ap_fixed<16,8> (16位宽，8位整数)
├── 优化指令: #pragma HLS (流水线、并行)
├── 接口: s_axilite (控制) + m_axi (数据)
└── 约束: 时钟周期10ns (100MHz)

生成文件:
├── IP核: xilinx_com_hls_ecg_classify_trained_1_0.zip
├── 驱动: C头文件和源文件
└── 文档: 综合报告和资源使用
```

### 系统层 (进行中)
```
Vivado 2024.1:
├── Block Design: 图形化系统集成
├── Zynq-7020: ARM Cortex-A9 + Artix-7 FPGA
├── 时钟管理: FCLK_CLK0 (100MHz)
├── 内存: DDR3 512MB (共享)
└── 接口: UART, GPIO, Ethernet

AXI互连架构:
├── GP0: ARM→PL控制 (32位地址)
├── HP0: PL→ARM数据 (64位数据)
└── 地址映射: 0x40000000 (控制), 0x00000000 (数据)
```

### 软件层 (待开发)
```
ARM应用:
├── 语言: C (GNU GCC)
├── 系统: 裸机或Linux
├── 库: Xilinx驱动库
└── 功能: ECG数据处理 + FPGA控制

接口代码:
├── 寄存器操作: Xil_Out32/Xil_In32
├── 内存管理: malloc/memcpy
├── 中断处理: 可选异步模式
└── 用户接口: 串口或网络
```

## 📁 文件依赖关系

### 核心依赖链
```
ecg_trained_classifier.cpp → (HLS) → IP核
                ↓
deploy_ecg.tcl → (Vivado) → 比特流
                ↓  
ecg_app.c → (Vitis) → ARM可执行文件
```

### 详细文件关系
```
源文件层:
├── hls_source/ecg_trained_classifier.cpp (主算法)
├── hls_source/weights.h (神经网络权重)
├── testbench/tb_ecg_classifier.cpp (C++测试)
└── ecg_app.c (ARM应用代码)

生成文件层:
├── ecg_fpga_project/solution1/impl/ip/ (IP核)
├── vivado_project/ECG_FPGA_System.xpr (Vivado项目)
├── vivado_project/.../system.bit (比特流)
└── vitis_workspace/.../ecg_app.elf (ARM程序)

配置文件层:
├── deploy_ecg.tcl (系统集成脚本)
├── README.md (部署指南)
├── PROJECT_STATUS.md (状态汇报)
└── AI_CONTEXT_GUIDE.md (AI助手指南)
```

## 🔗 接口协议详情

### AXI4-Lite控制接口
```
寄存器映射 (基地址: 0x40000000):
├── 0x00: 控制寄存器 (启动/停止)
├── 0x04: 状态寄存器 (完成/错误)
├── 0x10: features数组基地址
├── 0x14: results数组基地址
└── 0x18: 数据长度配置

操作流程:
1. 写入数据地址
2. 写入控制寄存器启动
3. 轮询状态寄存器
4. 读取结果数据
```

### AXI4内存接口  
```
gmem0: 输入特征数据 (46 × 4字节 = 184字节)
gmem1: 输出概率数据 (6 × 4字节 = 24字节)  
gmem2: 临时计算缓存 (动态分配)

内存布局:
├── 输入缓冲区: 46个float值
├── 输出缓冲区: 6个float概率
└── 结果: 最大概率对应的类别ID
```

## ⚡ 性能预期

### 计算性能
```
硬件资源 (Zynq-7020):
├── LUT: ~23,000 / 53,200 (43%)
├── FF: ~12,000 / 106,400 (11%)
├── DSP: ~276 / 220 (125% - 需要优化)
├── BRAM: ~8 / 140 (6%)
└── 功耗: 2-3W

处理延迟:
├── 单次分类: ~5微秒
├── 数据传输: ~1微秒  
├── 总延迟: <10微秒
└── 吞吐量: >100,000次分类/秒
```

### 精度保持
```
量化影响:
├── 浮点→定点: 精度损失<0.1%
├── 权重量化: 16位定点表示
├── 激活量化: 16位定点计算
└── 最终精度: 预期>99%
```

## 🚨 关键约束与限制

### 硬件约束
- DSP资源紧张: 需要优化并行度
- 时钟频率: 最高~150MHz (受DSP限制)
- 内存带宽: DDR3-1066 (8.5GB/s理论)

### 软件约束  
- 实时性要求: 心率监测需要<100ms响应
- 功耗限制: 便携设备要求<5W
- 精度要求: 医疗应用需要>99%准确率

---
*本文档提供完整的技术背景，确保AI助手理解所有依赖关系*
