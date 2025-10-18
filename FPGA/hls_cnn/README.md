# Wavelet-CNN Vitis HLS Template

本目录提供了针对 `main.py` 训练所得 CWT+CNN 模型的 Vitis HLS 推理模板。

## 结构

- `cnn_inference.hpp`：顶层函数声明、定点类型定义，包含自动导出的 `cnn_weights.h`。
- `cnn_inference.cpp`：卷积、批归一化、池化（融合在同一循环中以减少 BRAM 占用与延迟）、全连接及 Softmax 推理实现。
- `cnn_testbench.cpp`：可选的 C 仿真测试脚本，占位示例。

> **提示**：运行训练脚本后，`outputs/fpga_deployment_*/weights/cnn_weights.h` 会生成实际权重数组和尺寸常量。将其复制到本目录即可编译。

## 使用步骤

1. 运行 `python main.py` 完成模型训练、量化与部署包生成。
2. 将 `outputs/fpga_deployment_*/weights/cnn_weights.h` 复制到本目录。
3. 进入此目录执行 `vitis_hls -f run_hls.tcl` 或在 Vitis HLS GUI 中新建工程，添加 `cnn_inference.cpp` 与 `cnn_inference.hpp`。
4. 顶层函数为 `ecg_cnn_inference`，输入是 64×300 的归一化小波幅值图，输出为 `NUM_CLASSES` 维概率向量。
5. 根据综合报告调整 `CNN_HLS_TOTAL_BITS`/`CNN_HLS_INTEGER_BITS` 或循环 pragma 以满足资源约束。

## 约束

- 当前实现为参考版本，重点在于说明数据流、BN 折叠和 Softmax 处理流程。
- 若需要更高性能，可在卷积/池化循环中调节 `#pragma HLS PIPELINE` 或增加适度的 `UNROLL`/`ARRAY_PARTITION`；当前模板默认以单 MAC 复用实现，兼顾资源压力与时延。
- 默认定点格式为 `ap_fixed<16,6>`，与导出的权重头文件保持一致，可在 `cnn_inference.hpp` 中统一修改。

## 与 Vitis Libraries 的协同

- 模板中的卷积、池化、向量化 Softmax 已经封装为独立函数，若希望直接复用 [Vitis Libraries](https://xilinx.github.io/Vitis_Libraries/) 中的 `hpc`/`dsp`/`vision` 算子，可在相同接口处替换为 `xf::blas::gemm`, `xf::dsp::aie::conv` 等实现。
- 权重与激活张量的排布遵循 NCHW→分块流式的方式，已在 `cnn_inference.hpp` 中定义常量尺寸，因此替换算子时只需保持输入/输出缓冲的行主序布局；若算子需要 tile 化，可依据 `INPUT_HEIGHT`, `INPUT_WIDTH`, `INPUT_CHANNELS` 拆分。
- 重新综合前请在 `run_hls.tcl` 中添加对应的 Vitis Libraries include 与 `CONFIG.addIncludePath` 设置，同时确认 `ap_fixed` 精度与库函数的模板参数一致，以避免类型推断失败。
