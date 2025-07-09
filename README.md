# Project
大创项目
# 心电信号处理与分析项目

## 项目简介

本项目实现了对MIT-BIH等心电（ECG）数据的**读取、滤波、小波去噪、心拍分割、归一化、特征提取与可视化**等完整流程，适用于心电信号分析、心律失常检测等科研或教学场景。

---

## 主要功能模块

- **信号读取**：支持MIT-BIH格式数据的读取（`signal_read.py`）。
- **滤波处理**：去除基线漂移和工频干扰（`ecg_filter.py`）。
- **小波去噪**：基于小波变换的ECG信号降噪（`ecg_wavelet_denoising.py`）。
- **心拍分割**：基于R峰和标签的心拍分割，自动输出beats、labels及其索引（`ecg_segmenter.py`）。
- **归一化**：将每个心拍归一化到[-1, 1]区间（`ecg_normalize.py`）。
- **特征提取**：提取RR间期、QRS宽度、QT间期、R/Q/S/T波幅值、ST段斜率等时域特征（`ecg_feature_extraction.py`）。
- **可视化**：多视角展示原始信号、分割心拍及不同类型心拍的平均波形（`ecg_visualizer.py`）。

---

## 依赖环境

- Python 3.7+
- numpy
- scipy
- matplotlib
- wfdb
- pywt

安装依赖：
```
pip install numpy scipy matplotlib wfdb PyWavelets
```

---

## 快速开始

1. **准备数据**  
   下载MIT-BIH数据（如100号记录），放入`data/`目录下。

2. **运行主程序**  
   ```
   python ecg_process.py
   ```

3. **输出说明**  
   - 控制台会输出采样率、心拍数、特征等信息。
   - 生成的心电信号分割与特征可视化图片保存在当前目录下。

---

## 主要文件说明

- `ecg_process.py`：主流程脚本，串联各模块，输出分析与可视化结果。
- `signal_read.py`：心电信号及R峰、标签读取。
- `ecg_filter.py`：滤波处理。
- `ecg_wavelet_denoising.py`：小波去噪。
- `ecg_segmenter.py`：心拍分割，返回beats、labels、beat_indices。
- `ecg_normalize.py`：心拍归一化。
- `ecg_feature_extraction.py`：心拍特征提取。
- `ecg_visualizer.py`：信号与心拍可视化。

---

## 典型流程示意

1. 读取原始信号
2. 滤波去噪
3. 小波降噪
4. 心拍分割
5. 归一化
6. 特征提取
7. 可视化

---

## 联系方式

如有问题或建议，请联系项目维护者