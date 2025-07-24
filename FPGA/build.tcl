# ECG分类器HLS构建脚本
# 目标：优化版本，150-180 DSP使用，性能提升10-15倍

# 设置构建项目
open_project ecg_classifier_project -reset

# 添加源文件
add_files hls_source/classifier.cpp
add_files hls_source/weights.cpp  
add_files -tb testbench/test.cpp

# 设置顶层函数
set_top ecg_classifier

# 创建解决方案
open_solution "solution1" -flow_target vivado

# 设置目标器件为 Zynq-7020
set_part {xc7z020clg484-1}

# 设置时钟周期为 6.67ns (150MHz)
create_clock -period 6.67 -name default

# C 语言仿真
csim_design

# C 语言综合  
csynth_design

# 协同仿真 (可选，耗时较长)
# cosim_design

# 生成IP核
export_design -format ip_catalog -description "ECG Classifier Optimized" -vendor "user" -library "ECG" -version "1.0"

# 退出 HLS
exit
