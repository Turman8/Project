# E# 设置构建项目
open_project ecg_classifier_project -reset

# 添加源文件
add_files hls_source/classifier.cpp
add_files hls_source/weights.cpp  
add_files -tb testbench/test.cpp

# 设置顶层函数
set_top ecg_classifier优化版本构建脚本 (Windows PowerShell)
# 目标：150-180 DSP，10-15x性能提升

# 设置构建项目
open_project ecg_practical_project -reset

# 添加源文件
add_files hls_source/classifier_practical.cpp
add_files hls_source/weights_fixed.cpp  
add_files -tb testbench/test_practical.cpp

# 设置顶层函数
set_top ecg_classifier_practical

# 创建解决方案
open_solution "solution1" -flow_target vivado

# 设置目标器件为 Zynq-7020
set_part {xc7z020clg484-1}

# 设置时钟周期为 10ns (100MHz)
create_clock -period 10 -name default

# C 语言仿真
csim_design

# C 语言综合  
csynth_design

# 生成综合报告
export_design -format ip_catalog -description "ECG Classifier Practical" -vendor "user" -library "ECG" -version "3.0"

# 退出 HLS
exit
