# ECG分类器HLS项目自动化脚本
# 使用方法: 在Vitis HLS控制台执行 source run_hls.tcl

# 创建项目
open_project ecg_fpga_project
set_top ecg_classify_trained

# 添加源文件
add_files hls_source/ecg_trained_classifier.cpp
add_files hls_source/ecg_params.vh

# 添加测试激励
add_files -tb testbench/tb_ecg_classifier.cpp

# 创建解决方案
open_solution "solution1" -flow_target vivado

# 设置目标器件（可根据需要修改）
set_part {xc7z020clg400-1}

# 设置时钟约束
create_clock -period 10 -name default

# 设置优化指令
config_compile -name_max_length 80
config_interface -m_axi_latency 64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_max_widen_bitwidth 512

# 运行C仿真
puts "=== 开始C仿真 ==="
csim_design -clean

# 运行综合
puts "=== 开始C综合 ==="
csynth_design

# 运行RTL协同仿真（可选，耗时较长）
# puts "=== 开始RTL协同仿真 ==="
# cosim_design -trace_level all

# 导出RTL设计
puts "=== 导出RTL设计 ==="
export_design -format ip_catalog -description "ECG心电图分类器IP核"

puts "=== HLS流程完成 ==="
puts "项目文件位置: [pwd]/ecg_fpga_project"
