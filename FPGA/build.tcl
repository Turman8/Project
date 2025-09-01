# ECG分类器 HLS 构建与仿真脚本（Vivado HLS 2019.2）
# 目标：一键执行 C 仿真 → 综合 → C/RTL 协同仿真（输出波形）→ 导出 IP

# 打开/重置项目
open_project ecg_classifier_project -reset

# 添加源文件与测试台（相对路径从 FPGA 目录运行）
add_files hls_source/classifier.cpp
add_files hls_source/weights.cpp
add_files -tb testbench/test.cpp

# 设置顶层函数
set_top ecg_classifier

# 创建解决方案并指定 Vivado 流程
# Vivado HLS 2019.2 不支持 -flow_target 选项，直接打开解决方案
open_solution "solution1"

# 目标器件与时钟
set_part {xc7z020clg484-1}
create_clock -period 6.67 -name default   ;# 150MHz

# C 仿真（功能快速校验）
csim_design

# 综合
csynth_design

# C/RTL 协同仿真（使用 XSIM，启用波形，2019.2 用 trace_level）
cosim_design -tool xsim -rtl verilog -trace_level all

# 导出 IP（供 Vivado 集成）
export_design -format ip_catalog -description "ECG Classifier Optimized" -vendor "user" -library "ECG" -version "1.0"

exit
