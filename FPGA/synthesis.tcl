open_project ecg_classifier_project
open_solution solution1

# 设置目标器件和时钟
set_part {xc7z020clg484-1}
create_clock -period 6.67 -name default

# 添加源文件
add_files hls_source/classifier.cpp
add_files hls_source/weights.cpp
add_files -tb testbench/test.cpp
set_top ecg_classifier

csynth_design
exit
