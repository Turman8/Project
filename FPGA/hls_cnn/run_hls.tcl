open_project ecg_cnn_hls -reset
set_top ecg_cnn_inference
add_files cnn_inference.cpp -cflags "-I."
add_files cnn_testbench.cpp -cflags "-I."
open_solution "solution1" -reset
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
