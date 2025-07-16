############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project ecg_fpga_project
set_top ecg_classify_trained
add_files hls_source/weights.h
add_files hls_source/ecg_params.vh
add_files hls_source/ecg_trained_classifier.cpp
add_files -tb testbench/tb_ecg_classifier.cpp -cflags "-Wno-unknown-pragmas"
open_solution "solution1"

create_clock -period 2 -name default
source "./ecg_fpga_project/solution1/directives.tcl"
csim_design -clean
csynth_design
cosim_design
export_design -format ip_catalog
