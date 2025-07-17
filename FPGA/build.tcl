# Final AXI3 Compatible HLS Build for Zynq-7000
# Force AXI3 protocol compatibility for HP ports

# Switch to correct working directory
cd "D:/Git/Project/FPGA"

# Delete old project
if {[file exists hls_project]} {
    file delete -force hls_project
}

# Create project
open_project hls_project
set_top ecg_classify_manual_fixed

# Add source files
add_files hls_source/classifier.cpp
add_files hls_source/classifier.h
add_files hls_source/weights.h
add_files hls_source/params.vh
add_files -tb testbench/testbench.cpp -cflags "-Wno-unknown-pragmas"

# Create solution
open_solution "solution1" -flow_target vivado
set_part xc7z020-clg400-1
create_clock -period 10 -name default

# Compilation configuration 
config_compile -name_max_length 80

# Critical AXI3 configuration for Zynq-7000 compatibility
config_interface -m_axi_latency 64
config_interface -m_axi_alignment_byte_size 64  
config_interface -m_axi_max_widen_bitwidth 512
# Force AXI3 protocol - no 64-bit addressing
config_interface -m_axi_addr64=false
# Disable advanced AXI4 features 
config_interface -m_axi_max_read_burst_length=16
config_interface -m_axi_max_write_burst_length=16

# Set top function
set_directive_top -name ecg_classify_manual_fixed ecg_classify_manual_fixed

# Resource allocation
set_directive_allocation -limit 256 -type operation ecg_classify_manual_fixed mul
set_directive_allocation -limit 256 -type operation ecg_classify_manual_fixed add

puts "Starting C simulation (final AXI3 compatible)..."
csim_design -clean

puts "Starting C synthesis (final AXI3 compatible)..."
csynth_design

puts "Exporting IP core (AXI3 compatible)..."
export_design -format ip_catalog

puts "Final AXI3-compatible IP core creation completed!"
