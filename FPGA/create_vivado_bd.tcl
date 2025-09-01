# Auto-generate a Vivado project and Block Design with PS7 + ecg_classifier IP
# Vivado version: 2019.2

# Configuration
set part_name        "xc7z020clg484-1"
set proj_root        [file normalize "./vivado_ecg_proj"]
set proj_name        "ecg_proj"
set bd_name          "design_1"
set ip_repo_dir      [file normalize "./ecg_classifier_project/solution1/impl/ip"]
set fclk0_mhz        150.0

# Clean and create project
if {[file exists $proj_root]} {
  file delete -force $proj_root
}
create_project $proj_name $proj_root -part $part_name
set_property target_language Verilog [current_project]

# Add HLS packaged IP repository
if {![file isdirectory $ip_repo_dir]} {
  puts "ERROR: IP repository not found: $ip_repo_dir"
  exit 1
}
set_property ip_repo_paths [list $ip_repo_dir] [current_fileset]
update_ip_catalog

# Create Block Design
create_bd_design $bd_name
# PS7
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0
# Minimal PS7 configuration: enable GP0 master, HP0/HP1 slaves, enable FCLK0
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP1 {1} \
  CONFIG.PCW_EN_CLK0_PORT {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [format %.3f $fclk0_mhz] \
] [get_bd_cells ps7_0]

# Make DDR/FIXED_IO external
make_bd_intf_pins_external [get_bd_intf_pins ps7_0/DDR]
make_bd_intf_pins_external [get_bd_intf_pins ps7_0/FIXED_IO]

# Add ecg_classifier IP
set ecg_vlnv "user:ECG:ecg_classifier:1.0"
if {[llength [get_ipdefs -all $ecg_vlnv]] == 0} {
  puts "ERROR: IP $ecg_vlnv not found in catalog. Check ip_repo_paths."
  exit 2
}
create_bd_cell -type ip -vlnv $ecg_vlnv ecg_classifier_0

# Add SmartConnect for AXI protocol conversion to AXI4-Lite
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]

# AXI connections
# PS7 GP0 master -> SmartConnect S00
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] [get_bd_intf_pins smartconnect_0/S00_AXI]
# SmartConnect M00 -> ecg_classifier s_axi_AXILiteS
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins ecg_classifier_0/s_axi_AXILiteS]

# Data path SmartConnects for gmem0/gmem1 (handles AXI4->AXI3 + 32->64 width)
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_data0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_data0]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 sc_data1
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_data1]

# gmem0 path
connect_bd_intf_net [get_bd_intf_pins ecg_classifier_0/m_axi_gmem0] [get_bd_intf_pins sc_data0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_data0/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP0]
# gmem1 path
connect_bd_intf_net [get_bd_intf_pins ecg_classifier_0/m_axi_gmem1] [get_bd_intf_pins sc_data1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_data1/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP1]

# Clocking: drive all AXI ACLK from FCLK_CLK0
foreach clk_pin {M_AXI_GP0_ACLK S_AXI_HP0_ACLK S_AXI_HP1_ACLK} {
  connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins ps7_0/$clk_pin]
}
# SmartConnect aclk and aresetn
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins sc_data0/aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins sc_data1/aclk]
# IP core clock/reset
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins ecg_classifier_0/ap_clk]

# Synchronous reset generation
create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 inv_reset
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] [get_bd_cells inv_reset]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_sync
# ext_reset_in expects active-high; invert FCLK_RESET0_N
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins inv_reset/Op1]
connect_bd_net [get_bd_pins inv_reset/Res]       [get_bd_pins rst_sync/ext_reset_in]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins rst_sync/slowest_sync_clk]
# Distribute synchronous active-low resets
connect_bd_net [get_bd_pins rst_sync/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn]
connect_bd_net [get_bd_pins rst_sync/peripheral_aresetn] [get_bd_pins sc_data0/aresetn]
connect_bd_net [get_bd_pins rst_sync/peripheral_aresetn] [get_bd_pins sc_data1/aresetn]
connect_bd_net [get_bd_pins rst_sync/peripheral_aresetn] [get_bd_pins ecg_classifier_0/ap_rst_n]

# Address assignment
assign_bd_address

# Validate and save BD
validate_bd_design
save_bd_design

# Generate output products and HDL wrapper
make_wrapper -files [get_files "$proj_root/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"] -top
add_files -norecurse "$proj_root/$proj_name.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
update_compile_order -fileset sources_1
set_property top ${bd_name}_wrapper [current_fileset]

# Write reports (batch-safe)
file mkdir "$proj_root/reports"
report_ip_status -file "$proj_root/reports/ip_status.rpt" -quiet

puts "INFO: Project created at $proj_root. Top=${bd_name}_wrapper"
exit 0
