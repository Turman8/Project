# ECG分类器FPGA部署脚本
# 保持99.08%准确率，46维输入，6类输出
# 基于真实MIT-BIH训练权重

set project_name "ECG_FPGA_System"
set project_dir "D:/Git/Project/FPGA/vivado_project"
set part_name "xc7z020clg400-1"
set hls_ip_path "D:/Git/Project/FPGA/ecg_fpga_project/solution1/impl/ip"

# 清理并创建项目
file delete -force $project_dir
file mkdir $project_dir
create_project $project_name $project_dir -part $part_name -force

# 添加HLS IP
set_property ip_repo_paths $hls_ip_path [current_project]
update_ip_catalog

# 创建Block Design
create_bd_design "ecg_system"

# 添加Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# 配置Zynq（启用必要接口）
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
] [get_bd_cells processing_system7_0]

# 添加ECG分类器IP
create_bd_cell -type ip -vlnv xilinx.com:hls:ecg_classify_trained:1.0 ecg_classify_trained_0

# 自动连接（解决协议兼容性）
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} [get_bd_cells processing_system7_0]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/processing_system7_0/M_AXI_GP0" intc_ip "New AXI Interconnect" Clk_xbar "Auto" Clk_master "Auto" Clk_slave "Auto"} [get_bd_intf_pins ecg_classify_trained_0/s_axi_control]

# 手动创建AXI互连以解决HP0连接问题
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {3}] [get_bd_cells axi_interconnect_0]

# 手动连接时钟和复位（简化方式）
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S01_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/S02_ACLK]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins axi_interconnect_0/M00_ACLK]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/S01_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/S02_ARESETN]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins axi_interconnect_0/M00_ARESETN]

# 连接HP0接口
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins ecg_classify_trained_0/m_axi_gmem0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ecg_classify_trained_0/m_axi_gmem1] [get_bd_intf_pins axi_interconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins ecg_classify_trained_0/m_axi_gmem2] [get_bd_intf_pins axi_interconnect_0/S02_AXI]

# 分配地址并验证
assign_bd_address
validate_bd_design
save_bd_design

# 生成HDL包装器
make_wrapper -files [get_files $project_dir/$project_name.srcs/sources_1/bd/ecg_system/ecg_system.bd] -top
add_files -norecurse $project_dir/$project_name.gen/sources_1/bd/ecg_system/hdl/ecg_system_wrapper.v
set_property top ecg_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "项目创建完成！下一步：launch_runs impl_1 -to_step write_bitstream"
