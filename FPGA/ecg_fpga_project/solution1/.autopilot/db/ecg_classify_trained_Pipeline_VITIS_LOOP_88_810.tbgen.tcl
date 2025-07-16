set moduleName ecg_classify_trained_Pipeline_VITIS_LOOP_88_810
set isTopModule 0
set isCombinational 0
set isDatapathOnly 0
set isPipelined 1
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {ecg_classify_trained_Pipeline_VITIS_LOOP_88_810}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict dense_3_weights { MEM_WIDTH 32 MEM_SIZE 768 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ hidden3 int 16 regular  }
	{ hidden3_1 int 16 regular  }
	{ hidden3_2 int 16 regular  }
	{ hidden3_3 int 16 regular  }
	{ hidden3_4 int 16 regular  }
	{ hidden3_5 int 16 regular  }
	{ hidden3_6 int 16 regular  }
	{ hidden3_7 int 16 regular  }
	{ hidden3_8 int 16 regular  }
	{ hidden3_9 int 16 regular  }
	{ hidden3_10 int 16 regular  }
	{ hidden3_11 int 16 regular  }
	{ hidden3_12 int 16 regular  }
	{ hidden3_13 int 16 regular  }
	{ hidden3_14 int 16 regular  }
	{ hidden3_15 int 16 regular  }
	{ hidden3_16 int 16 regular  }
	{ hidden3_17 int 16 regular  }
	{ hidden3_18 int 16 regular  }
	{ hidden3_19 int 16 regular  }
	{ hidden3_20 int 16 regular  }
	{ hidden3_21 int 16 regular  }
	{ hidden3_22 int 16 regular  }
	{ hidden3_23 int 16 regular  }
	{ hidden3_24 int 16 regular  }
	{ hidden3_25 int 16 regular  }
	{ hidden3_26 int 16 regular  }
	{ hidden3_27 int 16 regular  }
	{ hidden3_28 int 16 regular  }
	{ hidden3_29 int 16 regular  }
	{ hidden3_30 int 16 regular  }
	{ hidden3_31 int 16 regular  }
	{ sum_9_out int 24 regular {pointer 1}  }
	{ dense_3_weights float 32 regular {array 192 { 1 } 1 1 } {global 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "hidden3", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_1", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_2", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_3", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_4", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_5", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_6", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_7", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_8", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_9", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_10", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_11", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_12", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_13", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_14", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_15", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_16", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_17", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_18", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_19", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_20", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_21", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_22", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_23", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_24", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_25", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_26", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_27", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_28", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_29", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden3_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sum_9_out", "interface" : "wire", "bitwidth" : 24, "direction" : "WRITEONLY"} , 
 	{ "Name" : "dense_3_weights", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 46
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ hidden3 sc_in sc_lv 16 signal 0 } 
	{ hidden3_1 sc_in sc_lv 16 signal 1 } 
	{ hidden3_2 sc_in sc_lv 16 signal 2 } 
	{ hidden3_3 sc_in sc_lv 16 signal 3 } 
	{ hidden3_4 sc_in sc_lv 16 signal 4 } 
	{ hidden3_5 sc_in sc_lv 16 signal 5 } 
	{ hidden3_6 sc_in sc_lv 16 signal 6 } 
	{ hidden3_7 sc_in sc_lv 16 signal 7 } 
	{ hidden3_8 sc_in sc_lv 16 signal 8 } 
	{ hidden3_9 sc_in sc_lv 16 signal 9 } 
	{ hidden3_10 sc_in sc_lv 16 signal 10 } 
	{ hidden3_11 sc_in sc_lv 16 signal 11 } 
	{ hidden3_12 sc_in sc_lv 16 signal 12 } 
	{ hidden3_13 sc_in sc_lv 16 signal 13 } 
	{ hidden3_14 sc_in sc_lv 16 signal 14 } 
	{ hidden3_15 sc_in sc_lv 16 signal 15 } 
	{ hidden3_16 sc_in sc_lv 16 signal 16 } 
	{ hidden3_17 sc_in sc_lv 16 signal 17 } 
	{ hidden3_18 sc_in sc_lv 16 signal 18 } 
	{ hidden3_19 sc_in sc_lv 16 signal 19 } 
	{ hidden3_20 sc_in sc_lv 16 signal 20 } 
	{ hidden3_21 sc_in sc_lv 16 signal 21 } 
	{ hidden3_22 sc_in sc_lv 16 signal 22 } 
	{ hidden3_23 sc_in sc_lv 16 signal 23 } 
	{ hidden3_24 sc_in sc_lv 16 signal 24 } 
	{ hidden3_25 sc_in sc_lv 16 signal 25 } 
	{ hidden3_26 sc_in sc_lv 16 signal 26 } 
	{ hidden3_27 sc_in sc_lv 16 signal 27 } 
	{ hidden3_28 sc_in sc_lv 16 signal 28 } 
	{ hidden3_29 sc_in sc_lv 16 signal 29 } 
	{ hidden3_30 sc_in sc_lv 16 signal 30 } 
	{ hidden3_31 sc_in sc_lv 16 signal 31 } 
	{ sum_9_out sc_out sc_lv 24 signal 32 } 
	{ sum_9_out_ap_vld sc_out sc_logic 1 outvld 32 } 
	{ dense_3_weights_address0 sc_out sc_lv 8 signal 33 } 
	{ dense_3_weights_ce0 sc_out sc_logic 1 signal 33 } 
	{ dense_3_weights_q0 sc_in sc_lv 32 signal 33 } 
	{ grp_fu_13213_p_din0 sc_out sc_lv 32 signal -1 } 
	{ grp_fu_13213_p_dout0 sc_in sc_lv 64 signal -1 } 
	{ grp_fu_13213_p_ce sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst", "role": "default" }} , 
 	{ "name": "ap_start", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "start", "bundle":{"name": "ap_start", "role": "default" }} , 
 	{ "name": "ap_done", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "predone", "bundle":{"name": "ap_done", "role": "default" }} , 
 	{ "name": "ap_idle", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "done", "bundle":{"name": "ap_idle", "role": "default" }} , 
 	{ "name": "ap_ready", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "ready", "bundle":{"name": "ap_ready", "role": "default" }} , 
 	{ "name": "hidden3", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3", "role": "default" }} , 
 	{ "name": "hidden3_1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_1", "role": "default" }} , 
 	{ "name": "hidden3_2", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_2", "role": "default" }} , 
 	{ "name": "hidden3_3", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_3", "role": "default" }} , 
 	{ "name": "hidden3_4", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_4", "role": "default" }} , 
 	{ "name": "hidden3_5", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_5", "role": "default" }} , 
 	{ "name": "hidden3_6", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_6", "role": "default" }} , 
 	{ "name": "hidden3_7", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_7", "role": "default" }} , 
 	{ "name": "hidden3_8", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_8", "role": "default" }} , 
 	{ "name": "hidden3_9", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_9", "role": "default" }} , 
 	{ "name": "hidden3_10", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_10", "role": "default" }} , 
 	{ "name": "hidden3_11", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_11", "role": "default" }} , 
 	{ "name": "hidden3_12", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_12", "role": "default" }} , 
 	{ "name": "hidden3_13", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_13", "role": "default" }} , 
 	{ "name": "hidden3_14", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_14", "role": "default" }} , 
 	{ "name": "hidden3_15", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_15", "role": "default" }} , 
 	{ "name": "hidden3_16", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_16", "role": "default" }} , 
 	{ "name": "hidden3_17", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_17", "role": "default" }} , 
 	{ "name": "hidden3_18", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_18", "role": "default" }} , 
 	{ "name": "hidden3_19", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_19", "role": "default" }} , 
 	{ "name": "hidden3_20", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_20", "role": "default" }} , 
 	{ "name": "hidden3_21", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_21", "role": "default" }} , 
 	{ "name": "hidden3_22", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_22", "role": "default" }} , 
 	{ "name": "hidden3_23", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_23", "role": "default" }} , 
 	{ "name": "hidden3_24", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_24", "role": "default" }} , 
 	{ "name": "hidden3_25", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_25", "role": "default" }} , 
 	{ "name": "hidden3_26", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_26", "role": "default" }} , 
 	{ "name": "hidden3_27", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_27", "role": "default" }} , 
 	{ "name": "hidden3_28", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_28", "role": "default" }} , 
 	{ "name": "hidden3_29", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_29", "role": "default" }} , 
 	{ "name": "hidden3_30", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_30", "role": "default" }} , 
 	{ "name": "hidden3_31", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden3_31", "role": "default" }} , 
 	{ "name": "sum_9_out", "direction": "out", "datatype": "sc_lv", "bitwidth":24, "type": "signal", "bundle":{"name": "sum_9_out", "role": "default" }} , 
 	{ "name": "sum_9_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "sum_9_out", "role": "ap_vld" }} , 
 	{ "name": "dense_3_weights_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "dense_3_weights", "role": "address0" }} , 
 	{ "name": "dense_3_weights_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "dense_3_weights", "role": "ce0" }} , 
 	{ "name": "dense_3_weights_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "dense_3_weights", "role": "q0" }} , 
 	{ "name": "grp_fu_13213_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_13213_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "grp_fu_13213_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_13213_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_810",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "42", "EstimateLatencyMax" : "42",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "hidden3", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_7", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_9", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_10", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_12", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_13", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_14", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_15", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_16", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_17", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_18", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_19", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_20", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_21", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_22", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_23", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_24", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_25", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_26", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_27", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_28", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_29", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_30", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden3_31", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_9_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_65_5_16_1_1_U913", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U914", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mac_muladd_16s_16s_32ns_32_4_1_U915", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_810 {
		hidden3 {Type I LastRead 0 FirstWrite -1}
		hidden3_1 {Type I LastRead 0 FirstWrite -1}
		hidden3_2 {Type I LastRead 0 FirstWrite -1}
		hidden3_3 {Type I LastRead 0 FirstWrite -1}
		hidden3_4 {Type I LastRead 0 FirstWrite -1}
		hidden3_5 {Type I LastRead 0 FirstWrite -1}
		hidden3_6 {Type I LastRead 0 FirstWrite -1}
		hidden3_7 {Type I LastRead 0 FirstWrite -1}
		hidden3_8 {Type I LastRead 0 FirstWrite -1}
		hidden3_9 {Type I LastRead 0 FirstWrite -1}
		hidden3_10 {Type I LastRead 0 FirstWrite -1}
		hidden3_11 {Type I LastRead 0 FirstWrite -1}
		hidden3_12 {Type I LastRead 0 FirstWrite -1}
		hidden3_13 {Type I LastRead 0 FirstWrite -1}
		hidden3_14 {Type I LastRead 0 FirstWrite -1}
		hidden3_15 {Type I LastRead 0 FirstWrite -1}
		hidden3_16 {Type I LastRead 0 FirstWrite -1}
		hidden3_17 {Type I LastRead 0 FirstWrite -1}
		hidden3_18 {Type I LastRead 0 FirstWrite -1}
		hidden3_19 {Type I LastRead 0 FirstWrite -1}
		hidden3_20 {Type I LastRead 0 FirstWrite -1}
		hidden3_21 {Type I LastRead 0 FirstWrite -1}
		hidden3_22 {Type I LastRead 0 FirstWrite -1}
		hidden3_23 {Type I LastRead 0 FirstWrite -1}
		hidden3_24 {Type I LastRead 0 FirstWrite -1}
		hidden3_25 {Type I LastRead 0 FirstWrite -1}
		hidden3_26 {Type I LastRead 0 FirstWrite -1}
		hidden3_27 {Type I LastRead 0 FirstWrite -1}
		hidden3_28 {Type I LastRead 0 FirstWrite -1}
		hidden3_29 {Type I LastRead 0 FirstWrite -1}
		hidden3_30 {Type I LastRead 0 FirstWrite -1}
		hidden3_31 {Type I LastRead 0 FirstWrite -1}
		sum_9_out {Type O LastRead -1 FirstWrite 8}
		dense_3_weights {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "42", "Max" : "42"}
	, {"Name" : "Interval", "Min" : "42", "Max" : "42"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	hidden3 { ap_none {  { hidden3 in_data 0 16 } } }
	hidden3_1 { ap_none {  { hidden3_1 in_data 0 16 } } }
	hidden3_2 { ap_none {  { hidden3_2 in_data 0 16 } } }
	hidden3_3 { ap_none {  { hidden3_3 in_data 0 16 } } }
	hidden3_4 { ap_none {  { hidden3_4 in_data 0 16 } } }
	hidden3_5 { ap_none {  { hidden3_5 in_data 0 16 } } }
	hidden3_6 { ap_none {  { hidden3_6 in_data 0 16 } } }
	hidden3_7 { ap_none {  { hidden3_7 in_data 0 16 } } }
	hidden3_8 { ap_none {  { hidden3_8 in_data 0 16 } } }
	hidden3_9 { ap_none {  { hidden3_9 in_data 0 16 } } }
	hidden3_10 { ap_none {  { hidden3_10 in_data 0 16 } } }
	hidden3_11 { ap_none {  { hidden3_11 in_data 0 16 } } }
	hidden3_12 { ap_none {  { hidden3_12 in_data 0 16 } } }
	hidden3_13 { ap_none {  { hidden3_13 in_data 0 16 } } }
	hidden3_14 { ap_none {  { hidden3_14 in_data 0 16 } } }
	hidden3_15 { ap_none {  { hidden3_15 in_data 0 16 } } }
	hidden3_16 { ap_none {  { hidden3_16 in_data 0 16 } } }
	hidden3_17 { ap_none {  { hidden3_17 in_data 0 16 } } }
	hidden3_18 { ap_none {  { hidden3_18 in_data 0 16 } } }
	hidden3_19 { ap_none {  { hidden3_19 in_data 0 16 } } }
	hidden3_20 { ap_none {  { hidden3_20 in_data 0 16 } } }
	hidden3_21 { ap_none {  { hidden3_21 in_data 0 16 } } }
	hidden3_22 { ap_none {  { hidden3_22 in_data 0 16 } } }
	hidden3_23 { ap_none {  { hidden3_23 in_data 0 16 } } }
	hidden3_24 { ap_none {  { hidden3_24 in_data 0 16 } } }
	hidden3_25 { ap_none {  { hidden3_25 in_data 0 16 } } }
	hidden3_26 { ap_none {  { hidden3_26 in_data 0 16 } } }
	hidden3_27 { ap_none {  { hidden3_27 in_data 0 16 } } }
	hidden3_28 { ap_none {  { hidden3_28 in_data 0 16 } } }
	hidden3_29 { ap_none {  { hidden3_29 in_data 0 16 } } }
	hidden3_30 { ap_none {  { hidden3_30 in_data 0 16 } } }
	hidden3_31 { ap_none {  { hidden3_31 in_data 0 16 } } }
	sum_9_out { ap_vld {  { sum_9_out out_data 1 24 }  { sum_9_out_ap_vld out_vld 1 1 } } }
	dense_3_weights { ap_memory {  { dense_3_weights_address0 mem_address 1 8 }  { dense_3_weights_ce0 mem_ce 1 1 }  { dense_3_weights_q0 mem_dout 0 32 } } }
}
