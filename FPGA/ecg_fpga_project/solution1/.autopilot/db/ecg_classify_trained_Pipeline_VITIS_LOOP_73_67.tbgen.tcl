set moduleName ecg_classify_trained_Pipeline_VITIS_LOOP_73_67
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
set C_modelName {ecg_classify_trained_Pipeline_VITIS_LOOP_73_67}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict dense_2_weights { MEM_WIDTH 32 MEM_SIZE 8192 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ sext_ln71_5 int 24 regular  }
	{ tmp_112 int 4 regular  }
	{ hidden2 int 16 regular  }
	{ hidden2_1 int 16 regular  }
	{ hidden2_2 int 16 regular  }
	{ hidden2_3 int 16 regular  }
	{ hidden2_4 int 16 regular  }
	{ hidden2_5 int 16 regular  }
	{ hidden2_6 int 16 regular  }
	{ hidden2_7 int 16 regular  }
	{ hidden2_8 int 16 regular  }
	{ hidden2_9 int 16 regular  }
	{ hidden2_10 int 16 regular  }
	{ hidden2_11 int 16 regular  }
	{ hidden2_12 int 16 regular  }
	{ hidden2_13 int 16 regular  }
	{ hidden2_14 int 16 regular  }
	{ hidden2_15 int 16 regular  }
	{ hidden2_16 int 16 regular  }
	{ hidden2_17 int 16 regular  }
	{ hidden2_18 int 16 regular  }
	{ hidden2_19 int 16 regular  }
	{ hidden2_20 int 16 regular  }
	{ hidden2_21 int 16 regular  }
	{ hidden2_22 int 16 regular  }
	{ hidden2_23 int 16 regular  }
	{ hidden2_24 int 16 regular  }
	{ hidden2_25 int 16 regular  }
	{ hidden2_26 int 16 regular  }
	{ hidden2_27 int 16 regular  }
	{ hidden2_28 int 16 regular  }
	{ hidden2_29 int 16 regular  }
	{ hidden2_30 int 16 regular  }
	{ hidden2_31 int 16 regular  }
	{ hidden2_32 int 16 regular  }
	{ hidden2_33 int 16 regular  }
	{ hidden2_34 int 16 regular  }
	{ hidden2_35 int 16 regular  }
	{ hidden2_36 int 16 regular  }
	{ hidden2_37 int 16 regular  }
	{ hidden2_38 int 16 regular  }
	{ hidden2_39 int 16 regular  }
	{ hidden2_40 int 16 regular  }
	{ hidden2_41 int 16 regular  }
	{ hidden2_42 int 16 regular  }
	{ hidden2_43 int 16 regular  }
	{ hidden2_44 int 16 regular  }
	{ hidden2_45 int 16 regular  }
	{ hidden2_46 int 16 regular  }
	{ hidden2_47 int 16 regular  }
	{ hidden2_48 int 16 regular  }
	{ hidden2_49 int 16 regular  }
	{ hidden2_50 int 16 regular  }
	{ hidden2_51 int 16 regular  }
	{ hidden2_52 int 16 regular  }
	{ hidden2_53 int 16 regular  }
	{ hidden2_54 int 16 regular  }
	{ hidden2_55 int 16 regular  }
	{ hidden2_56 int 16 regular  }
	{ hidden2_57 int 16 regular  }
	{ hidden2_58 int 16 regular  }
	{ hidden2_59 int 16 regular  }
	{ hidden2_60 int 16 regular  }
	{ hidden2_61 int 16 regular  }
	{ hidden2_62 int 16 regular  }
	{ hidden2_63 int 16 regular  }
	{ sum_26_out int 24 regular {pointer 1}  }
	{ dense_2_weights float 32 regular {array 2048 { 1 } 1 1 } {global 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "sext_ln71_5", "interface" : "wire", "bitwidth" : 24, "direction" : "READONLY"} , 
 	{ "Name" : "tmp_112", "interface" : "wire", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_1", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_2", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_3", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_4", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_5", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_6", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_7", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_8", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_9", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_10", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_11", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_12", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_13", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_14", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_15", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_16", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_17", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_18", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_19", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_20", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_21", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_22", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_23", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_24", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_25", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_26", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_27", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_28", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_29", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_32", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_33", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_34", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_35", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_36", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_37", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_38", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_39", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_40", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_41", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_42", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_43", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_44", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_45", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_46", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_47", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_48", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_49", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_50", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_51", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_52", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_53", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_54", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_55", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_56", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_57", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_58", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_59", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_60", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_61", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_62", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden2_63", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sum_26_out", "interface" : "wire", "bitwidth" : 24, "direction" : "WRITEONLY"} , 
 	{ "Name" : "dense_2_weights", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 80
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ sext_ln71_5 sc_in sc_lv 24 signal 0 } 
	{ tmp_112 sc_in sc_lv 4 signal 1 } 
	{ hidden2 sc_in sc_lv 16 signal 2 } 
	{ hidden2_1 sc_in sc_lv 16 signal 3 } 
	{ hidden2_2 sc_in sc_lv 16 signal 4 } 
	{ hidden2_3 sc_in sc_lv 16 signal 5 } 
	{ hidden2_4 sc_in sc_lv 16 signal 6 } 
	{ hidden2_5 sc_in sc_lv 16 signal 7 } 
	{ hidden2_6 sc_in sc_lv 16 signal 8 } 
	{ hidden2_7 sc_in sc_lv 16 signal 9 } 
	{ hidden2_8 sc_in sc_lv 16 signal 10 } 
	{ hidden2_9 sc_in sc_lv 16 signal 11 } 
	{ hidden2_10 sc_in sc_lv 16 signal 12 } 
	{ hidden2_11 sc_in sc_lv 16 signal 13 } 
	{ hidden2_12 sc_in sc_lv 16 signal 14 } 
	{ hidden2_13 sc_in sc_lv 16 signal 15 } 
	{ hidden2_14 sc_in sc_lv 16 signal 16 } 
	{ hidden2_15 sc_in sc_lv 16 signal 17 } 
	{ hidden2_16 sc_in sc_lv 16 signal 18 } 
	{ hidden2_17 sc_in sc_lv 16 signal 19 } 
	{ hidden2_18 sc_in sc_lv 16 signal 20 } 
	{ hidden2_19 sc_in sc_lv 16 signal 21 } 
	{ hidden2_20 sc_in sc_lv 16 signal 22 } 
	{ hidden2_21 sc_in sc_lv 16 signal 23 } 
	{ hidden2_22 sc_in sc_lv 16 signal 24 } 
	{ hidden2_23 sc_in sc_lv 16 signal 25 } 
	{ hidden2_24 sc_in sc_lv 16 signal 26 } 
	{ hidden2_25 sc_in sc_lv 16 signal 27 } 
	{ hidden2_26 sc_in sc_lv 16 signal 28 } 
	{ hidden2_27 sc_in sc_lv 16 signal 29 } 
	{ hidden2_28 sc_in sc_lv 16 signal 30 } 
	{ hidden2_29 sc_in sc_lv 16 signal 31 } 
	{ hidden2_30 sc_in sc_lv 16 signal 32 } 
	{ hidden2_31 sc_in sc_lv 16 signal 33 } 
	{ hidden2_32 sc_in sc_lv 16 signal 34 } 
	{ hidden2_33 sc_in sc_lv 16 signal 35 } 
	{ hidden2_34 sc_in sc_lv 16 signal 36 } 
	{ hidden2_35 sc_in sc_lv 16 signal 37 } 
	{ hidden2_36 sc_in sc_lv 16 signal 38 } 
	{ hidden2_37 sc_in sc_lv 16 signal 39 } 
	{ hidden2_38 sc_in sc_lv 16 signal 40 } 
	{ hidden2_39 sc_in sc_lv 16 signal 41 } 
	{ hidden2_40 sc_in sc_lv 16 signal 42 } 
	{ hidden2_41 sc_in sc_lv 16 signal 43 } 
	{ hidden2_42 sc_in sc_lv 16 signal 44 } 
	{ hidden2_43 sc_in sc_lv 16 signal 45 } 
	{ hidden2_44 sc_in sc_lv 16 signal 46 } 
	{ hidden2_45 sc_in sc_lv 16 signal 47 } 
	{ hidden2_46 sc_in sc_lv 16 signal 48 } 
	{ hidden2_47 sc_in sc_lv 16 signal 49 } 
	{ hidden2_48 sc_in sc_lv 16 signal 50 } 
	{ hidden2_49 sc_in sc_lv 16 signal 51 } 
	{ hidden2_50 sc_in sc_lv 16 signal 52 } 
	{ hidden2_51 sc_in sc_lv 16 signal 53 } 
	{ hidden2_52 sc_in sc_lv 16 signal 54 } 
	{ hidden2_53 sc_in sc_lv 16 signal 55 } 
	{ hidden2_54 sc_in sc_lv 16 signal 56 } 
	{ hidden2_55 sc_in sc_lv 16 signal 57 } 
	{ hidden2_56 sc_in sc_lv 16 signal 58 } 
	{ hidden2_57 sc_in sc_lv 16 signal 59 } 
	{ hidden2_58 sc_in sc_lv 16 signal 60 } 
	{ hidden2_59 sc_in sc_lv 16 signal 61 } 
	{ hidden2_60 sc_in sc_lv 16 signal 62 } 
	{ hidden2_61 sc_in sc_lv 16 signal 63 } 
	{ hidden2_62 sc_in sc_lv 16 signal 64 } 
	{ hidden2_63 sc_in sc_lv 16 signal 65 } 
	{ sum_26_out sc_out sc_lv 24 signal 66 } 
	{ sum_26_out_ap_vld sc_out sc_logic 1 outvld 66 } 
	{ dense_2_weights_address0 sc_out sc_lv 11 signal 67 } 
	{ dense_2_weights_ce0 sc_out sc_logic 1 signal 67 } 
	{ dense_2_weights_q0 sc_in sc_lv 32 signal 67 } 
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
 	{ "name": "sext_ln71_5", "direction": "in", "datatype": "sc_lv", "bitwidth":24, "type": "signal", "bundle":{"name": "sext_ln71_5", "role": "default" }} , 
 	{ "name": "tmp_112", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "tmp_112", "role": "default" }} , 
 	{ "name": "hidden2", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2", "role": "default" }} , 
 	{ "name": "hidden2_1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_1", "role": "default" }} , 
 	{ "name": "hidden2_2", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_2", "role": "default" }} , 
 	{ "name": "hidden2_3", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_3", "role": "default" }} , 
 	{ "name": "hidden2_4", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_4", "role": "default" }} , 
 	{ "name": "hidden2_5", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_5", "role": "default" }} , 
 	{ "name": "hidden2_6", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_6", "role": "default" }} , 
 	{ "name": "hidden2_7", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_7", "role": "default" }} , 
 	{ "name": "hidden2_8", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_8", "role": "default" }} , 
 	{ "name": "hidden2_9", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_9", "role": "default" }} , 
 	{ "name": "hidden2_10", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_10", "role": "default" }} , 
 	{ "name": "hidden2_11", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_11", "role": "default" }} , 
 	{ "name": "hidden2_12", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_12", "role": "default" }} , 
 	{ "name": "hidden2_13", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_13", "role": "default" }} , 
 	{ "name": "hidden2_14", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_14", "role": "default" }} , 
 	{ "name": "hidden2_15", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_15", "role": "default" }} , 
 	{ "name": "hidden2_16", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_16", "role": "default" }} , 
 	{ "name": "hidden2_17", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_17", "role": "default" }} , 
 	{ "name": "hidden2_18", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_18", "role": "default" }} , 
 	{ "name": "hidden2_19", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_19", "role": "default" }} , 
 	{ "name": "hidden2_20", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_20", "role": "default" }} , 
 	{ "name": "hidden2_21", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_21", "role": "default" }} , 
 	{ "name": "hidden2_22", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_22", "role": "default" }} , 
 	{ "name": "hidden2_23", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_23", "role": "default" }} , 
 	{ "name": "hidden2_24", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_24", "role": "default" }} , 
 	{ "name": "hidden2_25", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_25", "role": "default" }} , 
 	{ "name": "hidden2_26", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_26", "role": "default" }} , 
 	{ "name": "hidden2_27", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_27", "role": "default" }} , 
 	{ "name": "hidden2_28", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_28", "role": "default" }} , 
 	{ "name": "hidden2_29", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_29", "role": "default" }} , 
 	{ "name": "hidden2_30", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_30", "role": "default" }} , 
 	{ "name": "hidden2_31", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_31", "role": "default" }} , 
 	{ "name": "hidden2_32", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_32", "role": "default" }} , 
 	{ "name": "hidden2_33", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_33", "role": "default" }} , 
 	{ "name": "hidden2_34", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_34", "role": "default" }} , 
 	{ "name": "hidden2_35", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_35", "role": "default" }} , 
 	{ "name": "hidden2_36", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_36", "role": "default" }} , 
 	{ "name": "hidden2_37", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_37", "role": "default" }} , 
 	{ "name": "hidden2_38", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_38", "role": "default" }} , 
 	{ "name": "hidden2_39", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_39", "role": "default" }} , 
 	{ "name": "hidden2_40", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_40", "role": "default" }} , 
 	{ "name": "hidden2_41", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_41", "role": "default" }} , 
 	{ "name": "hidden2_42", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_42", "role": "default" }} , 
 	{ "name": "hidden2_43", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_43", "role": "default" }} , 
 	{ "name": "hidden2_44", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_44", "role": "default" }} , 
 	{ "name": "hidden2_45", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_45", "role": "default" }} , 
 	{ "name": "hidden2_46", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_46", "role": "default" }} , 
 	{ "name": "hidden2_47", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_47", "role": "default" }} , 
 	{ "name": "hidden2_48", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_48", "role": "default" }} , 
 	{ "name": "hidden2_49", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_49", "role": "default" }} , 
 	{ "name": "hidden2_50", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_50", "role": "default" }} , 
 	{ "name": "hidden2_51", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_51", "role": "default" }} , 
 	{ "name": "hidden2_52", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_52", "role": "default" }} , 
 	{ "name": "hidden2_53", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_53", "role": "default" }} , 
 	{ "name": "hidden2_54", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_54", "role": "default" }} , 
 	{ "name": "hidden2_55", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_55", "role": "default" }} , 
 	{ "name": "hidden2_56", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_56", "role": "default" }} , 
 	{ "name": "hidden2_57", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_57", "role": "default" }} , 
 	{ "name": "hidden2_58", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_58", "role": "default" }} , 
 	{ "name": "hidden2_59", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_59", "role": "default" }} , 
 	{ "name": "hidden2_60", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_60", "role": "default" }} , 
 	{ "name": "hidden2_61", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_61", "role": "default" }} , 
 	{ "name": "hidden2_62", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_62", "role": "default" }} , 
 	{ "name": "hidden2_63", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden2_63", "role": "default" }} , 
 	{ "name": "sum_26_out", "direction": "out", "datatype": "sc_lv", "bitwidth":24, "type": "signal", "bundle":{"name": "sum_26_out", "role": "default" }} , 
 	{ "name": "sum_26_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "sum_26_out", "role": "ap_vld" }} , 
 	{ "name": "dense_2_weights_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":11, "type": "signal", "bundle":{"name": "dense_2_weights", "role": "address0" }} , 
 	{ "name": "dense_2_weights_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "dense_2_weights", "role": "ce0" }} , 
 	{ "name": "dense_2_weights_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "dense_2_weights", "role": "q0" }} , 
 	{ "name": "grp_fu_13213_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_13213_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "grp_fu_13213_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_13213_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_73_67",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "74", "EstimateLatencyMax" : "74",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln71_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_112", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_7", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_9", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_10", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_12", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_13", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_14", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_15", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_16", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_17", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_18", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_19", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_20", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_21", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_22", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_23", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_24", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_25", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_26", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_27", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_28", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_29", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_30", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_31", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_32", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_33", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_34", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_35", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_36", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_37", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_38", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_39", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_40", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_41", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_42", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_43", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_44", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_45", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_46", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_47", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_48", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_49", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_50", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_51", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_52", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_53", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_54", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_55", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_56", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_57", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_58", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_59", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_60", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_61", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_62", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden2_63", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_26_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_2_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_73_6", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_129_6_16_1_1_U658", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U659", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mac_muladd_16s_16s_32ns_32_4_1_U660", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	ecg_classify_trained_Pipeline_VITIS_LOOP_73_67 {
		sext_ln71_5 {Type I LastRead 0 FirstWrite -1}
		tmp_112 {Type I LastRead 0 FirstWrite -1}
		hidden2 {Type I LastRead 0 FirstWrite -1}
		hidden2_1 {Type I LastRead 0 FirstWrite -1}
		hidden2_2 {Type I LastRead 0 FirstWrite -1}
		hidden2_3 {Type I LastRead 0 FirstWrite -1}
		hidden2_4 {Type I LastRead 0 FirstWrite -1}
		hidden2_5 {Type I LastRead 0 FirstWrite -1}
		hidden2_6 {Type I LastRead 0 FirstWrite -1}
		hidden2_7 {Type I LastRead 0 FirstWrite -1}
		hidden2_8 {Type I LastRead 0 FirstWrite -1}
		hidden2_9 {Type I LastRead 0 FirstWrite -1}
		hidden2_10 {Type I LastRead 0 FirstWrite -1}
		hidden2_11 {Type I LastRead 0 FirstWrite -1}
		hidden2_12 {Type I LastRead 0 FirstWrite -1}
		hidden2_13 {Type I LastRead 0 FirstWrite -1}
		hidden2_14 {Type I LastRead 0 FirstWrite -1}
		hidden2_15 {Type I LastRead 0 FirstWrite -1}
		hidden2_16 {Type I LastRead 0 FirstWrite -1}
		hidden2_17 {Type I LastRead 0 FirstWrite -1}
		hidden2_18 {Type I LastRead 0 FirstWrite -1}
		hidden2_19 {Type I LastRead 0 FirstWrite -1}
		hidden2_20 {Type I LastRead 0 FirstWrite -1}
		hidden2_21 {Type I LastRead 0 FirstWrite -1}
		hidden2_22 {Type I LastRead 0 FirstWrite -1}
		hidden2_23 {Type I LastRead 0 FirstWrite -1}
		hidden2_24 {Type I LastRead 0 FirstWrite -1}
		hidden2_25 {Type I LastRead 0 FirstWrite -1}
		hidden2_26 {Type I LastRead 0 FirstWrite -1}
		hidden2_27 {Type I LastRead 0 FirstWrite -1}
		hidden2_28 {Type I LastRead 0 FirstWrite -1}
		hidden2_29 {Type I LastRead 0 FirstWrite -1}
		hidden2_30 {Type I LastRead 0 FirstWrite -1}
		hidden2_31 {Type I LastRead 0 FirstWrite -1}
		hidden2_32 {Type I LastRead 0 FirstWrite -1}
		hidden2_33 {Type I LastRead 0 FirstWrite -1}
		hidden2_34 {Type I LastRead 0 FirstWrite -1}
		hidden2_35 {Type I LastRead 0 FirstWrite -1}
		hidden2_36 {Type I LastRead 0 FirstWrite -1}
		hidden2_37 {Type I LastRead 0 FirstWrite -1}
		hidden2_38 {Type I LastRead 0 FirstWrite -1}
		hidden2_39 {Type I LastRead 0 FirstWrite -1}
		hidden2_40 {Type I LastRead 0 FirstWrite -1}
		hidden2_41 {Type I LastRead 0 FirstWrite -1}
		hidden2_42 {Type I LastRead 0 FirstWrite -1}
		hidden2_43 {Type I LastRead 0 FirstWrite -1}
		hidden2_44 {Type I LastRead 0 FirstWrite -1}
		hidden2_45 {Type I LastRead 0 FirstWrite -1}
		hidden2_46 {Type I LastRead 0 FirstWrite -1}
		hidden2_47 {Type I LastRead 0 FirstWrite -1}
		hidden2_48 {Type I LastRead 0 FirstWrite -1}
		hidden2_49 {Type I LastRead 0 FirstWrite -1}
		hidden2_50 {Type I LastRead 0 FirstWrite -1}
		hidden2_51 {Type I LastRead 0 FirstWrite -1}
		hidden2_52 {Type I LastRead 0 FirstWrite -1}
		hidden2_53 {Type I LastRead 0 FirstWrite -1}
		hidden2_54 {Type I LastRead 0 FirstWrite -1}
		hidden2_55 {Type I LastRead 0 FirstWrite -1}
		hidden2_56 {Type I LastRead 0 FirstWrite -1}
		hidden2_57 {Type I LastRead 0 FirstWrite -1}
		hidden2_58 {Type I LastRead 0 FirstWrite -1}
		hidden2_59 {Type I LastRead 0 FirstWrite -1}
		hidden2_60 {Type I LastRead 0 FirstWrite -1}
		hidden2_61 {Type I LastRead 0 FirstWrite -1}
		hidden2_62 {Type I LastRead 0 FirstWrite -1}
		hidden2_63 {Type I LastRead 0 FirstWrite -1}
		sum_26_out {Type O LastRead -1 FirstWrite 8}
		dense_2_weights {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "74", "Max" : "74"}
	, {"Name" : "Interval", "Min" : "74", "Max" : "74"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	sext_ln71_5 { ap_none {  { sext_ln71_5 in_data 0 24 } } }
	tmp_112 { ap_none {  { tmp_112 in_data 0 4 } } }
	hidden2 { ap_none {  { hidden2 in_data 0 16 } } }
	hidden2_1 { ap_none {  { hidden2_1 in_data 0 16 } } }
	hidden2_2 { ap_none {  { hidden2_2 in_data 0 16 } } }
	hidden2_3 { ap_none {  { hidden2_3 in_data 0 16 } } }
	hidden2_4 { ap_none {  { hidden2_4 in_data 0 16 } } }
	hidden2_5 { ap_none {  { hidden2_5 in_data 0 16 } } }
	hidden2_6 { ap_none {  { hidden2_6 in_data 0 16 } } }
	hidden2_7 { ap_none {  { hidden2_7 in_data 0 16 } } }
	hidden2_8 { ap_none {  { hidden2_8 in_data 0 16 } } }
	hidden2_9 { ap_none {  { hidden2_9 in_data 0 16 } } }
	hidden2_10 { ap_none {  { hidden2_10 in_data 0 16 } } }
	hidden2_11 { ap_none {  { hidden2_11 in_data 0 16 } } }
	hidden2_12 { ap_none {  { hidden2_12 in_data 0 16 } } }
	hidden2_13 { ap_none {  { hidden2_13 in_data 0 16 } } }
	hidden2_14 { ap_none {  { hidden2_14 in_data 0 16 } } }
	hidden2_15 { ap_none {  { hidden2_15 in_data 0 16 } } }
	hidden2_16 { ap_none {  { hidden2_16 in_data 0 16 } } }
	hidden2_17 { ap_none {  { hidden2_17 in_data 0 16 } } }
	hidden2_18 { ap_none {  { hidden2_18 in_data 0 16 } } }
	hidden2_19 { ap_none {  { hidden2_19 in_data 0 16 } } }
	hidden2_20 { ap_none {  { hidden2_20 in_data 0 16 } } }
	hidden2_21 { ap_none {  { hidden2_21 in_data 0 16 } } }
	hidden2_22 { ap_none {  { hidden2_22 in_data 0 16 } } }
	hidden2_23 { ap_none {  { hidden2_23 in_data 0 16 } } }
	hidden2_24 { ap_none {  { hidden2_24 in_data 0 16 } } }
	hidden2_25 { ap_none {  { hidden2_25 in_data 0 16 } } }
	hidden2_26 { ap_none {  { hidden2_26 in_data 0 16 } } }
	hidden2_27 { ap_none {  { hidden2_27 in_data 0 16 } } }
	hidden2_28 { ap_none {  { hidden2_28 in_data 0 16 } } }
	hidden2_29 { ap_none {  { hidden2_29 in_data 0 16 } } }
	hidden2_30 { ap_none {  { hidden2_30 in_data 0 16 } } }
	hidden2_31 { ap_none {  { hidden2_31 in_data 0 16 } } }
	hidden2_32 { ap_none {  { hidden2_32 in_data 0 16 } } }
	hidden2_33 { ap_none {  { hidden2_33 in_data 0 16 } } }
	hidden2_34 { ap_none {  { hidden2_34 in_data 0 16 } } }
	hidden2_35 { ap_none {  { hidden2_35 in_data 0 16 } } }
	hidden2_36 { ap_none {  { hidden2_36 in_data 0 16 } } }
	hidden2_37 { ap_none {  { hidden2_37 in_data 0 16 } } }
	hidden2_38 { ap_none {  { hidden2_38 in_data 0 16 } } }
	hidden2_39 { ap_none {  { hidden2_39 in_data 0 16 } } }
	hidden2_40 { ap_none {  { hidden2_40 in_data 0 16 } } }
	hidden2_41 { ap_none {  { hidden2_41 in_data 0 16 } } }
	hidden2_42 { ap_none {  { hidden2_42 in_data 0 16 } } }
	hidden2_43 { ap_none {  { hidden2_43 in_data 0 16 } } }
	hidden2_44 { ap_none {  { hidden2_44 in_data 0 16 } } }
	hidden2_45 { ap_none {  { hidden2_45 in_data 0 16 } } }
	hidden2_46 { ap_none {  { hidden2_46 in_data 0 16 } } }
	hidden2_47 { ap_none {  { hidden2_47 in_data 0 16 } } }
	hidden2_48 { ap_none {  { hidden2_48 in_data 0 16 } } }
	hidden2_49 { ap_none {  { hidden2_49 in_data 0 16 } } }
	hidden2_50 { ap_none {  { hidden2_50 in_data 0 16 } } }
	hidden2_51 { ap_none {  { hidden2_51 in_data 0 16 } } }
	hidden2_52 { ap_none {  { hidden2_52 in_data 0 16 } } }
	hidden2_53 { ap_none {  { hidden2_53 in_data 0 16 } } }
	hidden2_54 { ap_none {  { hidden2_54 in_data 0 16 } } }
	hidden2_55 { ap_none {  { hidden2_55 in_data 0 16 } } }
	hidden2_56 { ap_none {  { hidden2_56 in_data 0 16 } } }
	hidden2_57 { ap_none {  { hidden2_57 in_data 0 16 } } }
	hidden2_58 { ap_none {  { hidden2_58 in_data 0 16 } } }
	hidden2_59 { ap_none {  { hidden2_59 in_data 0 16 } } }
	hidden2_60 { ap_none {  { hidden2_60 in_data 0 16 } } }
	hidden2_61 { ap_none {  { hidden2_61 in_data 0 16 } } }
	hidden2_62 { ap_none {  { hidden2_62 in_data 0 16 } } }
	hidden2_63 { ap_none {  { hidden2_63 in_data 0 16 } } }
	sum_26_out { ap_vld {  { sum_26_out out_data 1 24 }  { sum_26_out_ap_vld out_vld 1 1 } } }
	dense_2_weights { ap_memory {  { dense_2_weights_address0 mem_address 1 11 }  { dense_2_weights_ce0 mem_ce 1 1 }  { dense_2_weights_q0 mem_dout 0 32 } } }
}
