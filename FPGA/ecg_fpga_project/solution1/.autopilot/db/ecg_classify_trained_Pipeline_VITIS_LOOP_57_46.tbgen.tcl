set moduleName ecg_classify_trained_Pipeline_VITIS_LOOP_57_46
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
set C_modelName {ecg_classify_trained_Pipeline_VITIS_LOOP_57_46}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
dict set ap_memory_interface_dict dense_1_weights { MEM_WIDTH 32 MEM_SIZE 32768 MASTER_TYPE BRAM_CTRL MEM_ADDRESS_MODE WORD_ADDRESS PACKAGE_IO port READ_LATENCY 1 }
set C_modelArgList {
	{ sext_ln55_11 int 24 regular  }
	{ tmp_120 int 4 regular  }
	{ hidden1 int 16 regular  }
	{ hidden1_1 int 16 regular  }
	{ hidden1_2 int 16 regular  }
	{ hidden1_3 int 16 regular  }
	{ hidden1_4 int 16 regular  }
	{ hidden1_5 int 16 regular  }
	{ hidden1_6 int 16 regular  }
	{ hidden1_7 int 16 regular  }
	{ hidden1_8 int 16 regular  }
	{ hidden1_9 int 16 regular  }
	{ hidden1_10 int 16 regular  }
	{ hidden1_11 int 16 regular  }
	{ hidden1_12 int 16 regular  }
	{ hidden1_13 int 16 regular  }
	{ hidden1_14 int 16 regular  }
	{ hidden1_15 int 16 regular  }
	{ hidden1_16 int 16 regular  }
	{ hidden1_17 int 16 regular  }
	{ hidden1_18 int 16 regular  }
	{ hidden1_19 int 16 regular  }
	{ hidden1_20 int 16 regular  }
	{ hidden1_21 int 16 regular  }
	{ hidden1_22 int 16 regular  }
	{ hidden1_23 int 16 regular  }
	{ hidden1_24 int 16 regular  }
	{ hidden1_25 int 16 regular  }
	{ hidden1_26 int 16 regular  }
	{ hidden1_27 int 16 regular  }
	{ hidden1_28 int 16 regular  }
	{ hidden1_29 int 16 regular  }
	{ hidden1_30 int 16 regular  }
	{ hidden1_31 int 16 regular  }
	{ hidden1_32 int 16 regular  }
	{ hidden1_33 int 16 regular  }
	{ hidden1_34 int 16 regular  }
	{ hidden1_35 int 16 regular  }
	{ hidden1_36 int 16 regular  }
	{ hidden1_37 int 16 regular  }
	{ hidden1_38 int 16 regular  }
	{ hidden1_39 int 16 regular  }
	{ hidden1_40 int 16 regular  }
	{ hidden1_41 int 16 regular  }
	{ hidden1_42 int 16 regular  }
	{ hidden1_43 int 16 regular  }
	{ hidden1_44 int 16 regular  }
	{ hidden1_45 int 16 regular  }
	{ hidden1_46 int 16 regular  }
	{ hidden1_47 int 16 regular  }
	{ hidden1_48 int 16 regular  }
	{ hidden1_49 int 16 regular  }
	{ hidden1_50 int 16 regular  }
	{ hidden1_51 int 16 regular  }
	{ hidden1_52 int 16 regular  }
	{ hidden1_53 int 16 regular  }
	{ hidden1_54 int 16 regular  }
	{ hidden1_55 int 16 regular  }
	{ hidden1_56 int 16 regular  }
	{ hidden1_57 int 16 regular  }
	{ hidden1_58 int 16 regular  }
	{ hidden1_59 int 16 regular  }
	{ hidden1_60 int 16 regular  }
	{ hidden1_61 int 16 regular  }
	{ hidden1_62 int 16 regular  }
	{ hidden1_63 int 16 regular  }
	{ hidden1_64 int 16 regular  }
	{ hidden1_65 int 16 regular  }
	{ hidden1_66 int 16 regular  }
	{ hidden1_67 int 16 regular  }
	{ hidden1_68 int 16 regular  }
	{ hidden1_69 int 16 regular  }
	{ hidden1_70 int 16 regular  }
	{ hidden1_71 int 16 regular  }
	{ hidden1_72 int 16 regular  }
	{ hidden1_73 int 16 regular  }
	{ hidden1_74 int 16 regular  }
	{ hidden1_75 int 16 regular  }
	{ hidden1_76 int 16 regular  }
	{ hidden1_77 int 16 regular  }
	{ hidden1_78 int 16 regular  }
	{ hidden1_79 int 16 regular  }
	{ hidden1_80 int 16 regular  }
	{ hidden1_81 int 16 regular  }
	{ hidden1_82 int 16 regular  }
	{ hidden1_83 int 16 regular  }
	{ hidden1_84 int 16 regular  }
	{ hidden1_85 int 16 regular  }
	{ hidden1_86 int 16 regular  }
	{ hidden1_87 int 16 regular  }
	{ hidden1_88 int 16 regular  }
	{ hidden1_89 int 16 regular  }
	{ hidden1_90 int 16 regular  }
	{ hidden1_91 int 16 regular  }
	{ hidden1_92 int 16 regular  }
	{ hidden1_93 int 16 regular  }
	{ hidden1_94 int 16 regular  }
	{ hidden1_95 int 16 regular  }
	{ hidden1_96 int 16 regular  }
	{ hidden1_97 int 16 regular  }
	{ hidden1_98 int 16 regular  }
	{ hidden1_99 int 16 regular  }
	{ hidden1_100 int 16 regular  }
	{ hidden1_101 int 16 regular  }
	{ hidden1_102 int 16 regular  }
	{ hidden1_103 int 16 regular  }
	{ hidden1_104 int 16 regular  }
	{ hidden1_105 int 16 regular  }
	{ hidden1_106 int 16 regular  }
	{ hidden1_107 int 16 regular  }
	{ hidden1_108 int 16 regular  }
	{ hidden1_109 int 16 regular  }
	{ hidden1_110 int 16 regular  }
	{ hidden1_111 int 16 regular  }
	{ hidden1_112 int 16 regular  }
	{ hidden1_113 int 16 regular  }
	{ hidden1_114 int 16 regular  }
	{ hidden1_115 int 16 regular  }
	{ hidden1_116 int 16 regular  }
	{ hidden1_117 int 16 regular  }
	{ hidden1_118 int 16 regular  }
	{ hidden1_119 int 16 regular  }
	{ hidden1_120 int 16 regular  }
	{ hidden1_121 int 16 regular  }
	{ hidden1_122 int 16 regular  }
	{ hidden1_123 int 16 regular  }
	{ hidden1_124 int 16 regular  }
	{ hidden1_125 int 16 regular  }
	{ hidden1_126 int 16 regular  }
	{ hidden1_127 int 16 regular  }
	{ sum_53_out int 24 regular {pointer 1}  }
	{ dense_1_weights float 32 regular {array 8192 { 1 } 1 1 } {global 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "sext_ln55_11", "interface" : "wire", "bitwidth" : 24, "direction" : "READONLY"} , 
 	{ "Name" : "tmp_120", "interface" : "wire", "bitwidth" : 4, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_1", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_2", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_3", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_4", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_5", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_6", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_7", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_8", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_9", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_10", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_11", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_12", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_13", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_14", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_15", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_16", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_17", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_18", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_19", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_20", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_21", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_22", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_23", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_24", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_25", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_26", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_27", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_28", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_29", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_30", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_31", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_32", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_33", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_34", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_35", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_36", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_37", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_38", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_39", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_40", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_41", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_42", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_43", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_44", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_45", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_46", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_47", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_48", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_49", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_50", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_51", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_52", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_53", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_54", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_55", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_56", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_57", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_58", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_59", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_60", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_61", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_62", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_63", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_64", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_65", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_66", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_67", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_68", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_69", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_70", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_71", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_72", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_73", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_74", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_75", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_76", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_77", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_78", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_79", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_80", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_81", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_82", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_83", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_84", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_85", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_86", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_87", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_88", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_89", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_90", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_91", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_92", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_93", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_94", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_95", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_96", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_97", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_98", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_99", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_100", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_101", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_102", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_103", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_104", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_105", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_106", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_107", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_108", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_109", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_110", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_111", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_112", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_113", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_114", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_115", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_116", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_117", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_118", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_119", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_120", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_121", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_122", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_123", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_124", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_125", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_126", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "hidden1_127", "interface" : "wire", "bitwidth" : 16, "direction" : "READONLY"} , 
 	{ "Name" : "sum_53_out", "interface" : "wire", "bitwidth" : 24, "direction" : "WRITEONLY"} , 
 	{ "Name" : "dense_1_weights", "interface" : "memory", "bitwidth" : 32, "direction" : "READONLY", "extern" : 0} ]}
# RTL Port declarations: 
set portNum 144
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst sc_in sc_logic 1 reset -1 active_high_sync } 
	{ ap_start sc_in sc_logic 1 start -1 } 
	{ ap_done sc_out sc_logic 1 predone -1 } 
	{ ap_idle sc_out sc_logic 1 done -1 } 
	{ ap_ready sc_out sc_logic 1 ready -1 } 
	{ sext_ln55_11 sc_in sc_lv 24 signal 0 } 
	{ tmp_120 sc_in sc_lv 4 signal 1 } 
	{ hidden1 sc_in sc_lv 16 signal 2 } 
	{ hidden1_1 sc_in sc_lv 16 signal 3 } 
	{ hidden1_2 sc_in sc_lv 16 signal 4 } 
	{ hidden1_3 sc_in sc_lv 16 signal 5 } 
	{ hidden1_4 sc_in sc_lv 16 signal 6 } 
	{ hidden1_5 sc_in sc_lv 16 signal 7 } 
	{ hidden1_6 sc_in sc_lv 16 signal 8 } 
	{ hidden1_7 sc_in sc_lv 16 signal 9 } 
	{ hidden1_8 sc_in sc_lv 16 signal 10 } 
	{ hidden1_9 sc_in sc_lv 16 signal 11 } 
	{ hidden1_10 sc_in sc_lv 16 signal 12 } 
	{ hidden1_11 sc_in sc_lv 16 signal 13 } 
	{ hidden1_12 sc_in sc_lv 16 signal 14 } 
	{ hidden1_13 sc_in sc_lv 16 signal 15 } 
	{ hidden1_14 sc_in sc_lv 16 signal 16 } 
	{ hidden1_15 sc_in sc_lv 16 signal 17 } 
	{ hidden1_16 sc_in sc_lv 16 signal 18 } 
	{ hidden1_17 sc_in sc_lv 16 signal 19 } 
	{ hidden1_18 sc_in sc_lv 16 signal 20 } 
	{ hidden1_19 sc_in sc_lv 16 signal 21 } 
	{ hidden1_20 sc_in sc_lv 16 signal 22 } 
	{ hidden1_21 sc_in sc_lv 16 signal 23 } 
	{ hidden1_22 sc_in sc_lv 16 signal 24 } 
	{ hidden1_23 sc_in sc_lv 16 signal 25 } 
	{ hidden1_24 sc_in sc_lv 16 signal 26 } 
	{ hidden1_25 sc_in sc_lv 16 signal 27 } 
	{ hidden1_26 sc_in sc_lv 16 signal 28 } 
	{ hidden1_27 sc_in sc_lv 16 signal 29 } 
	{ hidden1_28 sc_in sc_lv 16 signal 30 } 
	{ hidden1_29 sc_in sc_lv 16 signal 31 } 
	{ hidden1_30 sc_in sc_lv 16 signal 32 } 
	{ hidden1_31 sc_in sc_lv 16 signal 33 } 
	{ hidden1_32 sc_in sc_lv 16 signal 34 } 
	{ hidden1_33 sc_in sc_lv 16 signal 35 } 
	{ hidden1_34 sc_in sc_lv 16 signal 36 } 
	{ hidden1_35 sc_in sc_lv 16 signal 37 } 
	{ hidden1_36 sc_in sc_lv 16 signal 38 } 
	{ hidden1_37 sc_in sc_lv 16 signal 39 } 
	{ hidden1_38 sc_in sc_lv 16 signal 40 } 
	{ hidden1_39 sc_in sc_lv 16 signal 41 } 
	{ hidden1_40 sc_in sc_lv 16 signal 42 } 
	{ hidden1_41 sc_in sc_lv 16 signal 43 } 
	{ hidden1_42 sc_in sc_lv 16 signal 44 } 
	{ hidden1_43 sc_in sc_lv 16 signal 45 } 
	{ hidden1_44 sc_in sc_lv 16 signal 46 } 
	{ hidden1_45 sc_in sc_lv 16 signal 47 } 
	{ hidden1_46 sc_in sc_lv 16 signal 48 } 
	{ hidden1_47 sc_in sc_lv 16 signal 49 } 
	{ hidden1_48 sc_in sc_lv 16 signal 50 } 
	{ hidden1_49 sc_in sc_lv 16 signal 51 } 
	{ hidden1_50 sc_in sc_lv 16 signal 52 } 
	{ hidden1_51 sc_in sc_lv 16 signal 53 } 
	{ hidden1_52 sc_in sc_lv 16 signal 54 } 
	{ hidden1_53 sc_in sc_lv 16 signal 55 } 
	{ hidden1_54 sc_in sc_lv 16 signal 56 } 
	{ hidden1_55 sc_in sc_lv 16 signal 57 } 
	{ hidden1_56 sc_in sc_lv 16 signal 58 } 
	{ hidden1_57 sc_in sc_lv 16 signal 59 } 
	{ hidden1_58 sc_in sc_lv 16 signal 60 } 
	{ hidden1_59 sc_in sc_lv 16 signal 61 } 
	{ hidden1_60 sc_in sc_lv 16 signal 62 } 
	{ hidden1_61 sc_in sc_lv 16 signal 63 } 
	{ hidden1_62 sc_in sc_lv 16 signal 64 } 
	{ hidden1_63 sc_in sc_lv 16 signal 65 } 
	{ hidden1_64 sc_in sc_lv 16 signal 66 } 
	{ hidden1_65 sc_in sc_lv 16 signal 67 } 
	{ hidden1_66 sc_in sc_lv 16 signal 68 } 
	{ hidden1_67 sc_in sc_lv 16 signal 69 } 
	{ hidden1_68 sc_in sc_lv 16 signal 70 } 
	{ hidden1_69 sc_in sc_lv 16 signal 71 } 
	{ hidden1_70 sc_in sc_lv 16 signal 72 } 
	{ hidden1_71 sc_in sc_lv 16 signal 73 } 
	{ hidden1_72 sc_in sc_lv 16 signal 74 } 
	{ hidden1_73 sc_in sc_lv 16 signal 75 } 
	{ hidden1_74 sc_in sc_lv 16 signal 76 } 
	{ hidden1_75 sc_in sc_lv 16 signal 77 } 
	{ hidden1_76 sc_in sc_lv 16 signal 78 } 
	{ hidden1_77 sc_in sc_lv 16 signal 79 } 
	{ hidden1_78 sc_in sc_lv 16 signal 80 } 
	{ hidden1_79 sc_in sc_lv 16 signal 81 } 
	{ hidden1_80 sc_in sc_lv 16 signal 82 } 
	{ hidden1_81 sc_in sc_lv 16 signal 83 } 
	{ hidden1_82 sc_in sc_lv 16 signal 84 } 
	{ hidden1_83 sc_in sc_lv 16 signal 85 } 
	{ hidden1_84 sc_in sc_lv 16 signal 86 } 
	{ hidden1_85 sc_in sc_lv 16 signal 87 } 
	{ hidden1_86 sc_in sc_lv 16 signal 88 } 
	{ hidden1_87 sc_in sc_lv 16 signal 89 } 
	{ hidden1_88 sc_in sc_lv 16 signal 90 } 
	{ hidden1_89 sc_in sc_lv 16 signal 91 } 
	{ hidden1_90 sc_in sc_lv 16 signal 92 } 
	{ hidden1_91 sc_in sc_lv 16 signal 93 } 
	{ hidden1_92 sc_in sc_lv 16 signal 94 } 
	{ hidden1_93 sc_in sc_lv 16 signal 95 } 
	{ hidden1_94 sc_in sc_lv 16 signal 96 } 
	{ hidden1_95 sc_in sc_lv 16 signal 97 } 
	{ hidden1_96 sc_in sc_lv 16 signal 98 } 
	{ hidden1_97 sc_in sc_lv 16 signal 99 } 
	{ hidden1_98 sc_in sc_lv 16 signal 100 } 
	{ hidden1_99 sc_in sc_lv 16 signal 101 } 
	{ hidden1_100 sc_in sc_lv 16 signal 102 } 
	{ hidden1_101 sc_in sc_lv 16 signal 103 } 
	{ hidden1_102 sc_in sc_lv 16 signal 104 } 
	{ hidden1_103 sc_in sc_lv 16 signal 105 } 
	{ hidden1_104 sc_in sc_lv 16 signal 106 } 
	{ hidden1_105 sc_in sc_lv 16 signal 107 } 
	{ hidden1_106 sc_in sc_lv 16 signal 108 } 
	{ hidden1_107 sc_in sc_lv 16 signal 109 } 
	{ hidden1_108 sc_in sc_lv 16 signal 110 } 
	{ hidden1_109 sc_in sc_lv 16 signal 111 } 
	{ hidden1_110 sc_in sc_lv 16 signal 112 } 
	{ hidden1_111 sc_in sc_lv 16 signal 113 } 
	{ hidden1_112 sc_in sc_lv 16 signal 114 } 
	{ hidden1_113 sc_in sc_lv 16 signal 115 } 
	{ hidden1_114 sc_in sc_lv 16 signal 116 } 
	{ hidden1_115 sc_in sc_lv 16 signal 117 } 
	{ hidden1_116 sc_in sc_lv 16 signal 118 } 
	{ hidden1_117 sc_in sc_lv 16 signal 119 } 
	{ hidden1_118 sc_in sc_lv 16 signal 120 } 
	{ hidden1_119 sc_in sc_lv 16 signal 121 } 
	{ hidden1_120 sc_in sc_lv 16 signal 122 } 
	{ hidden1_121 sc_in sc_lv 16 signal 123 } 
	{ hidden1_122 sc_in sc_lv 16 signal 124 } 
	{ hidden1_123 sc_in sc_lv 16 signal 125 } 
	{ hidden1_124 sc_in sc_lv 16 signal 126 } 
	{ hidden1_125 sc_in sc_lv 16 signal 127 } 
	{ hidden1_126 sc_in sc_lv 16 signal 128 } 
	{ hidden1_127 sc_in sc_lv 16 signal 129 } 
	{ sum_53_out sc_out sc_lv 24 signal 130 } 
	{ sum_53_out_ap_vld sc_out sc_logic 1 outvld 130 } 
	{ dense_1_weights_address0 sc_out sc_lv 13 signal 131 } 
	{ dense_1_weights_ce0 sc_out sc_logic 1 signal 131 } 
	{ dense_1_weights_q0 sc_in sc_lv 32 signal 131 } 
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
 	{ "name": "sext_ln55_11", "direction": "in", "datatype": "sc_lv", "bitwidth":24, "type": "signal", "bundle":{"name": "sext_ln55_11", "role": "default" }} , 
 	{ "name": "tmp_120", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "tmp_120", "role": "default" }} , 
 	{ "name": "hidden1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1", "role": "default" }} , 
 	{ "name": "hidden1_1", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_1", "role": "default" }} , 
 	{ "name": "hidden1_2", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_2", "role": "default" }} , 
 	{ "name": "hidden1_3", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_3", "role": "default" }} , 
 	{ "name": "hidden1_4", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_4", "role": "default" }} , 
 	{ "name": "hidden1_5", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_5", "role": "default" }} , 
 	{ "name": "hidden1_6", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_6", "role": "default" }} , 
 	{ "name": "hidden1_7", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_7", "role": "default" }} , 
 	{ "name": "hidden1_8", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_8", "role": "default" }} , 
 	{ "name": "hidden1_9", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_9", "role": "default" }} , 
 	{ "name": "hidden1_10", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_10", "role": "default" }} , 
 	{ "name": "hidden1_11", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_11", "role": "default" }} , 
 	{ "name": "hidden1_12", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_12", "role": "default" }} , 
 	{ "name": "hidden1_13", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_13", "role": "default" }} , 
 	{ "name": "hidden1_14", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_14", "role": "default" }} , 
 	{ "name": "hidden1_15", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_15", "role": "default" }} , 
 	{ "name": "hidden1_16", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_16", "role": "default" }} , 
 	{ "name": "hidden1_17", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_17", "role": "default" }} , 
 	{ "name": "hidden1_18", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_18", "role": "default" }} , 
 	{ "name": "hidden1_19", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_19", "role": "default" }} , 
 	{ "name": "hidden1_20", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_20", "role": "default" }} , 
 	{ "name": "hidden1_21", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_21", "role": "default" }} , 
 	{ "name": "hidden1_22", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_22", "role": "default" }} , 
 	{ "name": "hidden1_23", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_23", "role": "default" }} , 
 	{ "name": "hidden1_24", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_24", "role": "default" }} , 
 	{ "name": "hidden1_25", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_25", "role": "default" }} , 
 	{ "name": "hidden1_26", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_26", "role": "default" }} , 
 	{ "name": "hidden1_27", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_27", "role": "default" }} , 
 	{ "name": "hidden1_28", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_28", "role": "default" }} , 
 	{ "name": "hidden1_29", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_29", "role": "default" }} , 
 	{ "name": "hidden1_30", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_30", "role": "default" }} , 
 	{ "name": "hidden1_31", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_31", "role": "default" }} , 
 	{ "name": "hidden1_32", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_32", "role": "default" }} , 
 	{ "name": "hidden1_33", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_33", "role": "default" }} , 
 	{ "name": "hidden1_34", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_34", "role": "default" }} , 
 	{ "name": "hidden1_35", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_35", "role": "default" }} , 
 	{ "name": "hidden1_36", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_36", "role": "default" }} , 
 	{ "name": "hidden1_37", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_37", "role": "default" }} , 
 	{ "name": "hidden1_38", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_38", "role": "default" }} , 
 	{ "name": "hidden1_39", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_39", "role": "default" }} , 
 	{ "name": "hidden1_40", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_40", "role": "default" }} , 
 	{ "name": "hidden1_41", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_41", "role": "default" }} , 
 	{ "name": "hidden1_42", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_42", "role": "default" }} , 
 	{ "name": "hidden1_43", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_43", "role": "default" }} , 
 	{ "name": "hidden1_44", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_44", "role": "default" }} , 
 	{ "name": "hidden1_45", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_45", "role": "default" }} , 
 	{ "name": "hidden1_46", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_46", "role": "default" }} , 
 	{ "name": "hidden1_47", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_47", "role": "default" }} , 
 	{ "name": "hidden1_48", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_48", "role": "default" }} , 
 	{ "name": "hidden1_49", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_49", "role": "default" }} , 
 	{ "name": "hidden1_50", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_50", "role": "default" }} , 
 	{ "name": "hidden1_51", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_51", "role": "default" }} , 
 	{ "name": "hidden1_52", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_52", "role": "default" }} , 
 	{ "name": "hidden1_53", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_53", "role": "default" }} , 
 	{ "name": "hidden1_54", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_54", "role": "default" }} , 
 	{ "name": "hidden1_55", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_55", "role": "default" }} , 
 	{ "name": "hidden1_56", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_56", "role": "default" }} , 
 	{ "name": "hidden1_57", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_57", "role": "default" }} , 
 	{ "name": "hidden1_58", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_58", "role": "default" }} , 
 	{ "name": "hidden1_59", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_59", "role": "default" }} , 
 	{ "name": "hidden1_60", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_60", "role": "default" }} , 
 	{ "name": "hidden1_61", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_61", "role": "default" }} , 
 	{ "name": "hidden1_62", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_62", "role": "default" }} , 
 	{ "name": "hidden1_63", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_63", "role": "default" }} , 
 	{ "name": "hidden1_64", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_64", "role": "default" }} , 
 	{ "name": "hidden1_65", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_65", "role": "default" }} , 
 	{ "name": "hidden1_66", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_66", "role": "default" }} , 
 	{ "name": "hidden1_67", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_67", "role": "default" }} , 
 	{ "name": "hidden1_68", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_68", "role": "default" }} , 
 	{ "name": "hidden1_69", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_69", "role": "default" }} , 
 	{ "name": "hidden1_70", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_70", "role": "default" }} , 
 	{ "name": "hidden1_71", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_71", "role": "default" }} , 
 	{ "name": "hidden1_72", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_72", "role": "default" }} , 
 	{ "name": "hidden1_73", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_73", "role": "default" }} , 
 	{ "name": "hidden1_74", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_74", "role": "default" }} , 
 	{ "name": "hidden1_75", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_75", "role": "default" }} , 
 	{ "name": "hidden1_76", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_76", "role": "default" }} , 
 	{ "name": "hidden1_77", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_77", "role": "default" }} , 
 	{ "name": "hidden1_78", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_78", "role": "default" }} , 
 	{ "name": "hidden1_79", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_79", "role": "default" }} , 
 	{ "name": "hidden1_80", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_80", "role": "default" }} , 
 	{ "name": "hidden1_81", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_81", "role": "default" }} , 
 	{ "name": "hidden1_82", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_82", "role": "default" }} , 
 	{ "name": "hidden1_83", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_83", "role": "default" }} , 
 	{ "name": "hidden1_84", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_84", "role": "default" }} , 
 	{ "name": "hidden1_85", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_85", "role": "default" }} , 
 	{ "name": "hidden1_86", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_86", "role": "default" }} , 
 	{ "name": "hidden1_87", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_87", "role": "default" }} , 
 	{ "name": "hidden1_88", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_88", "role": "default" }} , 
 	{ "name": "hidden1_89", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_89", "role": "default" }} , 
 	{ "name": "hidden1_90", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_90", "role": "default" }} , 
 	{ "name": "hidden1_91", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_91", "role": "default" }} , 
 	{ "name": "hidden1_92", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_92", "role": "default" }} , 
 	{ "name": "hidden1_93", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_93", "role": "default" }} , 
 	{ "name": "hidden1_94", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_94", "role": "default" }} , 
 	{ "name": "hidden1_95", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_95", "role": "default" }} , 
 	{ "name": "hidden1_96", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_96", "role": "default" }} , 
 	{ "name": "hidden1_97", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_97", "role": "default" }} , 
 	{ "name": "hidden1_98", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_98", "role": "default" }} , 
 	{ "name": "hidden1_99", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_99", "role": "default" }} , 
 	{ "name": "hidden1_100", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_100", "role": "default" }} , 
 	{ "name": "hidden1_101", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_101", "role": "default" }} , 
 	{ "name": "hidden1_102", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_102", "role": "default" }} , 
 	{ "name": "hidden1_103", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_103", "role": "default" }} , 
 	{ "name": "hidden1_104", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_104", "role": "default" }} , 
 	{ "name": "hidden1_105", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_105", "role": "default" }} , 
 	{ "name": "hidden1_106", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_106", "role": "default" }} , 
 	{ "name": "hidden1_107", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_107", "role": "default" }} , 
 	{ "name": "hidden1_108", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_108", "role": "default" }} , 
 	{ "name": "hidden1_109", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_109", "role": "default" }} , 
 	{ "name": "hidden1_110", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_110", "role": "default" }} , 
 	{ "name": "hidden1_111", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_111", "role": "default" }} , 
 	{ "name": "hidden1_112", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_112", "role": "default" }} , 
 	{ "name": "hidden1_113", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_113", "role": "default" }} , 
 	{ "name": "hidden1_114", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_114", "role": "default" }} , 
 	{ "name": "hidden1_115", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_115", "role": "default" }} , 
 	{ "name": "hidden1_116", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_116", "role": "default" }} , 
 	{ "name": "hidden1_117", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_117", "role": "default" }} , 
 	{ "name": "hidden1_118", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_118", "role": "default" }} , 
 	{ "name": "hidden1_119", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_119", "role": "default" }} , 
 	{ "name": "hidden1_120", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_120", "role": "default" }} , 
 	{ "name": "hidden1_121", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_121", "role": "default" }} , 
 	{ "name": "hidden1_122", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_122", "role": "default" }} , 
 	{ "name": "hidden1_123", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_123", "role": "default" }} , 
 	{ "name": "hidden1_124", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_124", "role": "default" }} , 
 	{ "name": "hidden1_125", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_125", "role": "default" }} , 
 	{ "name": "hidden1_126", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_126", "role": "default" }} , 
 	{ "name": "hidden1_127", "direction": "in", "datatype": "sc_lv", "bitwidth":16, "type": "signal", "bundle":{"name": "hidden1_127", "role": "default" }} , 
 	{ "name": "sum_53_out", "direction": "out", "datatype": "sc_lv", "bitwidth":24, "type": "signal", "bundle":{"name": "sum_53_out", "role": "default" }} , 
 	{ "name": "sum_53_out_ap_vld", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "outvld", "bundle":{"name": "sum_53_out", "role": "ap_vld" }} , 
 	{ "name": "dense_1_weights_address0", "direction": "out", "datatype": "sc_lv", "bitwidth":13, "type": "signal", "bundle":{"name": "dense_1_weights", "role": "address0" }} , 
 	{ "name": "dense_1_weights_ce0", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "dense_1_weights", "role": "ce0" }} , 
 	{ "name": "dense_1_weights_q0", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "dense_1_weights", "role": "q0" }} , 
 	{ "name": "grp_fu_13213_p_din0", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "grp_fu_13213_p_din0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_dout0", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "grp_fu_13213_p_dout0", "role": "default" }} , 
 	{ "name": "grp_fu_13213_p_ce", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "grp_fu_13213_p_ce", "role": "default" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_57_46",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "138", "EstimateLatencyMax" : "138",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln55_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_120", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_3", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_4", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_6", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_7", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_9", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_10", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_12", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_13", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_14", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_15", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_16", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_17", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_18", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_19", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_20", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_21", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_22", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_23", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_24", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_25", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_26", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_27", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_28", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_29", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_30", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_31", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_32", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_33", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_34", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_35", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_36", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_37", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_38", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_39", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_40", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_41", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_42", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_43", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_44", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_45", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_46", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_47", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_48", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_49", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_50", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_51", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_52", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_53", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_54", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_55", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_56", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_57", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_58", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_59", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_60", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_61", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_62", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_63", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_64", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_65", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_66", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_67", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_68", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_69", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_70", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_71", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_72", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_73", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_74", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_75", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_76", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_77", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_78", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_79", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_80", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_81", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_82", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_83", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_84", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_85", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_86", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_87", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_88", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_89", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_90", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_91", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_92", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_93", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_94", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_95", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_96", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_97", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_98", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_99", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_100", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_101", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_102", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_103", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_104", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_105", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_106", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_107", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_108", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_109", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_110", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_111", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_112", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_113", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_114", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_115", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_116", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_117", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_118", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_119", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_120", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_121", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_122", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_123", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_124", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_125", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_126", "Type" : "None", "Direction" : "I"},
			{"Name" : "hidden1_127", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_53_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_1_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_57_4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_257_7_16_1_1_U449", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U450", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.mac_muladd_16s_16s_32ns_32_4_1_U451", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.flow_control_loop_pipe_sequential_init_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	ecg_classify_trained_Pipeline_VITIS_LOOP_57_46 {
		sext_ln55_11 {Type I LastRead 0 FirstWrite -1}
		tmp_120 {Type I LastRead 0 FirstWrite -1}
		hidden1 {Type I LastRead 0 FirstWrite -1}
		hidden1_1 {Type I LastRead 0 FirstWrite -1}
		hidden1_2 {Type I LastRead 0 FirstWrite -1}
		hidden1_3 {Type I LastRead 0 FirstWrite -1}
		hidden1_4 {Type I LastRead 0 FirstWrite -1}
		hidden1_5 {Type I LastRead 0 FirstWrite -1}
		hidden1_6 {Type I LastRead 0 FirstWrite -1}
		hidden1_7 {Type I LastRead 0 FirstWrite -1}
		hidden1_8 {Type I LastRead 0 FirstWrite -1}
		hidden1_9 {Type I LastRead 0 FirstWrite -1}
		hidden1_10 {Type I LastRead 0 FirstWrite -1}
		hidden1_11 {Type I LastRead 0 FirstWrite -1}
		hidden1_12 {Type I LastRead 0 FirstWrite -1}
		hidden1_13 {Type I LastRead 0 FirstWrite -1}
		hidden1_14 {Type I LastRead 0 FirstWrite -1}
		hidden1_15 {Type I LastRead 0 FirstWrite -1}
		hidden1_16 {Type I LastRead 0 FirstWrite -1}
		hidden1_17 {Type I LastRead 0 FirstWrite -1}
		hidden1_18 {Type I LastRead 0 FirstWrite -1}
		hidden1_19 {Type I LastRead 0 FirstWrite -1}
		hidden1_20 {Type I LastRead 0 FirstWrite -1}
		hidden1_21 {Type I LastRead 0 FirstWrite -1}
		hidden1_22 {Type I LastRead 0 FirstWrite -1}
		hidden1_23 {Type I LastRead 0 FirstWrite -1}
		hidden1_24 {Type I LastRead 0 FirstWrite -1}
		hidden1_25 {Type I LastRead 0 FirstWrite -1}
		hidden1_26 {Type I LastRead 0 FirstWrite -1}
		hidden1_27 {Type I LastRead 0 FirstWrite -1}
		hidden1_28 {Type I LastRead 0 FirstWrite -1}
		hidden1_29 {Type I LastRead 0 FirstWrite -1}
		hidden1_30 {Type I LastRead 0 FirstWrite -1}
		hidden1_31 {Type I LastRead 0 FirstWrite -1}
		hidden1_32 {Type I LastRead 0 FirstWrite -1}
		hidden1_33 {Type I LastRead 0 FirstWrite -1}
		hidden1_34 {Type I LastRead 0 FirstWrite -1}
		hidden1_35 {Type I LastRead 0 FirstWrite -1}
		hidden1_36 {Type I LastRead 0 FirstWrite -1}
		hidden1_37 {Type I LastRead 0 FirstWrite -1}
		hidden1_38 {Type I LastRead 0 FirstWrite -1}
		hidden1_39 {Type I LastRead 0 FirstWrite -1}
		hidden1_40 {Type I LastRead 0 FirstWrite -1}
		hidden1_41 {Type I LastRead 0 FirstWrite -1}
		hidden1_42 {Type I LastRead 0 FirstWrite -1}
		hidden1_43 {Type I LastRead 0 FirstWrite -1}
		hidden1_44 {Type I LastRead 0 FirstWrite -1}
		hidden1_45 {Type I LastRead 0 FirstWrite -1}
		hidden1_46 {Type I LastRead 0 FirstWrite -1}
		hidden1_47 {Type I LastRead 0 FirstWrite -1}
		hidden1_48 {Type I LastRead 0 FirstWrite -1}
		hidden1_49 {Type I LastRead 0 FirstWrite -1}
		hidden1_50 {Type I LastRead 0 FirstWrite -1}
		hidden1_51 {Type I LastRead 0 FirstWrite -1}
		hidden1_52 {Type I LastRead 0 FirstWrite -1}
		hidden1_53 {Type I LastRead 0 FirstWrite -1}
		hidden1_54 {Type I LastRead 0 FirstWrite -1}
		hidden1_55 {Type I LastRead 0 FirstWrite -1}
		hidden1_56 {Type I LastRead 0 FirstWrite -1}
		hidden1_57 {Type I LastRead 0 FirstWrite -1}
		hidden1_58 {Type I LastRead 0 FirstWrite -1}
		hidden1_59 {Type I LastRead 0 FirstWrite -1}
		hidden1_60 {Type I LastRead 0 FirstWrite -1}
		hidden1_61 {Type I LastRead 0 FirstWrite -1}
		hidden1_62 {Type I LastRead 0 FirstWrite -1}
		hidden1_63 {Type I LastRead 0 FirstWrite -1}
		hidden1_64 {Type I LastRead 0 FirstWrite -1}
		hidden1_65 {Type I LastRead 0 FirstWrite -1}
		hidden1_66 {Type I LastRead 0 FirstWrite -1}
		hidden1_67 {Type I LastRead 0 FirstWrite -1}
		hidden1_68 {Type I LastRead 0 FirstWrite -1}
		hidden1_69 {Type I LastRead 0 FirstWrite -1}
		hidden1_70 {Type I LastRead 0 FirstWrite -1}
		hidden1_71 {Type I LastRead 0 FirstWrite -1}
		hidden1_72 {Type I LastRead 0 FirstWrite -1}
		hidden1_73 {Type I LastRead 0 FirstWrite -1}
		hidden1_74 {Type I LastRead 0 FirstWrite -1}
		hidden1_75 {Type I LastRead 0 FirstWrite -1}
		hidden1_76 {Type I LastRead 0 FirstWrite -1}
		hidden1_77 {Type I LastRead 0 FirstWrite -1}
		hidden1_78 {Type I LastRead 0 FirstWrite -1}
		hidden1_79 {Type I LastRead 0 FirstWrite -1}
		hidden1_80 {Type I LastRead 0 FirstWrite -1}
		hidden1_81 {Type I LastRead 0 FirstWrite -1}
		hidden1_82 {Type I LastRead 0 FirstWrite -1}
		hidden1_83 {Type I LastRead 0 FirstWrite -1}
		hidden1_84 {Type I LastRead 0 FirstWrite -1}
		hidden1_85 {Type I LastRead 0 FirstWrite -1}
		hidden1_86 {Type I LastRead 0 FirstWrite -1}
		hidden1_87 {Type I LastRead 0 FirstWrite -1}
		hidden1_88 {Type I LastRead 0 FirstWrite -1}
		hidden1_89 {Type I LastRead 0 FirstWrite -1}
		hidden1_90 {Type I LastRead 0 FirstWrite -1}
		hidden1_91 {Type I LastRead 0 FirstWrite -1}
		hidden1_92 {Type I LastRead 0 FirstWrite -1}
		hidden1_93 {Type I LastRead 0 FirstWrite -1}
		hidden1_94 {Type I LastRead 0 FirstWrite -1}
		hidden1_95 {Type I LastRead 0 FirstWrite -1}
		hidden1_96 {Type I LastRead 0 FirstWrite -1}
		hidden1_97 {Type I LastRead 0 FirstWrite -1}
		hidden1_98 {Type I LastRead 0 FirstWrite -1}
		hidden1_99 {Type I LastRead 0 FirstWrite -1}
		hidden1_100 {Type I LastRead 0 FirstWrite -1}
		hidden1_101 {Type I LastRead 0 FirstWrite -1}
		hidden1_102 {Type I LastRead 0 FirstWrite -1}
		hidden1_103 {Type I LastRead 0 FirstWrite -1}
		hidden1_104 {Type I LastRead 0 FirstWrite -1}
		hidden1_105 {Type I LastRead 0 FirstWrite -1}
		hidden1_106 {Type I LastRead 0 FirstWrite -1}
		hidden1_107 {Type I LastRead 0 FirstWrite -1}
		hidden1_108 {Type I LastRead 0 FirstWrite -1}
		hidden1_109 {Type I LastRead 0 FirstWrite -1}
		hidden1_110 {Type I LastRead 0 FirstWrite -1}
		hidden1_111 {Type I LastRead 0 FirstWrite -1}
		hidden1_112 {Type I LastRead 0 FirstWrite -1}
		hidden1_113 {Type I LastRead 0 FirstWrite -1}
		hidden1_114 {Type I LastRead 0 FirstWrite -1}
		hidden1_115 {Type I LastRead 0 FirstWrite -1}
		hidden1_116 {Type I LastRead 0 FirstWrite -1}
		hidden1_117 {Type I LastRead 0 FirstWrite -1}
		hidden1_118 {Type I LastRead 0 FirstWrite -1}
		hidden1_119 {Type I LastRead 0 FirstWrite -1}
		hidden1_120 {Type I LastRead 0 FirstWrite -1}
		hidden1_121 {Type I LastRead 0 FirstWrite -1}
		hidden1_122 {Type I LastRead 0 FirstWrite -1}
		hidden1_123 {Type I LastRead 0 FirstWrite -1}
		hidden1_124 {Type I LastRead 0 FirstWrite -1}
		hidden1_125 {Type I LastRead 0 FirstWrite -1}
		hidden1_126 {Type I LastRead 0 FirstWrite -1}
		hidden1_127 {Type I LastRead 0 FirstWrite -1}
		sum_53_out {Type O LastRead -1 FirstWrite 8}
		dense_1_weights {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "138", "Max" : "138"}
	, {"Name" : "Interval", "Min" : "138", "Max" : "138"}
]}

set PipelineEnableSignalInfo {[
	{"Pipeline" : "0", "EnableSignal" : "ap_enable_pp0"}
]}

set Spec2ImplPortList { 
	sext_ln55_11 { ap_none {  { sext_ln55_11 in_data 0 24 } } }
	tmp_120 { ap_none {  { tmp_120 in_data 0 4 } } }
	hidden1 { ap_none {  { hidden1 in_data 0 16 } } }
	hidden1_1 { ap_none {  { hidden1_1 in_data 0 16 } } }
	hidden1_2 { ap_none {  { hidden1_2 in_data 0 16 } } }
	hidden1_3 { ap_none {  { hidden1_3 in_data 0 16 } } }
	hidden1_4 { ap_none {  { hidden1_4 in_data 0 16 } } }
	hidden1_5 { ap_none {  { hidden1_5 in_data 0 16 } } }
	hidden1_6 { ap_none {  { hidden1_6 in_data 0 16 } } }
	hidden1_7 { ap_none {  { hidden1_7 in_data 0 16 } } }
	hidden1_8 { ap_none {  { hidden1_8 in_data 0 16 } } }
	hidden1_9 { ap_none {  { hidden1_9 in_data 0 16 } } }
	hidden1_10 { ap_none {  { hidden1_10 in_data 0 16 } } }
	hidden1_11 { ap_none {  { hidden1_11 in_data 0 16 } } }
	hidden1_12 { ap_none {  { hidden1_12 in_data 0 16 } } }
	hidden1_13 { ap_none {  { hidden1_13 in_data 0 16 } } }
	hidden1_14 { ap_none {  { hidden1_14 in_data 0 16 } } }
	hidden1_15 { ap_none {  { hidden1_15 in_data 0 16 } } }
	hidden1_16 { ap_none {  { hidden1_16 in_data 0 16 } } }
	hidden1_17 { ap_none {  { hidden1_17 in_data 0 16 } } }
	hidden1_18 { ap_none {  { hidden1_18 in_data 0 16 } } }
	hidden1_19 { ap_none {  { hidden1_19 in_data 0 16 } } }
	hidden1_20 { ap_none {  { hidden1_20 in_data 0 16 } } }
	hidden1_21 { ap_none {  { hidden1_21 in_data 0 16 } } }
	hidden1_22 { ap_none {  { hidden1_22 in_data 0 16 } } }
	hidden1_23 { ap_none {  { hidden1_23 in_data 0 16 } } }
	hidden1_24 { ap_none {  { hidden1_24 in_data 0 16 } } }
	hidden1_25 { ap_none {  { hidden1_25 in_data 0 16 } } }
	hidden1_26 { ap_none {  { hidden1_26 in_data 0 16 } } }
	hidden1_27 { ap_none {  { hidden1_27 in_data 0 16 } } }
	hidden1_28 { ap_none {  { hidden1_28 in_data 0 16 } } }
	hidden1_29 { ap_none {  { hidden1_29 in_data 0 16 } } }
	hidden1_30 { ap_none {  { hidden1_30 in_data 0 16 } } }
	hidden1_31 { ap_none {  { hidden1_31 in_data 0 16 } } }
	hidden1_32 { ap_none {  { hidden1_32 in_data 0 16 } } }
	hidden1_33 { ap_none {  { hidden1_33 in_data 0 16 } } }
	hidden1_34 { ap_none {  { hidden1_34 in_data 0 16 } } }
	hidden1_35 { ap_none {  { hidden1_35 in_data 0 16 } } }
	hidden1_36 { ap_none {  { hidden1_36 in_data 0 16 } } }
	hidden1_37 { ap_none {  { hidden1_37 in_data 0 16 } } }
	hidden1_38 { ap_none {  { hidden1_38 in_data 0 16 } } }
	hidden1_39 { ap_none {  { hidden1_39 in_data 0 16 } } }
	hidden1_40 { ap_none {  { hidden1_40 in_data 0 16 } } }
	hidden1_41 { ap_none {  { hidden1_41 in_data 0 16 } } }
	hidden1_42 { ap_none {  { hidden1_42 in_data 0 16 } } }
	hidden1_43 { ap_none {  { hidden1_43 in_data 0 16 } } }
	hidden1_44 { ap_none {  { hidden1_44 in_data 0 16 } } }
	hidden1_45 { ap_none {  { hidden1_45 in_data 0 16 } } }
	hidden1_46 { ap_none {  { hidden1_46 in_data 0 16 } } }
	hidden1_47 { ap_none {  { hidden1_47 in_data 0 16 } } }
	hidden1_48 { ap_none {  { hidden1_48 in_data 0 16 } } }
	hidden1_49 { ap_none {  { hidden1_49 in_data 0 16 } } }
	hidden1_50 { ap_none {  { hidden1_50 in_data 0 16 } } }
	hidden1_51 { ap_none {  { hidden1_51 in_data 0 16 } } }
	hidden1_52 { ap_none {  { hidden1_52 in_data 0 16 } } }
	hidden1_53 { ap_none {  { hidden1_53 in_data 0 16 } } }
	hidden1_54 { ap_none {  { hidden1_54 in_data 0 16 } } }
	hidden1_55 { ap_none {  { hidden1_55 in_data 0 16 } } }
	hidden1_56 { ap_none {  { hidden1_56 in_data 0 16 } } }
	hidden1_57 { ap_none {  { hidden1_57 in_data 0 16 } } }
	hidden1_58 { ap_none {  { hidden1_58 in_data 0 16 } } }
	hidden1_59 { ap_none {  { hidden1_59 in_data 0 16 } } }
	hidden1_60 { ap_none {  { hidden1_60 in_data 0 16 } } }
	hidden1_61 { ap_none {  { hidden1_61 in_data 0 16 } } }
	hidden1_62 { ap_none {  { hidden1_62 in_data 0 16 } } }
	hidden1_63 { ap_none {  { hidden1_63 in_data 0 16 } } }
	hidden1_64 { ap_none {  { hidden1_64 in_data 0 16 } } }
	hidden1_65 { ap_none {  { hidden1_65 in_data 0 16 } } }
	hidden1_66 { ap_none {  { hidden1_66 in_data 0 16 } } }
	hidden1_67 { ap_none {  { hidden1_67 in_data 0 16 } } }
	hidden1_68 { ap_none {  { hidden1_68 in_data 0 16 } } }
	hidden1_69 { ap_none {  { hidden1_69 in_data 0 16 } } }
	hidden1_70 { ap_none {  { hidden1_70 in_data 0 16 } } }
	hidden1_71 { ap_none {  { hidden1_71 in_data 0 16 } } }
	hidden1_72 { ap_none {  { hidden1_72 in_data 0 16 } } }
	hidden1_73 { ap_none {  { hidden1_73 in_data 0 16 } } }
	hidden1_74 { ap_none {  { hidden1_74 in_data 0 16 } } }
	hidden1_75 { ap_none {  { hidden1_75 in_data 0 16 } } }
	hidden1_76 { ap_none {  { hidden1_76 in_data 0 16 } } }
	hidden1_77 { ap_none {  { hidden1_77 in_data 0 16 } } }
	hidden1_78 { ap_none {  { hidden1_78 in_data 0 16 } } }
	hidden1_79 { ap_none {  { hidden1_79 in_data 0 16 } } }
	hidden1_80 { ap_none {  { hidden1_80 in_data 0 16 } } }
	hidden1_81 { ap_none {  { hidden1_81 in_data 0 16 } } }
	hidden1_82 { ap_none {  { hidden1_82 in_data 0 16 } } }
	hidden1_83 { ap_none {  { hidden1_83 in_data 0 16 } } }
	hidden1_84 { ap_none {  { hidden1_84 in_data 0 16 } } }
	hidden1_85 { ap_none {  { hidden1_85 in_data 0 16 } } }
	hidden1_86 { ap_none {  { hidden1_86 in_data 0 16 } } }
	hidden1_87 { ap_none {  { hidden1_87 in_data 0 16 } } }
	hidden1_88 { ap_none {  { hidden1_88 in_data 0 16 } } }
	hidden1_89 { ap_none {  { hidden1_89 in_data 0 16 } } }
	hidden1_90 { ap_none {  { hidden1_90 in_data 0 16 } } }
	hidden1_91 { ap_none {  { hidden1_91 in_data 0 16 } } }
	hidden1_92 { ap_none {  { hidden1_92 in_data 0 16 } } }
	hidden1_93 { ap_none {  { hidden1_93 in_data 0 16 } } }
	hidden1_94 { ap_none {  { hidden1_94 in_data 0 16 } } }
	hidden1_95 { ap_none {  { hidden1_95 in_data 0 16 } } }
	hidden1_96 { ap_none {  { hidden1_96 in_data 0 16 } } }
	hidden1_97 { ap_none {  { hidden1_97 in_data 0 16 } } }
	hidden1_98 { ap_none {  { hidden1_98 in_data 0 16 } } }
	hidden1_99 { ap_none {  { hidden1_99 in_data 0 16 } } }
	hidden1_100 { ap_none {  { hidden1_100 in_data 0 16 } } }
	hidden1_101 { ap_none {  { hidden1_101 in_data 0 16 } } }
	hidden1_102 { ap_none {  { hidden1_102 in_data 0 16 } } }
	hidden1_103 { ap_none {  { hidden1_103 in_data 0 16 } } }
	hidden1_104 { ap_none {  { hidden1_104 in_data 0 16 } } }
	hidden1_105 { ap_none {  { hidden1_105 in_data 0 16 } } }
	hidden1_106 { ap_none {  { hidden1_106 in_data 0 16 } } }
	hidden1_107 { ap_none {  { hidden1_107 in_data 0 16 } } }
	hidden1_108 { ap_none {  { hidden1_108 in_data 0 16 } } }
	hidden1_109 { ap_none {  { hidden1_109 in_data 0 16 } } }
	hidden1_110 { ap_none {  { hidden1_110 in_data 0 16 } } }
	hidden1_111 { ap_none {  { hidden1_111 in_data 0 16 } } }
	hidden1_112 { ap_none {  { hidden1_112 in_data 0 16 } } }
	hidden1_113 { ap_none {  { hidden1_113 in_data 0 16 } } }
	hidden1_114 { ap_none {  { hidden1_114 in_data 0 16 } } }
	hidden1_115 { ap_none {  { hidden1_115 in_data 0 16 } } }
	hidden1_116 { ap_none {  { hidden1_116 in_data 0 16 } } }
	hidden1_117 { ap_none {  { hidden1_117 in_data 0 16 } } }
	hidden1_118 { ap_none {  { hidden1_118 in_data 0 16 } } }
	hidden1_119 { ap_none {  { hidden1_119 in_data 0 16 } } }
	hidden1_120 { ap_none {  { hidden1_120 in_data 0 16 } } }
	hidden1_121 { ap_none {  { hidden1_121 in_data 0 16 } } }
	hidden1_122 { ap_none {  { hidden1_122 in_data 0 16 } } }
	hidden1_123 { ap_none {  { hidden1_123 in_data 0 16 } } }
	hidden1_124 { ap_none {  { hidden1_124 in_data 0 16 } } }
	hidden1_125 { ap_none {  { hidden1_125 in_data 0 16 } } }
	hidden1_126 { ap_none {  { hidden1_126 in_data 0 16 } } }
	hidden1_127 { ap_none {  { hidden1_127 in_data 0 16 } } }
	sum_53_out { ap_vld {  { sum_53_out out_data 1 24 }  { sum_53_out_ap_vld out_vld 1 1 } } }
	dense_1_weights { ap_memory {  { dense_1_weights_address0 mem_address 1 13 }  { dense_1_weights_ce0 mem_ce 1 1 }  { dense_1_weights_q0 mem_dout 0 32 } } }
}
