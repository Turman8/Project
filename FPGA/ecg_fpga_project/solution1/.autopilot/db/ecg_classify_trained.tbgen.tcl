set moduleName ecg_classify_trained
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_hs
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {ecg_classify_trained}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ gmem0 int 32 regular {axi_master 0}  }
	{ gmem1 int 16 regular {axi_master 2}  }
	{ gmem2 int 32 regular {axi_master 1}  }
	{ features int 64 regular {axi_slave 0}  }
	{ probabilities int 64 regular {axi_slave 0}  }
	{ predicted_class int 64 regular {axi_slave 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "gmem0", "interface" : "axi_master", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "features","offset": { "type": "dynamic","port_name": "features","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "gmem1", "interface" : "axi_master", "bitwidth" : 16, "direction" : "READWRITE", "bitSlice":[ {"cElement": [{"cName": "probabilities","offset": { "type": "dynamic","port_name": "probabilities","bundle": "control"},"direction": "READWRITE"}]}]} , 
 	{ "Name" : "gmem2", "interface" : "axi_master", "bitwidth" : 32, "direction" : "WRITEONLY", "bitSlice":[ {"cElement": [{"cName": "predicted_class","offset": { "type": "dynamic","port_name": "predicted_class","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "features", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":27}} , 
 	{ "Name" : "probabilities", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":28}, "offset_end" : {"in":39}} , 
 	{ "Name" : "predicted_class", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":40}, "offset_end" : {"in":51}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ m_axi_gmem0_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_gmem0_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem0_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WDATA sc_out sc_lv 32 signal 0 } 
	{ m_axi_gmem0_WSTRB sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_ARADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_gmem0_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_gmem0_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_gmem0_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_gmem0_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_gmem0_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RDATA sc_in sc_lv 32 signal 0 } 
	{ m_axi_gmem0_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_gmem0_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_gmem0_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem0_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_gmem1_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_AWADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem1_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WDATA sc_out sc_lv 32 signal 1 } 
	{ m_axi_gmem1_WSTRB sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_ARADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_gmem1_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_gmem1_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_gmem1_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_gmem1_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_gmem1_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RDATA sc_in sc_lv 32 signal 1 } 
	{ m_axi_gmem1_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_gmem1_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_gmem1_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem1_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_gmem2_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_AWADDR sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem2_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem2_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem2_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem2_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem2_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem2_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_WDATA sc_out sc_lv 32 signal 2 } 
	{ m_axi_gmem2_WSTRB sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_ARADDR sc_out sc_lv 64 signal 2 } 
	{ m_axi_gmem2_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_gmem2_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem2_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem2_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_gmem2_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_gmem2_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_gmem2_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_gmem2_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_RDATA sc_in sc_lv 32 signal 2 } 
	{ m_axi_gmem2_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem2_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem2_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem2_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_gmem2_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_gmem2_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_gmem2_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_gmem2_BUSER sc_in sc_lv 1 signal 2 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 6 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"ecg_classify_trained","role":"start","value":"0","valid_bit":"0"},{"name":"ecg_classify_trained","role":"continue","value":"0","valid_bit":"4"},{"name":"ecg_classify_trained","role":"auto_start","value":"0","valid_bit":"7"},{"name":"features","role":"data","value":"16"},{"name":"probabilities","role":"data","value":"28"},{"name":"predicted_class","role":"data","value":"40"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":6, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"ecg_classify_trained","role":"start","value":"0","valid_bit":"0"},{"name":"ecg_classify_trained","role":"done","value":"0","valid_bit":"1"},{"name":"ecg_classify_trained","role":"idle","value":"0","valid_bit":"2"},{"name":"ecg_classify_trained","role":"ready","value":"0","valid_bit":"3"},{"name":"ecg_classify_trained","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_gmem0_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem0_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem0_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem0_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem0_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem0", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem0_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem0_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem0_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem0_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem0_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem0_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem0_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem0_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem0_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem0_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem0_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem0", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem0_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem0_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem0_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WID" }} , 
 	{ "name": "m_axi_gmem0_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem0_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem0_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem0_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem0", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem0_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem0_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem0", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem0_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem0_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem0_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem0_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem0_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem0", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem0_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem0_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem0", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem0_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem0_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem0_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem0_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem0", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem0_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem0_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RID" }} , 
 	{ "name": "m_axi_gmem0_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem0_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem0_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem0_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem0_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem0", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem0_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BID" }} , 
 	{ "name": "m_axi_gmem0_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem0", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem1", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem1_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem1_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem1_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem1_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem1_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem1_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem1_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem1_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem1_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem1_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem1_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem1", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem1_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem1_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem1_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WID" }} , 
 	{ "name": "m_axi_gmem1_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem1_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem1_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem1_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem1", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem1_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem1", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem1_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem1_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem1_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem1_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem1_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem1", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem1_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem1_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem1", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem1_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem1_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem1_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem1_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem1", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem1_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem1_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RID" }} , 
 	{ "name": "m_axi_gmem1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem1", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BID" }} , 
 	{ "name": "m_axi_gmem1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem1", "role": "BUSER" }} , 
 	{ "name": "m_axi_gmem2_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "AWVALID" }} , 
 	{ "name": "m_axi_gmem2_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "AWREADY" }} , 
 	{ "name": "m_axi_gmem2_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem2", "role": "AWADDR" }} , 
 	{ "name": "m_axi_gmem2_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "AWID" }} , 
 	{ "name": "m_axi_gmem2_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem2", "role": "AWLEN" }} , 
 	{ "name": "m_axi_gmem2_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem2", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_gmem2_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "AWBURST" }} , 
 	{ "name": "m_axi_gmem2_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_gmem2_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_gmem2_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem2", "role": "AWPROT" }} , 
 	{ "name": "m_axi_gmem2_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "AWQOS" }} , 
 	{ "name": "m_axi_gmem2_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "AWREGION" }} , 
 	{ "name": "m_axi_gmem2_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "AWUSER" }} , 
 	{ "name": "m_axi_gmem2_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "WVALID" }} , 
 	{ "name": "m_axi_gmem2_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "WREADY" }} , 
 	{ "name": "m_axi_gmem2_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem2", "role": "WDATA" }} , 
 	{ "name": "m_axi_gmem2_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "WSTRB" }} , 
 	{ "name": "m_axi_gmem2_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "WLAST" }} , 
 	{ "name": "m_axi_gmem2_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "WID" }} , 
 	{ "name": "m_axi_gmem2_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "WUSER" }} , 
 	{ "name": "m_axi_gmem2_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "ARVALID" }} , 
 	{ "name": "m_axi_gmem2_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "ARREADY" }} , 
 	{ "name": "m_axi_gmem2_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "gmem2", "role": "ARADDR" }} , 
 	{ "name": "m_axi_gmem2_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "ARID" }} , 
 	{ "name": "m_axi_gmem2_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "gmem2", "role": "ARLEN" }} , 
 	{ "name": "m_axi_gmem2_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem2", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_gmem2_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "ARBURST" }} , 
 	{ "name": "m_axi_gmem2_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_gmem2_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_gmem2_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "gmem2", "role": "ARPROT" }} , 
 	{ "name": "m_axi_gmem2_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "ARQOS" }} , 
 	{ "name": "m_axi_gmem2_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "gmem2", "role": "ARREGION" }} , 
 	{ "name": "m_axi_gmem2_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "ARUSER" }} , 
 	{ "name": "m_axi_gmem2_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "RVALID" }} , 
 	{ "name": "m_axi_gmem2_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "RREADY" }} , 
 	{ "name": "m_axi_gmem2_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "gmem2", "role": "RDATA" }} , 
 	{ "name": "m_axi_gmem2_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "RLAST" }} , 
 	{ "name": "m_axi_gmem2_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "RID" }} , 
 	{ "name": "m_axi_gmem2_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "RUSER" }} , 
 	{ "name": "m_axi_gmem2_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "RRESP" }} , 
 	{ "name": "m_axi_gmem2_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "BVALID" }} , 
 	{ "name": "m_axi_gmem2_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "BREADY" }} , 
 	{ "name": "m_axi_gmem2_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "gmem2", "role": "BRESP" }} , 
 	{ "name": "m_axi_gmem2_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "BID" }} , 
 	{ "name": "m_axi_gmem2_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "gmem2", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3", "4", "5", "6", "7", "8", "12", "16", "20", "24", "29", "34", "39", "44", "49", "54", "59", "64", "69", "74", "79", "84", "89", "94", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107"],
		"CDFG" : "ecg_classify_trained",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "29377", "EstimateLatencyMax" : "29569",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_AR", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695", "Port" : "gmem0", "Inst_start_state" : "73", "Inst_end_state" : "74"},
					{"ID" : "12", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707", "Port" : "gmem0", "Inst_start_state" : "154", "Inst_end_state" : "155"},
					{"ID" : "16", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719", "Port" : "gmem0", "Inst_start_state" : "235", "Inst_end_state" : "236"},
					{"ID" : "20", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731", "Port" : "gmem0", "Inst_start_state" : "316", "Inst_end_state" : "317"}]},
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "IO",
				"BlockSignal" : [
					{"Name" : "gmem1_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "gmem1_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "gmem1_blk_n_B", "Type" : "RtlSignal"},
					{"Name" : "gmem1_blk_n_AR", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "94", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_101_9_fu_2817", "Port" : "gmem1", "Inst_start_state" : "531", "Inst_end_state" : "532"}]},
			{"Name" : "gmem2", "Type" : "MAXI", "Direction" : "O",
				"BlockSignal" : [
					{"Name" : "gmem2_blk_n_AW", "Type" : "RtlSignal"},
					{"Name" : "gmem2_blk_n_W", "Type" : "RtlSignal"},
					{"Name" : "gmem2_blk_n_B", "Type" : "RtlSignal"}]},
			{"Name" : "features", "Type" : "None", "Direction" : "I"},
			{"Name" : "probabilities", "Type" : "None", "Direction" : "I"},
			{"Name" : "predicted_class", "Type" : "None", "Direction" : "I"},
			{"Name" : "dense_biases", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "dense_weights", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "8", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695", "Port" : "dense_weights", "Inst_start_state" : "73", "Inst_end_state" : "74"},
					{"ID" : "12", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707", "Port" : "dense_weights", "Inst_start_state" : "154", "Inst_end_state" : "155"},
					{"ID" : "16", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719", "Port" : "dense_weights", "Inst_start_state" : "235", "Inst_end_state" : "236"},
					{"ID" : "20", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731", "Port" : "dense_weights", "Inst_start_state" : "316", "Inst_end_state" : "317"}]},
			{"Name" : "dense_1_biases", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "dense_1_weights", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "34", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017", "Port" : "dense_1_weights", "Inst_start_state" : "349", "Inst_end_state" : "350"},
					{"ID" : "24", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743", "Port" : "dense_1_weights", "Inst_start_state" : "325", "Inst_end_state" : "326"},
					{"ID" : "39", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154", "Port" : "dense_1_weights", "Inst_start_state" : "361", "Inst_end_state" : "362"},
					{"ID" : "29", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880", "Port" : "dense_1_weights", "Inst_start_state" : "337", "Inst_end_state" : "338"}]},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "89", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778", "Port" : "dense_3_weights", "Inst_start_state" : "389", "Inst_end_state" : "390"},
					{"ID" : "44", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291", "Port" : "dense_3_weights", "Inst_start_state" : "365", "Inst_end_state" : "380"},
					{"ID" : "69", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622", "Port" : "dense_3_weights", "Inst_start_state" : "381", "Inst_end_state" : "382"},
					{"ID" : "74", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661", "Port" : "dense_3_weights", "Inst_start_state" : "383", "Inst_end_state" : "384"},
					{"ID" : "79", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700", "Port" : "dense_3_weights", "Inst_start_state" : "385", "Inst_end_state" : "386"},
					{"ID" : "84", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739", "Port" : "dense_3_weights", "Inst_start_state" : "387", "Inst_end_state" : "388"}]},
			{"Name" : "dense_2_biases", "Type" : "Memory", "Direction" : "I"},
			{"Name" : "dense_2_weights", "Type" : "Memory", "Direction" : "I",
				"SubConnect" : [
					{"ID" : "49", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330", "Port" : "dense_2_weights", "Inst_start_state" : "371", "Inst_end_state" : "372"},
					{"ID" : "54", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403", "Port" : "dense_2_weights", "Inst_start_state" : "373", "Inst_end_state" : "374"},
					{"ID" : "59", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476", "Port" : "dense_2_weights", "Inst_start_state" : "375", "Inst_end_state" : "376"},
					{"ID" : "64", "SubInstance" : "grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549", "Port" : "dense_2_weights", "Inst_start_state" : "377", "Inst_end_state" : "378"}]}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_37_1", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "601", "FirstState" : "ap_ST_fsm_state2", "LastState" : ["ap_ST_fsm_state318"], "QuitState" : ["ap_ST_fsm_state2"], "PreState" : ["ap_ST_fsm_state1"], "PostState" : ["ap_ST_fsm_state319"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "VITIS_LOOP_53_3", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "601", "FirstState" : "ap_ST_fsm_state319", "LastState" : ["ap_ST_fsm_state364"], "QuitState" : ["ap_ST_fsm_state319"], "PreState" : ["ap_ST_fsm_state2"], "PostState" : ["ap_ST_fsm_state365"], "OneDepthLoop" : "0", "OneStateBlock": ""}},
			{"Name" : "VITIS_LOOP_69_5", "PipelineType" : "no",
				"LoopDec" : {"FSMBitwidth" : "601", "FirstState" : "ap_ST_fsm_state365", "LastState" : ["ap_ST_fsm_state379"], "QuitState" : ["ap_ST_fsm_state365"], "PreState" : ["ap_ST_fsm_state319"], "PostState" : ["ap_ST_fsm_state380"], "OneDepthLoop" : "0", "OneStateBlock": ""}}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_biases_U", "Parent" : "0"},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_weights_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_1_biases_U", "Parent" : "0"},
	{"ID" : "4", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_1_weights_U", "Parent" : "0"},
	{"ID" : "5", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_3_weights_U", "Parent" : "0"},
	{"ID" : "6", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_2_biases_U", "Parent" : "0"},
	{"ID" : "7", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.dense_2_weights_U", "Parent" : "0"},
	{"ID" : "8", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695", "Parent" : "0", "Child" : ["9", "10", "11"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_41_2",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "56", "EstimateLatencyMax" : "56",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln39_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "p_cast_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln37_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_2_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_41_2", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "9", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695.sparsemux_9_3_16_1_1_U2", "Parent" : "8"},
	{"ID" : "10", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695.mac_muladd_16s_16s_32ns_32_4_1_U3", "Parent" : "8"},
	{"ID" : "11", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_2_fu_1695.flow_control_loop_pipe_sequential_init_U", "Parent" : "8"},
	{"ID" : "12", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707", "Parent" : "0", "Child" : ["13", "14", "15"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_41_21",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "56", "EstimateLatencyMax" : "56",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln39_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "p_cast_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_109", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_19_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_41_2", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "13", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707.sparsemux_9_3_16_1_1_U13", "Parent" : "12"},
	{"ID" : "14", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707.mac_muladd_16s_16s_32ns_32_4_1_U14", "Parent" : "12"},
	{"ID" : "15", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_21_fu_1707.flow_control_loop_pipe_sequential_init_U", "Parent" : "12"},
	{"ID" : "16", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719", "Parent" : "0", "Child" : ["17", "18", "19"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_41_22",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "56", "EstimateLatencyMax" : "56",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln39_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "p_cast_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_117", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_36_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_41_2", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "17", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719.sparsemux_9_3_16_1_1_U22", "Parent" : "16"},
	{"ID" : "18", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719.mac_muladd_16s_16s_32ns_32_4_1_U23", "Parent" : "16"},
	{"ID" : "19", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_22_fu_1719.flow_control_loop_pipe_sequential_init_U", "Parent" : "16"},
	{"ID" : "20", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731", "Parent" : "0", "Child" : ["21", "22", "23"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_41_23",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "56", "EstimateLatencyMax" : "56",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "sext_ln39_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "p_cast_cast", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_117", "Type" : "None", "Direction" : "I"},
			{"Name" : "sum_49_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_41_2", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "21", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731.sparsemux_9_3_16_1_1_U31", "Parent" : "20"},
	{"ID" : "22", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731.mac_muladd_16s_16s_32ns_32_4_1_U32", "Parent" : "20"},
	{"ID" : "23", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_41_23_fu_1731.flow_control_loop_pipe_sequential_init_U", "Parent" : "20"},
	{"ID" : "24", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743", "Parent" : "0", "Child" : ["25", "26", "27", "28"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_57_4",
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
			{"Name" : "sext_ln55_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln53_1", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_6_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_1_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_57_4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "25", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743.sparsemux_257_7_16_1_1_U40", "Parent" : "24"},
	{"ID" : "26", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743.sparsemux_9_3_16_1_1_U41", "Parent" : "24"},
	{"ID" : "27", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743.mac_muladd_16s_16s_32ns_32_4_1_U42", "Parent" : "24"},
	{"ID" : "28", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_4_fu_1743.flow_control_loop_pipe_sequential_init_U", "Parent" : "24"},
	{"ID" : "29", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880", "Parent" : "0", "Child" : ["30", "31", "32", "33"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_57_44",
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
			{"Name" : "sext_ln55_5", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_111", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_25_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_1_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_57_4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "30", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880.sparsemux_257_7_16_1_1_U177", "Parent" : "29"},
	{"ID" : "31", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880.sparsemux_9_3_16_1_1_U178", "Parent" : "29"},
	{"ID" : "32", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880.mac_muladd_16s_16s_32ns_32_4_1_U179", "Parent" : "29"},
	{"ID" : "33", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_44_fu_1880.flow_control_loop_pipe_sequential_init_U", "Parent" : "29"},
	{"ID" : "34", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017", "Parent" : "0", "Child" : ["35", "36", "37", "38"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_57_45",
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
			{"Name" : "sext_ln55_8", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_41_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_1_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_57_4", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "35", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017.sparsemux_257_7_16_1_1_U313", "Parent" : "34"},
	{"ID" : "36", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017.sparsemux_9_3_16_1_1_U314", "Parent" : "34"},
	{"ID" : "37", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017.mac_muladd_16s_16s_32ns_32_4_1_U315", "Parent" : "34"},
	{"ID" : "38", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_45_fu_2017.flow_control_loop_pipe_sequential_init_U", "Parent" : "34"},
	{"ID" : "39", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154", "Parent" : "0", "Child" : ["40", "41", "42", "43"],
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
	{"ID" : "40", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154.sparsemux_257_7_16_1_1_U449", "Parent" : "39"},
	{"ID" : "41", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154.sparsemux_9_3_16_1_1_U450", "Parent" : "39"},
	{"ID" : "42", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154.mac_muladd_16s_16s_32ns_32_4_1_U451", "Parent" : "39"},
	{"ID" : "43", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_57_46_fu_2154.flow_control_loop_pipe_sequential_init_U", "Parent" : "39"},
	{"ID" : "44", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291", "Parent" : "0", "Child" : ["45", "46", "47", "48"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_8",
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
			{"Name" : "sum_5_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "45", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291.sparsemux_65_5_16_1_1_U874", "Parent" : "44"},
	{"ID" : "46", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291.sparsemux_9_3_16_1_1_U875", "Parent" : "44"},
	{"ID" : "47", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291.mac_muladd_16s_16s_32ns_32_4_1_U876", "Parent" : "44"},
	{"ID" : "48", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_8_fu_2291.flow_control_loop_pipe_sequential_init_U", "Parent" : "44"},
	{"ID" : "49", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330", "Parent" : "0", "Child" : ["50", "51", "52", "53"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_73_6",
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
			{"Name" : "sext_ln71_2", "Type" : "None", "Direction" : "I"},
			{"Name" : "zext_ln69_1", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_10_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_2_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_73_6", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "50", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330.sparsemux_129_6_16_1_1_U585", "Parent" : "49"},
	{"ID" : "51", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330.sparsemux_9_3_16_1_1_U586", "Parent" : "49"},
	{"ID" : "52", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330.mac_muladd_16s_16s_32ns_32_4_1_U587", "Parent" : "49"},
	{"ID" : "53", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_6_fu_2330.flow_control_loop_pipe_sequential_init_U", "Parent" : "49"},
	{"ID" : "54", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403", "Parent" : "0", "Child" : ["55", "56", "57", "58"],
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
	{"ID" : "55", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403.sparsemux_129_6_16_1_1_U658", "Parent" : "54"},
	{"ID" : "56", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403.sparsemux_9_3_16_1_1_U659", "Parent" : "54"},
	{"ID" : "57", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403.mac_muladd_16s_16s_32ns_32_4_1_U660", "Parent" : "54"},
	{"ID" : "58", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_67_fu_2403.flow_control_loop_pipe_sequential_init_U", "Parent" : "54"},
	{"ID" : "59", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476", "Parent" : "0", "Child" : ["60", "61", "62", "63"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_73_68",
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
			{"Name" : "sext_ln71_8", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_114", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_34_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_2_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_73_6", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "60", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476.sparsemux_129_6_16_1_1_U730", "Parent" : "59"},
	{"ID" : "61", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476.sparsemux_9_3_16_1_1_U731", "Parent" : "59"},
	{"ID" : "62", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476.mac_muladd_16s_16s_32ns_32_4_1_U732", "Parent" : "59"},
	{"ID" : "63", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_68_fu_2476.flow_control_loop_pipe_sequential_init_U", "Parent" : "59"},
	{"ID" : "64", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549", "Parent" : "0", "Child" : ["65", "66", "67", "68"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_73_69",
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
			{"Name" : "sext_ln71_11", "Type" : "None", "Direction" : "I"},
			{"Name" : "tmp_114", "Type" : "None", "Direction" : "I"},
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
			{"Name" : "sum_40_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_2_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_73_6", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter9", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter9", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "65", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549.sparsemux_129_6_16_1_1_U802", "Parent" : "64"},
	{"ID" : "66", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549.sparsemux_9_3_16_1_1_U803", "Parent" : "64"},
	{"ID" : "67", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549.mac_muladd_16s_16s_32ns_32_4_1_U804", "Parent" : "64"},
	{"ID" : "68", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_73_69_fu_2549.flow_control_loop_pipe_sequential_init_U", "Parent" : "64"},
	{"ID" : "69", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622", "Parent" : "0", "Child" : ["70", "71", "72", "73"],
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
	{"ID" : "70", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622.sparsemux_65_5_16_1_1_U913", "Parent" : "69"},
	{"ID" : "71", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622.sparsemux_9_3_16_1_1_U914", "Parent" : "69"},
	{"ID" : "72", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622.mac_muladd_16s_16s_32ns_32_4_1_U915", "Parent" : "69"},
	{"ID" : "73", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_810_fu_2622.flow_control_loop_pipe_sequential_init_U", "Parent" : "69"},
	{"ID" : "74", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661", "Parent" : "0", "Child" : ["75", "76", "77", "78"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_811",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "43", "EstimateLatencyMax" : "43",
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
			{"Name" : "sum_11_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter10", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter10", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "75", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661.sparsemux_65_5_16_1_1_U951", "Parent" : "74"},
	{"ID" : "76", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661.sparsemux_9_3_16_1_1_U952", "Parent" : "74"},
	{"ID" : "77", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661.mac_muladd_16s_16s_32ns_32_4_1_U953", "Parent" : "74"},
	{"ID" : "78", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_811_fu_2661.flow_control_loop_pipe_sequential_init_U", "Parent" : "74"},
	{"ID" : "79", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700", "Parent" : "0", "Child" : ["80", "81", "82", "83"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_812",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "43", "EstimateLatencyMax" : "43",
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
			{"Name" : "sum_17_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter10", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter10", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "80", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700.sparsemux_65_5_16_1_1_U989", "Parent" : "79"},
	{"ID" : "81", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700.sparsemux_9_3_16_1_1_U990", "Parent" : "79"},
	{"ID" : "82", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700.mac_muladd_16s_16s_32ns_32_4_1_U991", "Parent" : "79"},
	{"ID" : "83", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_812_fu_2700.flow_control_loop_pipe_sequential_init_U", "Parent" : "79"},
	{"ID" : "84", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739", "Parent" : "0", "Child" : ["85", "86", "87", "88"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_813",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "43", "EstimateLatencyMax" : "43",
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
			{"Name" : "sum_23_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter10", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter10", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "85", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739.sparsemux_65_5_16_1_1_U1027", "Parent" : "84"},
	{"ID" : "86", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739.sparsemux_9_3_16_1_1_U1028", "Parent" : "84"},
	{"ID" : "87", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739.mac_muladd_16s_16s_32ns_32_4_1_U1029", "Parent" : "84"},
	{"ID" : "88", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_813_fu_2739.flow_control_loop_pipe_sequential_init_U", "Parent" : "84"},
	{"ID" : "89", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778", "Parent" : "0", "Child" : ["90", "91", "92", "93"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_88_814",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "43", "EstimateLatencyMax" : "43",
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
			{"Name" : "sum_28_out", "Type" : "Vld", "Direction" : "O"},
			{"Name" : "dense_3_weights", "Type" : "Memory", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_88_8", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter10", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter10", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "90", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778.sparsemux_65_5_16_1_1_U1065", "Parent" : "89"},
	{"ID" : "91", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778.sparsemux_9_3_16_1_1_U1066", "Parent" : "89"},
	{"ID" : "92", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778.mac_muladd_16s_16s_32ns_32_4_1_U1067", "Parent" : "89"},
	{"ID" : "93", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_88_814_fu_2778.flow_control_loop_pipe_sequential_init_U", "Parent" : "89"},
	{"ID" : "94", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_101_9_fu_2817", "Parent" : "0", "Child" : ["95"],
		"CDFG" : "ecg_classify_trained_Pipeline_VITIS_LOOP_101_9",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "8", "EstimateLatencyMax" : "8",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "output_r", "Type" : "None", "Direction" : "I"},
			{"Name" : "gmem1", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "gmem1_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "sext_ln94_1", "Type" : "None", "Direction" : "I"},
			{"Name" : "i120_01023_out", "Type" : "Vld", "Direction" : "O"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_101_9", "PipelineType" : "UPC",
				"LoopDec" : {"FSMBitwidth" : "1", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter2", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter2", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "OneDepthLoop" : "0", "has_ap_ctrl" : "1", "has_continue" : "0"}}]},
	{"ID" : "95", "Level" : "2", "Path" : "`AUTOTB_DUT_INST.grp_ecg_classify_trained_Pipeline_VITIS_LOOP_101_9_fu_2817.flow_control_loop_pipe_sequential_init_U", "Parent" : "94"},
	{"ID" : "96", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "97", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem0_m_axi_U", "Parent" : "0"},
	{"ID" : "98", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem1_m_axi_U", "Parent" : "0"},
	{"ID" : "99", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.gmem2_m_axi_U", "Parent" : "0"},
	{"ID" : "100", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_2_no_dsp_1_U1106", "Parent" : "0"},
	{"ID" : "101", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1107", "Parent" : "0"},
	{"ID" : "102", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1108", "Parent" : "0"},
	{"ID" : "103", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1109", "Parent" : "0"},
	{"ID" : "104", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1110", "Parent" : "0"},
	{"ID" : "105", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1111", "Parent" : "0"},
	{"ID" : "106", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.sparsemux_9_3_16_1_1_U1112", "Parent" : "0"},
	{"ID" : "107", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.fpext_32ns_64_2_no_dsp_1_U1113", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	ecg_classify_trained {
		gmem0 {Type I LastRead 238 FirstWrite -1}
		gmem1 {Type IO LastRead 84 FirstWrite -1}
		gmem2 {Type O LastRead 158 FirstWrite 157}
		features {Type I LastRead 0 FirstWrite -1}
		probabilities {Type I LastRead 0 FirstWrite -1}
		predicted_class {Type I LastRead 0 FirstWrite -1}
		dense_biases {Type I LastRead -1 FirstWrite -1}
		dense_weights {Type I LastRead -1 FirstWrite -1}
		dense_1_biases {Type I LastRead -1 FirstWrite -1}
		dense_1_weights {Type I LastRead -1 FirstWrite -1}
		dense_3_weights {Type I LastRead -1 FirstWrite -1}
		dense_2_biases {Type I LastRead -1 FirstWrite -1}
		dense_2_weights {Type I LastRead -1 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_41_2 {
		sext_ln39_2 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type I LastRead 5 FirstWrite -1}
		p_cast_cast {Type I LastRead 0 FirstWrite -1}
		zext_ln37_1 {Type I LastRead 0 FirstWrite -1}
		sum_2_out {Type O LastRead -1 FirstWrite 8}
		dense_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_41_21 {
		sext_ln39_5 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type I LastRead 5 FirstWrite -1}
		p_cast_cast {Type I LastRead 0 FirstWrite -1}
		tmp_109 {Type I LastRead 0 FirstWrite -1}
		sum_19_out {Type O LastRead -1 FirstWrite 8}
		dense_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_41_22 {
		sext_ln39_8 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type I LastRead 5 FirstWrite -1}
		p_cast_cast {Type I LastRead 0 FirstWrite -1}
		tmp_117 {Type I LastRead 0 FirstWrite -1}
		sum_36_out {Type O LastRead -1 FirstWrite 8}
		dense_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_41_23 {
		sext_ln39_11 {Type I LastRead 0 FirstWrite -1}
		gmem0 {Type I LastRead 5 FirstWrite -1}
		p_cast_cast {Type I LastRead 0 FirstWrite -1}
		tmp_117 {Type I LastRead 0 FirstWrite -1}
		sum_49_out {Type O LastRead -1 FirstWrite 8}
		dense_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_57_4 {
		sext_ln55_2 {Type I LastRead 0 FirstWrite -1}
		zext_ln53_1 {Type I LastRead 0 FirstWrite -1}
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
		sum_6_out {Type O LastRead -1 FirstWrite 8}
		dense_1_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_57_44 {
		sext_ln55_5 {Type I LastRead 0 FirstWrite -1}
		tmp_111 {Type I LastRead 0 FirstWrite -1}
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
		sum_25_out {Type O LastRead -1 FirstWrite 8}
		dense_1_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_57_45 {
		sext_ln55_8 {Type I LastRead 0 FirstWrite -1}
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
		sum_41_out {Type O LastRead -1 FirstWrite 8}
		dense_1_weights {Type I LastRead 0 FirstWrite -1}}
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
		dense_1_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_8 {
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
		sum_5_out {Type O LastRead -1 FirstWrite 8}
		dense_3_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_73_6 {
		sext_ln71_2 {Type I LastRead 0 FirstWrite -1}
		zext_ln69_1 {Type I LastRead 0 FirstWrite -1}
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
		sum_10_out {Type O LastRead -1 FirstWrite 8}
		dense_2_weights {Type I LastRead 0 FirstWrite -1}}
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
		dense_2_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_73_68 {
		sext_ln71_8 {Type I LastRead 0 FirstWrite -1}
		tmp_114 {Type I LastRead 0 FirstWrite -1}
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
		sum_34_out {Type O LastRead -1 FirstWrite 8}
		dense_2_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_73_69 {
		sext_ln71_11 {Type I LastRead 0 FirstWrite -1}
		tmp_114 {Type I LastRead 0 FirstWrite -1}
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
		sum_40_out {Type O LastRead -1 FirstWrite 8}
		dense_2_weights {Type I LastRead 0 FirstWrite -1}}
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
		dense_3_weights {Type I LastRead 0 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_811 {
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
		sum_11_out {Type O LastRead -1 FirstWrite 9}
		dense_3_weights {Type I LastRead 1 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_812 {
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
		sum_17_out {Type O LastRead -1 FirstWrite 9}
		dense_3_weights {Type I LastRead 1 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_813 {
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
		sum_23_out {Type O LastRead -1 FirstWrite 9}
		dense_3_weights {Type I LastRead 1 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_88_814 {
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
		sum_28_out {Type O LastRead -1 FirstWrite 9}
		dense_3_weights {Type I LastRead 1 FirstWrite -1}}
	ecg_classify_trained_Pipeline_VITIS_LOOP_101_9 {
		output_r {Type I LastRead 0 FirstWrite -1}
		gmem1 {Type I LastRead 1 FirstWrite -1}
		sext_ln94_1 {Type I LastRead 0 FirstWrite -1}
		i120_01023_out {Type O LastRead -1 FirstWrite 1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "29377", "Max" : "29569"}
	, {"Name" : "Interval", "Min" : "29378", "Max" : "29570"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	gmem0 { m_axi {  { m_axi_gmem0_AWVALID VALID 1 1 }  { m_axi_gmem0_AWREADY READY 0 1 }  { m_axi_gmem0_AWADDR ADDR 1 64 }  { m_axi_gmem0_AWID ID 1 1 }  { m_axi_gmem0_AWLEN SIZE 1 8 }  { m_axi_gmem0_AWSIZE BURST 1 3 }  { m_axi_gmem0_AWBURST LOCK 1 2 }  { m_axi_gmem0_AWLOCK CACHE 1 2 }  { m_axi_gmem0_AWCACHE PROT 1 4 }  { m_axi_gmem0_AWPROT QOS 1 3 }  { m_axi_gmem0_AWQOS REGION 1 4 }  { m_axi_gmem0_AWREGION USER 1 4 }  { m_axi_gmem0_AWUSER DATA 1 1 }  { m_axi_gmem0_WVALID VALID 1 1 }  { m_axi_gmem0_WREADY READY 0 1 }  { m_axi_gmem0_WDATA FIFONUM 1 32 }  { m_axi_gmem0_WSTRB STRB 1 4 }  { m_axi_gmem0_WLAST LAST 1 1 }  { m_axi_gmem0_WID ID 1 1 }  { m_axi_gmem0_WUSER DATA 1 1 }  { m_axi_gmem0_ARVALID VALID 1 1 }  { m_axi_gmem0_ARREADY READY 0 1 }  { m_axi_gmem0_ARADDR ADDR 1 64 }  { m_axi_gmem0_ARID ID 1 1 }  { m_axi_gmem0_ARLEN SIZE 1 8 }  { m_axi_gmem0_ARSIZE BURST 1 3 }  { m_axi_gmem0_ARBURST LOCK 1 2 }  { m_axi_gmem0_ARLOCK CACHE 1 2 }  { m_axi_gmem0_ARCACHE PROT 1 4 }  { m_axi_gmem0_ARPROT QOS 1 3 }  { m_axi_gmem0_ARQOS REGION 1 4 }  { m_axi_gmem0_ARREGION USER 1 4 }  { m_axi_gmem0_ARUSER DATA 1 1 }  { m_axi_gmem0_RVALID VALID 0 1 }  { m_axi_gmem0_RREADY READY 1 1 }  { m_axi_gmem0_RDATA FIFONUM 0 32 }  { m_axi_gmem0_RLAST LAST 0 1 }  { m_axi_gmem0_RID ID 0 1 }  { m_axi_gmem0_RUSER DATA 0 1 }  { m_axi_gmem0_RRESP RESP 0 2 }  { m_axi_gmem0_BVALID VALID 0 1 }  { m_axi_gmem0_BREADY READY 1 1 }  { m_axi_gmem0_BRESP RESP 0 2 }  { m_axi_gmem0_BID ID 0 1 }  { m_axi_gmem0_BUSER DATA 0 1 } } }
	gmem1 { m_axi {  { m_axi_gmem1_AWVALID VALID 1 1 }  { m_axi_gmem1_AWREADY READY 0 1 }  { m_axi_gmem1_AWADDR ADDR 1 64 }  { m_axi_gmem1_AWID ID 1 1 }  { m_axi_gmem1_AWLEN SIZE 1 8 }  { m_axi_gmem1_AWSIZE BURST 1 3 }  { m_axi_gmem1_AWBURST LOCK 1 2 }  { m_axi_gmem1_AWLOCK CACHE 1 2 }  { m_axi_gmem1_AWCACHE PROT 1 4 }  { m_axi_gmem1_AWPROT QOS 1 3 }  { m_axi_gmem1_AWQOS REGION 1 4 }  { m_axi_gmem1_AWREGION USER 1 4 }  { m_axi_gmem1_AWUSER DATA 1 1 }  { m_axi_gmem1_WVALID VALID 1 1 }  { m_axi_gmem1_WREADY READY 0 1 }  { m_axi_gmem1_WDATA FIFONUM 1 32 }  { m_axi_gmem1_WSTRB STRB 1 4 }  { m_axi_gmem1_WLAST LAST 1 1 }  { m_axi_gmem1_WID ID 1 1 }  { m_axi_gmem1_WUSER DATA 1 1 }  { m_axi_gmem1_ARVALID VALID 1 1 }  { m_axi_gmem1_ARREADY READY 0 1 }  { m_axi_gmem1_ARADDR ADDR 1 64 }  { m_axi_gmem1_ARID ID 1 1 }  { m_axi_gmem1_ARLEN SIZE 1 8 }  { m_axi_gmem1_ARSIZE BURST 1 3 }  { m_axi_gmem1_ARBURST LOCK 1 2 }  { m_axi_gmem1_ARLOCK CACHE 1 2 }  { m_axi_gmem1_ARCACHE PROT 1 4 }  { m_axi_gmem1_ARPROT QOS 1 3 }  { m_axi_gmem1_ARQOS REGION 1 4 }  { m_axi_gmem1_ARREGION USER 1 4 }  { m_axi_gmem1_ARUSER DATA 1 1 }  { m_axi_gmem1_RVALID VALID 0 1 }  { m_axi_gmem1_RREADY READY 1 1 }  { m_axi_gmem1_RDATA FIFONUM 0 32 }  { m_axi_gmem1_RLAST LAST 0 1 }  { m_axi_gmem1_RID ID 0 1 }  { m_axi_gmem1_RUSER DATA 0 1 }  { m_axi_gmem1_RRESP RESP 0 2 }  { m_axi_gmem1_BVALID VALID 0 1 }  { m_axi_gmem1_BREADY READY 1 1 }  { m_axi_gmem1_BRESP RESP 0 2 }  { m_axi_gmem1_BID ID 0 1 }  { m_axi_gmem1_BUSER DATA 0 1 } } }
	gmem2 { m_axi {  { m_axi_gmem2_AWVALID VALID 1 1 }  { m_axi_gmem2_AWREADY READY 0 1 }  { m_axi_gmem2_AWADDR ADDR 1 64 }  { m_axi_gmem2_AWID ID 1 1 }  { m_axi_gmem2_AWLEN SIZE 1 8 }  { m_axi_gmem2_AWSIZE BURST 1 3 }  { m_axi_gmem2_AWBURST LOCK 1 2 }  { m_axi_gmem2_AWLOCK CACHE 1 2 }  { m_axi_gmem2_AWCACHE PROT 1 4 }  { m_axi_gmem2_AWPROT QOS 1 3 }  { m_axi_gmem2_AWQOS REGION 1 4 }  { m_axi_gmem2_AWREGION USER 1 4 }  { m_axi_gmem2_AWUSER DATA 1 1 }  { m_axi_gmem2_WVALID VALID 1 1 }  { m_axi_gmem2_WREADY READY 0 1 }  { m_axi_gmem2_WDATA FIFONUM 1 32 }  { m_axi_gmem2_WSTRB STRB 1 4 }  { m_axi_gmem2_WLAST LAST 1 1 }  { m_axi_gmem2_WID ID 1 1 }  { m_axi_gmem2_WUSER DATA 1 1 }  { m_axi_gmem2_ARVALID VALID 1 1 }  { m_axi_gmem2_ARREADY READY 0 1 }  { m_axi_gmem2_ARADDR ADDR 1 64 }  { m_axi_gmem2_ARID ID 1 1 }  { m_axi_gmem2_ARLEN SIZE 1 8 }  { m_axi_gmem2_ARSIZE BURST 1 3 }  { m_axi_gmem2_ARBURST LOCK 1 2 }  { m_axi_gmem2_ARLOCK CACHE 1 2 }  { m_axi_gmem2_ARCACHE PROT 1 4 }  { m_axi_gmem2_ARPROT QOS 1 3 }  { m_axi_gmem2_ARQOS REGION 1 4 }  { m_axi_gmem2_ARREGION USER 1 4 }  { m_axi_gmem2_ARUSER DATA 1 1 }  { m_axi_gmem2_RVALID VALID 0 1 }  { m_axi_gmem2_RREADY READY 1 1 }  { m_axi_gmem2_RDATA FIFONUM 0 32 }  { m_axi_gmem2_RLAST LAST 0 1 }  { m_axi_gmem2_RID ID 0 1 }  { m_axi_gmem2_RUSER DATA 0 1 }  { m_axi_gmem2_RRESP RESP 0 2 }  { m_axi_gmem2_BVALID VALID 0 1 }  { m_axi_gmem2_BREADY READY 1 1 }  { m_axi_gmem2_BRESP RESP 0 2 }  { m_axi_gmem2_BID ID 0 1 }  { m_axi_gmem2_BUSER DATA 0 1 } } }
}

set maxi_interface_dict [dict create]
dict set maxi_interface_dict gmem0 { CHANNEL_NUM 0 BUNDLE gmem0 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}
dict set maxi_interface_dict gmem1 { CHANNEL_NUM 0 BUNDLE gmem1 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_WRITE}
dict set maxi_interface_dict gmem2 { CHANNEL_NUM 0 BUNDLE gmem2 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE WRITE_ONLY}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ gmem0 64 }
	{ gmem1 64 }
	{ gmem2 64 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ gmem0 64 }
	{ gmem1 64 }
	{ gmem2 64 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
