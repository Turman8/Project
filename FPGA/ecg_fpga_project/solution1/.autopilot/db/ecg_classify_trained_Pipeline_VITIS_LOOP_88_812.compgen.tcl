# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# XIL_BRAM:
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc ::AESL_LIB_XILADAPTER::xil_bram_gen] == "::AESL_LIB_XILADAPTER::xil_bram_gen"} {
eval "::AESL_LIB_XILADAPTER::xil_bram_gen { \
    id 1025 \
    name dense_3_weights \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename dense_3_weights \
    op interface \
    ports { dense_3_weights_address0 { O 8 vector } dense_3_weights_ce0 { O 1 bit } dense_3_weights_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'dense_3_weights'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 992 \
    name hidden3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3 \
    op interface \
    ports { hidden3 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 993 \
    name hidden3_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_1 \
    op interface \
    ports { hidden3_1 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 994 \
    name hidden3_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_2 \
    op interface \
    ports { hidden3_2 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 995 \
    name hidden3_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_3 \
    op interface \
    ports { hidden3_3 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 996 \
    name hidden3_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_4 \
    op interface \
    ports { hidden3_4 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 997 \
    name hidden3_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_5 \
    op interface \
    ports { hidden3_5 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 998 \
    name hidden3_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_6 \
    op interface \
    ports { hidden3_6 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 999 \
    name hidden3_7 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_7 \
    op interface \
    ports { hidden3_7 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1000 \
    name hidden3_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_8 \
    op interface \
    ports { hidden3_8 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1001 \
    name hidden3_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_9 \
    op interface \
    ports { hidden3_9 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1002 \
    name hidden3_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_10 \
    op interface \
    ports { hidden3_10 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1003 \
    name hidden3_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_11 \
    op interface \
    ports { hidden3_11 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1004 \
    name hidden3_12 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_12 \
    op interface \
    ports { hidden3_12 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1005 \
    name hidden3_13 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_13 \
    op interface \
    ports { hidden3_13 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1006 \
    name hidden3_14 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_14 \
    op interface \
    ports { hidden3_14 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1007 \
    name hidden3_15 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_15 \
    op interface \
    ports { hidden3_15 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1008 \
    name hidden3_16 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_16 \
    op interface \
    ports { hidden3_16 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1009 \
    name hidden3_17 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_17 \
    op interface \
    ports { hidden3_17 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1010 \
    name hidden3_18 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_18 \
    op interface \
    ports { hidden3_18 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1011 \
    name hidden3_19 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_19 \
    op interface \
    ports { hidden3_19 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1012 \
    name hidden3_20 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_20 \
    op interface \
    ports { hidden3_20 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1013 \
    name hidden3_21 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_21 \
    op interface \
    ports { hidden3_21 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1014 \
    name hidden3_22 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_22 \
    op interface \
    ports { hidden3_22 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1015 \
    name hidden3_23 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_23 \
    op interface \
    ports { hidden3_23 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1016 \
    name hidden3_24 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_24 \
    op interface \
    ports { hidden3_24 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1017 \
    name hidden3_25 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_25 \
    op interface \
    ports { hidden3_25 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1018 \
    name hidden3_26 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_26 \
    op interface \
    ports { hidden3_26 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1019 \
    name hidden3_27 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_27 \
    op interface \
    ports { hidden3_27 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1020 \
    name hidden3_28 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_28 \
    op interface \
    ports { hidden3_28 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1021 \
    name hidden3_29 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_29 \
    op interface \
    ports { hidden3_29 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1022 \
    name hidden3_30 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_30 \
    op interface \
    ports { hidden3_30 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1023 \
    name hidden3_31 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden3_31 \
    op interface \
    ports { hidden3_31 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1024 \
    name sum_17_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_sum_17_out \
    op interface \
    ports { sum_17_out { O 24 vector } sum_17_out_ap_vld { O 1 bit } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


# flow_control definition:
set InstName ecg_classify_trained_flow_control_loop_pipe_sequential_init_U
set CompName ecg_classify_trained_flow_control_loop_pipe_sequential_init
set name flow_control_loop_pipe_sequential_init
if {${::AESL::PGuard_autocg_gen} && ${::AESL::PGuard_autocg_ipmgen}} {
if {[info proc ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control] == "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control"} {
eval "::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control { \
    name ${name} \
    prefix ecg_classify_trained_ \
}"
} else {
puts "@W \[IMPL-107\] Cannot find ::AESL_LIB_VIRTEX::xil_gen_UPC_flow_control, check your platform lib"
}
}


if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler $CompName BINDTYPE interface TYPE internal_upc_flow_control INSTNAME $InstName
}


