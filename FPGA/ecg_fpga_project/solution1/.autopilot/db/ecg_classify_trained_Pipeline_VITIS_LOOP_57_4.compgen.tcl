# This script segment is generated automatically by AutoPilot

if {${::AESL::PGuard_rtl_comp_handler}} {
	::AP::rtl_comp_handler ecg_classify_trained_sparsemux_257_7_16_1_1 BINDTYPE {op} TYPE {sparsemux} IMPL {auto}
}


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
    id 175 \
    name dense_1_weights \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename dense_1_weights \
    op interface \
    ports { dense_1_weights_address0 { O 13 vector } dense_1_weights_ce0 { O 1 bit } dense_1_weights_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'dense_1_weights'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 44 \
    name sext_ln55_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln55_2 \
    op interface \
    ports { sext_ln55_2 { I 24 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 45 \
    name zext_ln53_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_zext_ln53_1 \
    op interface \
    ports { zext_ln53_1 { I 6 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 46 \
    name hidden1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1 \
    op interface \
    ports { hidden1 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 47 \
    name hidden1_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_1 \
    op interface \
    ports { hidden1_1 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 48 \
    name hidden1_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_2 \
    op interface \
    ports { hidden1_2 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 49 \
    name hidden1_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_3 \
    op interface \
    ports { hidden1_3 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 50 \
    name hidden1_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_4 \
    op interface \
    ports { hidden1_4 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 51 \
    name hidden1_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_5 \
    op interface \
    ports { hidden1_5 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 52 \
    name hidden1_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_6 \
    op interface \
    ports { hidden1_6 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 53 \
    name hidden1_7 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_7 \
    op interface \
    ports { hidden1_7 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 54 \
    name hidden1_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_8 \
    op interface \
    ports { hidden1_8 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 55 \
    name hidden1_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_9 \
    op interface \
    ports { hidden1_9 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 56 \
    name hidden1_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_10 \
    op interface \
    ports { hidden1_10 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 57 \
    name hidden1_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_11 \
    op interface \
    ports { hidden1_11 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 58 \
    name hidden1_12 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_12 \
    op interface \
    ports { hidden1_12 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 59 \
    name hidden1_13 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_13 \
    op interface \
    ports { hidden1_13 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 60 \
    name hidden1_14 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_14 \
    op interface \
    ports { hidden1_14 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 61 \
    name hidden1_15 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_15 \
    op interface \
    ports { hidden1_15 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 62 \
    name hidden1_16 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_16 \
    op interface \
    ports { hidden1_16 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 63 \
    name hidden1_17 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_17 \
    op interface \
    ports { hidden1_17 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 64 \
    name hidden1_18 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_18 \
    op interface \
    ports { hidden1_18 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 65 \
    name hidden1_19 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_19 \
    op interface \
    ports { hidden1_19 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 66 \
    name hidden1_20 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_20 \
    op interface \
    ports { hidden1_20 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 67 \
    name hidden1_21 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_21 \
    op interface \
    ports { hidden1_21 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 68 \
    name hidden1_22 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_22 \
    op interface \
    ports { hidden1_22 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 69 \
    name hidden1_23 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_23 \
    op interface \
    ports { hidden1_23 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 70 \
    name hidden1_24 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_24 \
    op interface \
    ports { hidden1_24 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 71 \
    name hidden1_25 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_25 \
    op interface \
    ports { hidden1_25 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 72 \
    name hidden1_26 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_26 \
    op interface \
    ports { hidden1_26 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 73 \
    name hidden1_27 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_27 \
    op interface \
    ports { hidden1_27 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 74 \
    name hidden1_28 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_28 \
    op interface \
    ports { hidden1_28 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 75 \
    name hidden1_29 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_29 \
    op interface \
    ports { hidden1_29 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 76 \
    name hidden1_30 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_30 \
    op interface \
    ports { hidden1_30 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 77 \
    name hidden1_31 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_31 \
    op interface \
    ports { hidden1_31 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 78 \
    name hidden1_32 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_32 \
    op interface \
    ports { hidden1_32 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 79 \
    name hidden1_33 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_33 \
    op interface \
    ports { hidden1_33 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 80 \
    name hidden1_34 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_34 \
    op interface \
    ports { hidden1_34 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 81 \
    name hidden1_35 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_35 \
    op interface \
    ports { hidden1_35 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 82 \
    name hidden1_36 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_36 \
    op interface \
    ports { hidden1_36 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 83 \
    name hidden1_37 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_37 \
    op interface \
    ports { hidden1_37 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 84 \
    name hidden1_38 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_38 \
    op interface \
    ports { hidden1_38 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 85 \
    name hidden1_39 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_39 \
    op interface \
    ports { hidden1_39 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 86 \
    name hidden1_40 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_40 \
    op interface \
    ports { hidden1_40 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 87 \
    name hidden1_41 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_41 \
    op interface \
    ports { hidden1_41 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 88 \
    name hidden1_42 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_42 \
    op interface \
    ports { hidden1_42 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 89 \
    name hidden1_43 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_43 \
    op interface \
    ports { hidden1_43 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 90 \
    name hidden1_44 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_44 \
    op interface \
    ports { hidden1_44 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 91 \
    name hidden1_45 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_45 \
    op interface \
    ports { hidden1_45 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 92 \
    name hidden1_46 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_46 \
    op interface \
    ports { hidden1_46 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 93 \
    name hidden1_47 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_47 \
    op interface \
    ports { hidden1_47 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 94 \
    name hidden1_48 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_48 \
    op interface \
    ports { hidden1_48 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 95 \
    name hidden1_49 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_49 \
    op interface \
    ports { hidden1_49 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 96 \
    name hidden1_50 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_50 \
    op interface \
    ports { hidden1_50 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 97 \
    name hidden1_51 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_51 \
    op interface \
    ports { hidden1_51 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 98 \
    name hidden1_52 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_52 \
    op interface \
    ports { hidden1_52 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 99 \
    name hidden1_53 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_53 \
    op interface \
    ports { hidden1_53 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 100 \
    name hidden1_54 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_54 \
    op interface \
    ports { hidden1_54 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 101 \
    name hidden1_55 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_55 \
    op interface \
    ports { hidden1_55 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 102 \
    name hidden1_56 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_56 \
    op interface \
    ports { hidden1_56 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 103 \
    name hidden1_57 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_57 \
    op interface \
    ports { hidden1_57 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 104 \
    name hidden1_58 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_58 \
    op interface \
    ports { hidden1_58 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 105 \
    name hidden1_59 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_59 \
    op interface \
    ports { hidden1_59 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 106 \
    name hidden1_60 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_60 \
    op interface \
    ports { hidden1_60 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 107 \
    name hidden1_61 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_61 \
    op interface \
    ports { hidden1_61 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 108 \
    name hidden1_62 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_62 \
    op interface \
    ports { hidden1_62 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 109 \
    name hidden1_63 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_63 \
    op interface \
    ports { hidden1_63 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 110 \
    name hidden1_64 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_64 \
    op interface \
    ports { hidden1_64 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 111 \
    name hidden1_65 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_65 \
    op interface \
    ports { hidden1_65 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 112 \
    name hidden1_66 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_66 \
    op interface \
    ports { hidden1_66 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 113 \
    name hidden1_67 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_67 \
    op interface \
    ports { hidden1_67 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 114 \
    name hidden1_68 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_68 \
    op interface \
    ports { hidden1_68 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 115 \
    name hidden1_69 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_69 \
    op interface \
    ports { hidden1_69 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 116 \
    name hidden1_70 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_70 \
    op interface \
    ports { hidden1_70 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 117 \
    name hidden1_71 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_71 \
    op interface \
    ports { hidden1_71 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 118 \
    name hidden1_72 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_72 \
    op interface \
    ports { hidden1_72 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 119 \
    name hidden1_73 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_73 \
    op interface \
    ports { hidden1_73 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 120 \
    name hidden1_74 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_74 \
    op interface \
    ports { hidden1_74 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 121 \
    name hidden1_75 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_75 \
    op interface \
    ports { hidden1_75 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 122 \
    name hidden1_76 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_76 \
    op interface \
    ports { hidden1_76 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 123 \
    name hidden1_77 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_77 \
    op interface \
    ports { hidden1_77 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 124 \
    name hidden1_78 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_78 \
    op interface \
    ports { hidden1_78 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 125 \
    name hidden1_79 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_79 \
    op interface \
    ports { hidden1_79 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 126 \
    name hidden1_80 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_80 \
    op interface \
    ports { hidden1_80 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 127 \
    name hidden1_81 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_81 \
    op interface \
    ports { hidden1_81 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 128 \
    name hidden1_82 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_82 \
    op interface \
    ports { hidden1_82 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 129 \
    name hidden1_83 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_83 \
    op interface \
    ports { hidden1_83 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 130 \
    name hidden1_84 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_84 \
    op interface \
    ports { hidden1_84 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 131 \
    name hidden1_85 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_85 \
    op interface \
    ports { hidden1_85 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 132 \
    name hidden1_86 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_86 \
    op interface \
    ports { hidden1_86 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 133 \
    name hidden1_87 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_87 \
    op interface \
    ports { hidden1_87 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 134 \
    name hidden1_88 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_88 \
    op interface \
    ports { hidden1_88 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 135 \
    name hidden1_89 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_89 \
    op interface \
    ports { hidden1_89 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 136 \
    name hidden1_90 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_90 \
    op interface \
    ports { hidden1_90 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 137 \
    name hidden1_91 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_91 \
    op interface \
    ports { hidden1_91 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 138 \
    name hidden1_92 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_92 \
    op interface \
    ports { hidden1_92 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 139 \
    name hidden1_93 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_93 \
    op interface \
    ports { hidden1_93 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 140 \
    name hidden1_94 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_94 \
    op interface \
    ports { hidden1_94 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 141 \
    name hidden1_95 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_95 \
    op interface \
    ports { hidden1_95 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 142 \
    name hidden1_96 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_96 \
    op interface \
    ports { hidden1_96 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 143 \
    name hidden1_97 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_97 \
    op interface \
    ports { hidden1_97 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 144 \
    name hidden1_98 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_98 \
    op interface \
    ports { hidden1_98 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 145 \
    name hidden1_99 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_99 \
    op interface \
    ports { hidden1_99 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 146 \
    name hidden1_100 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_100 \
    op interface \
    ports { hidden1_100 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 147 \
    name hidden1_101 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_101 \
    op interface \
    ports { hidden1_101 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 148 \
    name hidden1_102 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_102 \
    op interface \
    ports { hidden1_102 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 149 \
    name hidden1_103 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_103 \
    op interface \
    ports { hidden1_103 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 150 \
    name hidden1_104 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_104 \
    op interface \
    ports { hidden1_104 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 151 \
    name hidden1_105 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_105 \
    op interface \
    ports { hidden1_105 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 152 \
    name hidden1_106 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_106 \
    op interface \
    ports { hidden1_106 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 153 \
    name hidden1_107 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_107 \
    op interface \
    ports { hidden1_107 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 154 \
    name hidden1_108 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_108 \
    op interface \
    ports { hidden1_108 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 155 \
    name hidden1_109 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_109 \
    op interface \
    ports { hidden1_109 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 156 \
    name hidden1_110 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_110 \
    op interface \
    ports { hidden1_110 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 157 \
    name hidden1_111 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_111 \
    op interface \
    ports { hidden1_111 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 158 \
    name hidden1_112 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_112 \
    op interface \
    ports { hidden1_112 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 159 \
    name hidden1_113 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_113 \
    op interface \
    ports { hidden1_113 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 160 \
    name hidden1_114 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_114 \
    op interface \
    ports { hidden1_114 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 161 \
    name hidden1_115 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_115 \
    op interface \
    ports { hidden1_115 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 162 \
    name hidden1_116 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_116 \
    op interface \
    ports { hidden1_116 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 163 \
    name hidden1_117 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_117 \
    op interface \
    ports { hidden1_117 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 164 \
    name hidden1_118 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_118 \
    op interface \
    ports { hidden1_118 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 165 \
    name hidden1_119 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_119 \
    op interface \
    ports { hidden1_119 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 166 \
    name hidden1_120 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_120 \
    op interface \
    ports { hidden1_120 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 167 \
    name hidden1_121 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_121 \
    op interface \
    ports { hidden1_121 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 168 \
    name hidden1_122 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_122 \
    op interface \
    ports { hidden1_122 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 169 \
    name hidden1_123 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_123 \
    op interface \
    ports { hidden1_123 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 170 \
    name hidden1_124 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_124 \
    op interface \
    ports { hidden1_124 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 171 \
    name hidden1_125 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_125 \
    op interface \
    ports { hidden1_125 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 172 \
    name hidden1_126 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_126 \
    op interface \
    ports { hidden1_126 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 173 \
    name hidden1_127 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden1_127 \
    op interface \
    ports { hidden1_127 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 174 \
    name sum_6_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_sum_6_out \
    op interface \
    ports { sum_6_out { O 24 vector } sum_6_out_ap_vld { O 1 bit } } \
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


