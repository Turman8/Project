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
    id 728 \
    name dense_2_weights \
    reset_level 1 \
    sync_rst true \
    dir I \
    corename dense_2_weights \
    op interface \
    ports { dense_2_weights_address0 { O 11 vector } dense_2_weights_ce0 { O 1 bit } dense_2_weights_q0 { I 32 vector } } \
} "
} else {
puts "@W \[IMPL-110\] Cannot find bus interface model in the library. Ignored generation of bus interface for 'dense_2_weights'"
}
}


# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 661 \
    name sext_ln71_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln71_5 \
    op interface \
    ports { sext_ln71_5 { I 24 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 662 \
    name tmp_112 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_tmp_112 \
    op interface \
    ports { tmp_112 { I 4 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 663 \
    name hidden2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2 \
    op interface \
    ports { hidden2 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 664 \
    name hidden2_1 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_1 \
    op interface \
    ports { hidden2_1 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 665 \
    name hidden2_2 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_2 \
    op interface \
    ports { hidden2_2 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 666 \
    name hidden2_3 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_3 \
    op interface \
    ports { hidden2_3 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 667 \
    name hidden2_4 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_4 \
    op interface \
    ports { hidden2_4 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 668 \
    name hidden2_5 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_5 \
    op interface \
    ports { hidden2_5 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 669 \
    name hidden2_6 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_6 \
    op interface \
    ports { hidden2_6 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 670 \
    name hidden2_7 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_7 \
    op interface \
    ports { hidden2_7 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 671 \
    name hidden2_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_8 \
    op interface \
    ports { hidden2_8 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 672 \
    name hidden2_9 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_9 \
    op interface \
    ports { hidden2_9 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 673 \
    name hidden2_10 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_10 \
    op interface \
    ports { hidden2_10 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 674 \
    name hidden2_11 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_11 \
    op interface \
    ports { hidden2_11 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 675 \
    name hidden2_12 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_12 \
    op interface \
    ports { hidden2_12 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 676 \
    name hidden2_13 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_13 \
    op interface \
    ports { hidden2_13 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 677 \
    name hidden2_14 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_14 \
    op interface \
    ports { hidden2_14 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 678 \
    name hidden2_15 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_15 \
    op interface \
    ports { hidden2_15 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 679 \
    name hidden2_16 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_16 \
    op interface \
    ports { hidden2_16 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 680 \
    name hidden2_17 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_17 \
    op interface \
    ports { hidden2_17 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 681 \
    name hidden2_18 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_18 \
    op interface \
    ports { hidden2_18 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 682 \
    name hidden2_19 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_19 \
    op interface \
    ports { hidden2_19 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 683 \
    name hidden2_20 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_20 \
    op interface \
    ports { hidden2_20 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 684 \
    name hidden2_21 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_21 \
    op interface \
    ports { hidden2_21 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 685 \
    name hidden2_22 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_22 \
    op interface \
    ports { hidden2_22 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 686 \
    name hidden2_23 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_23 \
    op interface \
    ports { hidden2_23 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 687 \
    name hidden2_24 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_24 \
    op interface \
    ports { hidden2_24 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 688 \
    name hidden2_25 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_25 \
    op interface \
    ports { hidden2_25 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 689 \
    name hidden2_26 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_26 \
    op interface \
    ports { hidden2_26 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 690 \
    name hidden2_27 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_27 \
    op interface \
    ports { hidden2_27 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 691 \
    name hidden2_28 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_28 \
    op interface \
    ports { hidden2_28 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 692 \
    name hidden2_29 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_29 \
    op interface \
    ports { hidden2_29 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 693 \
    name hidden2_30 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_30 \
    op interface \
    ports { hidden2_30 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 694 \
    name hidden2_31 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_31 \
    op interface \
    ports { hidden2_31 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 695 \
    name hidden2_32 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_32 \
    op interface \
    ports { hidden2_32 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 696 \
    name hidden2_33 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_33 \
    op interface \
    ports { hidden2_33 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 697 \
    name hidden2_34 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_34 \
    op interface \
    ports { hidden2_34 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 698 \
    name hidden2_35 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_35 \
    op interface \
    ports { hidden2_35 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 699 \
    name hidden2_36 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_36 \
    op interface \
    ports { hidden2_36 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 700 \
    name hidden2_37 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_37 \
    op interface \
    ports { hidden2_37 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 701 \
    name hidden2_38 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_38 \
    op interface \
    ports { hidden2_38 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 702 \
    name hidden2_39 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_39 \
    op interface \
    ports { hidden2_39 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 703 \
    name hidden2_40 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_40 \
    op interface \
    ports { hidden2_40 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 704 \
    name hidden2_41 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_41 \
    op interface \
    ports { hidden2_41 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 705 \
    name hidden2_42 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_42 \
    op interface \
    ports { hidden2_42 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 706 \
    name hidden2_43 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_43 \
    op interface \
    ports { hidden2_43 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 707 \
    name hidden2_44 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_44 \
    op interface \
    ports { hidden2_44 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 708 \
    name hidden2_45 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_45 \
    op interface \
    ports { hidden2_45 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 709 \
    name hidden2_46 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_46 \
    op interface \
    ports { hidden2_46 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 710 \
    name hidden2_47 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_47 \
    op interface \
    ports { hidden2_47 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 711 \
    name hidden2_48 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_48 \
    op interface \
    ports { hidden2_48 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 712 \
    name hidden2_49 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_49 \
    op interface \
    ports { hidden2_49 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 713 \
    name hidden2_50 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_50 \
    op interface \
    ports { hidden2_50 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 714 \
    name hidden2_51 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_51 \
    op interface \
    ports { hidden2_51 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 715 \
    name hidden2_52 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_52 \
    op interface \
    ports { hidden2_52 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 716 \
    name hidden2_53 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_53 \
    op interface \
    ports { hidden2_53 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 717 \
    name hidden2_54 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_54 \
    op interface \
    ports { hidden2_54 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 718 \
    name hidden2_55 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_55 \
    op interface \
    ports { hidden2_55 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 719 \
    name hidden2_56 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_56 \
    op interface \
    ports { hidden2_56 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 720 \
    name hidden2_57 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_57 \
    op interface \
    ports { hidden2_57 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 721 \
    name hidden2_58 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_58 \
    op interface \
    ports { hidden2_58 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 722 \
    name hidden2_59 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_59 \
    op interface \
    ports { hidden2_59 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 723 \
    name hidden2_60 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_60 \
    op interface \
    ports { hidden2_60 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 724 \
    name hidden2_61 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_61 \
    op interface \
    ports { hidden2_61 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 725 \
    name hidden2_62 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_62 \
    op interface \
    ports { hidden2_62 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 726 \
    name hidden2_63 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_hidden2_63 \
    op interface \
    ports { hidden2_63 { I 16 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 727 \
    name sum_26_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_sum_26_out \
    op interface \
    ports { sum_26_out { O 24 vector } sum_26_out_ap_vld { O 1 bit } } \
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


