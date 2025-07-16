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
    id 447 \
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
    id 316 \
    name sext_ln55_8 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln55_8 \
    op interface \
    ports { sext_ln55_8 { I 24 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 317 \
    name tmp_120 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_tmp_120 \
    op interface \
    ports { tmp_120 { I 4 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 318 \
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
    id 319 \
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
    id 320 \
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
    id 321 \
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
    id 322 \
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
    id 323 \
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
    id 324 \
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
    id 325 \
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
    id 326 \
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
    id 327 \
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
    id 328 \
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
    id 329 \
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
    id 330 \
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
    id 331 \
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
    id 332 \
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
    id 333 \
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
    id 334 \
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
    id 335 \
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
    id 336 \
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
    id 337 \
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
    id 338 \
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
    id 339 \
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
    id 340 \
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
    id 341 \
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
    id 342 \
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
    id 343 \
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
    id 344 \
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
    id 345 \
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
    id 346 \
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
    id 347 \
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
    id 348 \
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
    id 349 \
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
    id 350 \
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
    id 351 \
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
    id 352 \
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
    id 353 \
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
    id 354 \
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
    id 355 \
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
    id 356 \
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
    id 357 \
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
    id 358 \
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
    id 359 \
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
    id 360 \
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
    id 361 \
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
    id 362 \
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
    id 363 \
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
    id 364 \
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
    id 365 \
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
    id 366 \
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
    id 367 \
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
    id 368 \
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
    id 369 \
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
    id 370 \
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
    id 371 \
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
    id 372 \
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
    id 373 \
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
    id 374 \
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
    id 375 \
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
    id 376 \
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
    id 377 \
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
    id 378 \
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
    id 379 \
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
    id 380 \
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
    id 381 \
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
    id 382 \
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
    id 383 \
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
    id 384 \
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
    id 385 \
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
    id 386 \
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
    id 387 \
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
    id 388 \
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
    id 389 \
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
    id 390 \
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
    id 391 \
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
    id 392 \
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
    id 393 \
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
    id 394 \
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
    id 395 \
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
    id 396 \
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
    id 397 \
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
    id 398 \
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
    id 399 \
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
    id 400 \
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
    id 401 \
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
    id 402 \
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
    id 403 \
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
    id 404 \
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
    id 405 \
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
    id 406 \
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
    id 407 \
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
    id 408 \
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
    id 409 \
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
    id 410 \
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
    id 411 \
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
    id 412 \
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
    id 413 \
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
    id 414 \
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
    id 415 \
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
    id 416 \
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
    id 417 \
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
    id 418 \
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
    id 419 \
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
    id 420 \
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
    id 421 \
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
    id 422 \
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
    id 423 \
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
    id 424 \
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
    id 425 \
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
    id 426 \
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
    id 427 \
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
    id 428 \
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
    id 429 \
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
    id 430 \
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
    id 431 \
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
    id 432 \
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
    id 433 \
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
    id 434 \
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
    id 435 \
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
    id 436 \
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
    id 437 \
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
    id 438 \
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
    id 439 \
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
    id 440 \
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
    id 441 \
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
    id 442 \
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
    id 443 \
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
    id 444 \
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
    id 445 \
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
    id 446 \
    name sum_41_out \
    type other \
    dir O \
    reset_level 1 \
    sync_rst true \
    corename dc_sum_41_out \
    op interface \
    ports { sum_41_out { O 24 vector } sum_41_out_ap_vld { O 1 bit } } \
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


