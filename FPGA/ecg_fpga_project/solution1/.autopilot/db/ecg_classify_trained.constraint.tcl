set clock_constraint { \
    name clk \
    module ecg_classify_trained \
    port ap_clk \
    period 10 \
    uncertainty 2.7 \
}

set all_path {}

set false_path {}

