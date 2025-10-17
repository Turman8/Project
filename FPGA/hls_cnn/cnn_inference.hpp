#pragma once

#include <ap_fixed.h>
#include "cnn_weights.h"

using data_t = ap_fixed<CNN_HLS_TOTAL_BITS, CNN_HLS_INTEGER_BITS>;

void ecg_cnn_inference(
    const data_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],
    data_t output[NUM_CLASSES]
);
