#include "cnn_inference.hpp"

#include <hls_math.h>

namespace {

template <int IN_H, int IN_W, int IN_C,
          int OUT_H, int OUT_W, int OUT_C,
          int K_H, int K_W>
void conv_bn_relu(
    const data_t input[IN_H][IN_W][IN_C],
    data_t output[OUT_H][OUT_W][OUT_C],
    const float weights[OUT_C * IN_C * K_H * K_W],
    const float biases[OUT_C],
    const float bn_scale[OUT_C],
    const float bn_offset[OUT_C]
) {
#pragma HLS INLINE
    const int pad_h = K_H / 2;
    const int pad_w = K_W / 2;

    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w = 0; w < OUT_W; ++w) {
#pragma HLS PIPELINE II=1
                data_t acc = data_t(biases[oc]);

                for (int ic = 0; ic < IN_C; ++ic) {
                    for (int kh = 0; kh < K_H; ++kh) {
                        int in_h = h + kh - pad_h;
                        if (in_h < 0 || in_h >= IN_H) {
                            continue;
                        }
                        for (int kw = 0; kw < K_W; ++kw) {
                            int in_w = w + kw - pad_w;
                            if (in_w < 0 || in_w >= IN_W) {
                                continue;
                            }

                            int weight_idx = (((oc * IN_C) + ic) * K_H + kh) * K_W + kw;
                            data_t weight_val = data_t(weights[weight_idx]);
                            data_t input_val = input[in_h][in_w][ic];
                            acc += weight_val * input_val;
                        }
                    }
                }

                data_t scale = data_t(bn_scale[oc]);
                data_t offset = data_t(bn_offset[oc]);
                data_t bn_out = acc * scale + offset;
                output[h][w][oc] = (bn_out > data_t(0)) ? bn_out : data_t(0);
            }
        }
    }
}

template <int IN_H, int IN_W, int CHANNELS,
          int OUT_H, int OUT_W>
void maxpool2x2(
    const data_t input[IN_H][IN_W][CHANNELS],
    data_t output[OUT_H][OUT_W][CHANNELS]
) {
#pragma HLS INLINE
    for (int c = 0; c < CHANNELS; ++c) {
        for (int h = 0; h < OUT_H; ++h) {
            for (int w = 0; w < OUT_W; ++w) {
#pragma HLS PIPELINE II=1
                int base_h = h * 2;
                int base_w = w * 2;
                data_t max_val = data_t(-1e6);

                for (int kh = 0; kh < 2; ++kh) {
                    int in_h = base_h + kh;
                    if (in_h >= IN_H) {
                        continue;
                    }
                    for (int kw = 0; kw < 2; ++kw) {
                        int in_w = base_w + kw;
                        if (in_w >= IN_W) {
                            continue;
                        }
                        data_t val = input[in_h][in_w][c];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }

                output[h][w][c] = max_val;
            }
        }
    }
}

template <int IN_H, int IN_W, int CHANNELS, int OUT_SIZE>
void flatten(
    const data_t input[IN_H][IN_W][CHANNELS],
    data_t output[OUT_SIZE]
) {
#pragma HLS INLINE
    int idx = 0;
    for (int h = 0; h < IN_H; ++h) {
        for (int w = 0; w < IN_W; ++w) {
            for (int c = 0; c < CHANNELS; ++c) {
#pragma HLS PIPELINE II=1
                output[idx++] = input[h][w][c];
            }
        }
    }
}

template <int IN_DIM, int OUT_DIM, bool APPLY_RELU>
void dense_layer(
    const data_t input[IN_DIM],
    data_t output[OUT_DIM],
    const float weights[OUT_DIM * IN_DIM],
    const float biases[OUT_DIM]
) {
#pragma HLS INLINE
    for (int o = 0; o < OUT_DIM; ++o) {
#pragma HLS PIPELINE II=1
        data_t acc = data_t(biases[o]);
        for (int i = 0; i < IN_DIM; ++i) {
            int weight_idx = o * IN_DIM + i;
            acc += data_t(weights[weight_idx]) * input[i];
        }

        if (APPLY_RELU && acc < data_t(0)) {
            acc = data_t(0);
        }
        output[o] = acc;
    }
}

template <int DIM>
void softmax_layer(
    const data_t input[DIM],
    data_t output[DIM]
) {
#pragma HLS INLINE
    float max_val = input[0].to_float();
    for (int i = 1; i < DIM; ++i) {
        float val = input[i].to_float();
        if (val > max_val) {
            max_val = val;
        }
    }

    float exp_buffer[DIM];
#pragma HLS ARRAY_PARTITION variable=exp_buffer complete dim=1
    float sum_val = 0.0f;
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE II=1
        float shifted = input[i].to_float() - max_val;
        float e = hls::expf(shifted);
        exp_buffer[i] = e;
        sum_val += e;
    }

    float inv_sum = 1.0f / sum_val;
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = data_t(exp_buffer[i] * inv_sum);
    }
}

}  // namespace

void ecg_cnn_inference(
    const data_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],
    data_t output[NUM_CLASSES]
) {
#pragma HLS DATAFLOW
    static data_t stage0[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=stage0 complete dim=3

    for (int h = 0; h < INPUT_HEIGHT; ++h) {
        for (int w = 0; w < INPUT_WIDTH; ++w) {
            for (int c = 0; c < INPUT_CHANNELS; ++c) {
#pragma HLS PIPELINE II=1
                stage0[h][w][c] = input[h][w][c];
            }
        }
    }

    static data_t conv1_out[CONV1_OUTPUT_HEIGHT][CONV1_OUTPUT_WIDTH][CONV1_OUT_CHANNELS];
    static data_t pool1_out[POOL1_OUTPUT_HEIGHT][POOL1_OUTPUT_WIDTH][POOL1_OUTPUT_CHANNELS];
    static data_t conv2_out[CONV2_OUTPUT_HEIGHT][CONV2_OUTPUT_WIDTH][CONV2_OUT_CHANNELS];
    static data_t pool2_out[POOL2_OUTPUT_HEIGHT][POOL2_OUTPUT_WIDTH][POOL2_OUTPUT_CHANNELS];
    static data_t conv3_out[CONV3_OUTPUT_HEIGHT][CONV3_OUTPUT_WIDTH][CONV3_OUT_CHANNELS];
    static data_t pool3_out[POOL3_OUTPUT_HEIGHT][POOL3_OUTPUT_WIDTH][POOL3_OUTPUT_CHANNELS];
    static data_t dense_input[FLATTEN_SIZE];
    static data_t dense1_out[DENSE1_OUTPUTS];
    static data_t logits[NUM_CLASSES];

    conv_bn_relu<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
                 CONV1_OUTPUT_HEIGHT, CONV1_OUTPUT_WIDTH, CONV1_OUT_CHANNELS,
                 CONV1_KERNEL_H, CONV1_KERNEL_W>(
        stage0, conv1_out,
        CONV1_WEIGHTS, CONV1_BIASES,
        CONV1_BN_SCALE, CONV1_BN_OFFSET
    );

    maxpool2x2<CONV1_OUTPUT_HEIGHT, CONV1_OUTPUT_WIDTH, CONV1_OUT_CHANNELS,
               POOL1_OUTPUT_HEIGHT, POOL1_OUTPUT_WIDTH>(
        conv1_out, pool1_out
    );

    conv_bn_relu<POOL1_OUTPUT_HEIGHT, POOL1_OUTPUT_WIDTH, POOL1_OUTPUT_CHANNELS,
                 CONV2_OUTPUT_HEIGHT, CONV2_OUTPUT_WIDTH, CONV2_OUT_CHANNELS,
                 CONV2_KERNEL_H, CONV2_KERNEL_W>(
        pool1_out, conv2_out,
        CONV2_WEIGHTS, CONV2_BIASES,
        CONV2_BN_SCALE, CONV2_BN_OFFSET
    );

    maxpool2x2<CONV2_OUTPUT_HEIGHT, CONV2_OUTPUT_WIDTH, CONV2_OUT_CHANNELS,
               POOL2_OUTPUT_HEIGHT, POOL2_OUTPUT_WIDTH>(
        conv2_out, pool2_out
    );

    conv_bn_relu<POOL2_OUTPUT_HEIGHT, POOL2_OUTPUT_WIDTH, POOL2_OUTPUT_CHANNELS,
                 CONV3_OUTPUT_HEIGHT, CONV3_OUTPUT_WIDTH, CONV3_OUT_CHANNELS,
                 CONV3_KERNEL_H, CONV3_KERNEL_W>(
        pool2_out, conv3_out,
        CONV3_WEIGHTS, CONV3_BIASES,
        CONV3_BN_SCALE, CONV3_BN_OFFSET
    );

    maxpool2x2<CONV3_OUTPUT_HEIGHT, CONV3_OUTPUT_WIDTH, CONV3_OUT_CHANNELS,
               POOL3_OUTPUT_HEIGHT, POOL3_OUTPUT_WIDTH>(
        conv3_out, pool3_out
    );

    flatten<POOL3_OUTPUT_HEIGHT, POOL3_OUTPUT_WIDTH, POOL3_OUTPUT_CHANNELS,
            FLATTEN_SIZE>(
        pool3_out, dense_input
    );

    dense_layer<FLATTEN_SIZE, DENSE1_OUTPUTS, true>(
        dense_input, dense1_out,
        DENSE1_WEIGHTS, DENSE1_BIASES
    );

    dense_layer<DENSE1_OUTPUTS, NUM_CLASSES, false>(
        dense1_out, logits,
        DENSE2_WEIGHTS, DENSE2_BIASES
    );

    softmax_layer<NUM_CLASSES>(logits, output);
}
