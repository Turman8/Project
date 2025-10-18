#include "cnn_inference.hpp"

#include <hls_math.h>

namespace {

constexpr data_t kNegInf = data_t(-512.0f);

template <int IN_H, int IN_W, int IN_C,
          int OUT_H, int OUT_W, int OUT_C,
          int POOL_H, int POOL_W,
          int K_H, int K_W>
void conv_bn_relu_pool(
    const data_t input[IN_H][IN_W][IN_C],
    data_t output[POOL_H][POOL_W][OUT_C],
    const float weights[OUT_C * IN_C * K_H * K_W],
    const float biases[OUT_C],
    const float bn_scale[OUT_C],
    const float bn_offset[OUT_C]
) {
#pragma HLS INLINE
    const int pad_h = K_H / 2;
    const int pad_w = K_W / 2;

    for (int oc = 0; oc < OUT_C; ++oc) {
        data_t even_row_max[POOL_W];
#pragma HLS RESOURCE variable=even_row_max core=RAM_S2P_BRAM
        for (int pw = 0; pw < POOL_W; ++pw) {
#pragma HLS PIPELINE II=1
            even_row_max[pw] = kNegInf;
        }

        const data_t bias = data_t(biases[oc]);
        const data_t scale = data_t(bn_scale[oc]);
        const data_t offset = data_t(bn_offset[oc]);

        for (int h = 0; h < OUT_H; ++h) {
            data_t curr_row_max[POOL_W];
#pragma HLS RESOURCE variable=curr_row_max core=RAM_S2P_BRAM
            for (int pw = 0; pw < POOL_W; ++pw) {
#pragma HLS PIPELINE II=1
                curr_row_max[pw] = kNegInf;
            }

            for (int w = 0; w < OUT_W; ++w) {
#pragma HLS PIPELINE II=1
                data_t acc = bias;

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
                            acc += weight_val * input[in_h][in_w][ic];
                        }
                    }
                }

                data_t bn_out = acc * scale + offset;
                data_t relu_out = (bn_out > data_t(0)) ? bn_out : data_t(0);

                int pool_w_idx = w >> 1;
                if (relu_out > curr_row_max[pool_w_idx]) {
                    curr_row_max[pool_w_idx] = relu_out;
                }

                if ((h & 1) && (w & 1)) {
                    data_t pooled = even_row_max[pool_w_idx];
                    if (curr_row_max[pool_w_idx] > pooled) {
                        pooled = curr_row_max[pool_w_idx];
                    }
                    output[h >> 1][pool_w_idx][oc] = pooled;
                }
            }

            if ((h & 1) == 0) {
                for (int pw = 0; pw < POOL_W; ++pw) {
#pragma HLS PIPELINE II=1
                    if (curr_row_max[pw] > even_row_max[pw]) {
                        even_row_max[pw] = curr_row_max[pw];
                    }
                }
            } else {
                for (int pw = 0; pw < POOL_W; ++pw) {
#pragma HLS PIPELINE II=1
                    even_row_max[pw] = kNegInf;
                }
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
    static data_t pool1_out[POOL1_OUTPUT_HEIGHT][POOL1_OUTPUT_WIDTH][POOL1_OUTPUT_CHANNELS];
#pragma HLS RESOURCE variable=pool1_out core=RAM_S2P_BRAM
    static data_t pool2_out[POOL2_OUTPUT_HEIGHT][POOL2_OUTPUT_WIDTH][POOL2_OUTPUT_CHANNELS];
#pragma HLS RESOURCE variable=pool2_out core=RAM_S2P_BRAM
    static data_t pool3_out[POOL3_OUTPUT_HEIGHT][POOL3_OUTPUT_WIDTH][POOL3_OUTPUT_CHANNELS];
#pragma HLS RESOURCE variable=pool3_out core=RAM_S2P_BRAM
    static data_t dense_input[FLATTEN_SIZE];
    static data_t dense1_out[DENSE1_OUTPUTS];
    static data_t logits[NUM_CLASSES];

    conv_bn_relu_pool<INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
                      CONV1_OUTPUT_HEIGHT, CONV1_OUTPUT_WIDTH, CONV1_OUT_CHANNELS,
                      POOL1_OUTPUT_HEIGHT, POOL1_OUTPUT_WIDTH,
                      CONV1_KERNEL_H, CONV1_KERNEL_W>(
        input, pool1_out,
        CONV1_WEIGHTS, CONV1_BIASES,
        CONV1_BN_SCALE, CONV1_BN_OFFSET
    );

    conv_bn_relu_pool<POOL1_OUTPUT_HEIGHT, POOL1_OUTPUT_WIDTH, POOL1_OUTPUT_CHANNELS,
                      CONV2_OUTPUT_HEIGHT, CONV2_OUTPUT_WIDTH, CONV2_OUT_CHANNELS,
                      POOL2_OUTPUT_HEIGHT, POOL2_OUTPUT_WIDTH,
                      CONV2_KERNEL_H, CONV2_KERNEL_W>(
        pool1_out, pool2_out,
        CONV2_WEIGHTS, CONV2_BIASES,
        CONV2_BN_SCALE, CONV2_BN_OFFSET
    );

    conv_bn_relu_pool<POOL2_OUTPUT_HEIGHT, POOL2_OUTPUT_WIDTH, POOL2_OUTPUT_CHANNELS,
                      CONV3_OUTPUT_HEIGHT, CONV3_OUTPUT_WIDTH, CONV3_OUT_CHANNELS,
                      POOL3_OUTPUT_HEIGHT, POOL3_OUTPUT_WIDTH,
                      CONV3_KERNEL_H, CONV3_KERNEL_W>(
        pool2_out, pool3_out,
        CONV3_WEIGHTS, CONV3_BIASES,
        CONV3_BN_SCALE, CONV3_BN_OFFSET
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
