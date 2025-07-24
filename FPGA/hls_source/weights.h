#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <ap_int.h>

// 定点数类型定义：Q8.8格式 (16位，8位整数，8位小数)
typedef ap_int<16> weight_t;

// 权重数组声明（外部定义在weights.cpp中）
extern const weight_t dense_0_weights[46 * 128];
extern const weight_t dense_0_bias[128];
extern const weight_t dense_1_weights[128 * 64];
extern const weight_t dense_1_bias[64];
extern const weight_t dense_2_weights[64 * 32];
extern const weight_t dense_2_bias[32];
extern const weight_t dense_3_weights[32 * 6];
extern const weight_t dense_3_bias[6];

#endif // WEIGHTS_FIXED_H
