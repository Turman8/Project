#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "ap_fixed.h"

// Q8.8 定点数类型定义 (16位，8位整数，8位小数)
typedef ap_fixed<16, 8> weight_t;

// 网络结构常量
const int INPUT_DIM = 46;
const int HIDDEN1_DIM = 128;
const int HIDDEN2_DIM = 64;
const int HIDDEN3_DIM = 32;
const int OUTPUT_DIM = 6;

// 层1权重 (46x128) - Q8.8定点数格式
extern const weight_t layer1_weights[INPUT_DIM * HIDDEN1_DIM];
extern const weight_t layer1_biases[HIDDEN1_DIM];

// 层2权重 (128x64) - Q8.8定点数格式  
extern const weight_t layer2_weights[HIDDEN1_DIM * HIDDEN2_DIM];
extern const weight_t layer2_biases[HIDDEN2_DIM];

// 层3权重 (64x32) - Q8.8定点数格式
extern const weight_t layer3_weights[HIDDEN2_DIM * HIDDEN3_DIM];
extern const weight_t layer3_biases[HIDDEN3_DIM];

// 层4权重 (32x6) - Q8.8定点数格式
extern const weight_t layer4_weights[HIDDEN3_DIM * OUTPUT_DIM];
extern const weight_t layer4_biases[OUTPUT_DIM];

#endif // WEIGHTS_H
