// ECG分类器 - 基于真实MIT-BIH数据训练
// 训练准确率: 0.9910
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"
#include "weights.h"  // 包含训练好的权重

#define INPUT_DIM 46
#define HIDDEN1_DIM 128
#define HIDDEN2_DIM 64
#define HIDDEN3_DIM 32
#define OUTPUT_DIM 6

typedef ap_fixed<16, 8> fixed_t;
typedef ap_fixed<32, 16> acc_t;

// ReLU激活函数
fixed_t relu(fixed_t x) {
    return (x > fixed_t(0)) ? x : fixed_t(0);
}

// 主分类函数
void ecg_classify_trained(
    fixed_t features[INPUT_DIM], 
    fixed_t probabilities[OUTPUT_DIM], 
    int* predicted_class
) {
    #pragma HLS INTERFACE m_axi port=features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=probabilities bundle=gmem1
    #pragma HLS INTERFACE m_axi port=predicted_class bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return
    
    // 第一层 (46 -> 128)
    fixed_t hidden1[HIDDEN1_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete
    
    for(int i = 0; i < HIDDEN1_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_biases[i]);  // 添加偏置
        
        for(int j = 0; j < INPUT_DIM; j++) {
            #pragma HLS PIPELINE
            sum += features[j] * fixed_t(dense_weights[j * HIDDEN1_DIM + i]);  // 使用正确的权重访问
        }
        
        hidden1[i] = relu(sum);
    }
    
    // 第二层 (128 -> 64)
    fixed_t hidden2[HIDDEN2_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete
    
    for(int i = 0; i < HIDDEN2_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_1_biases[i]);  // 添加偏置
        
        for(int j = 0; j < HIDDEN1_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden1[j] * fixed_t(dense_1_weights[j * HIDDEN2_DIM + i]);  // 使用正确的权重访问
        }
        
        hidden2[i] = relu(sum);
    }
    
    // 第三层 (64 -> 32)
    fixed_t hidden3[HIDDEN3_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden3 complete
    
    for(int i = 0; i < HIDDEN3_DIM; i++) {
        #pragma HLS UNROLL factor=4
        acc_t sum = fixed_t(dense_2_biases[i]);  // 添加偏置
        
        for(int j = 0; j < HIDDEN2_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden2[j] * fixed_t(dense_2_weights[j * HIDDEN3_DIM + i]);  // 使用正确的权重访问
        }
        
        hidden3[i] = relu(sum);
    }
    
    // 输出层 (32 -> 6)
    fixed_t output[OUTPUT_DIM];
    fixed_t max_val = fixed_t(-1000);
    
    for(int i = 0; i < OUTPUT_DIM; i++) {
        #pragma HLS UNROLL
        acc_t sum = fixed_t(dense_3_biases[i]);  // 添加偏置
        
        for(int j = 0; j < HIDDEN3_DIM; j++) {
            #pragma HLS PIPELINE
            sum += hidden3[j] * fixed_t(dense_3_weights[j * OUTPUT_DIM + i]);  // 使用正确的权重访问
        }
        
        output[i] = fixed_t(sum);
        if(output[i] > max_val) max_val = output[i];
    }
    
    // 简化的Softmax
    fixed_t sum_exp = fixed_t(0);
    for(int i = 0; i < OUTPUT_DIM; i++) {
        probabilities[i] = hls::exp(output[i] - max_val);
        sum_exp += probabilities[i];
    }
    
    for(int i = 0; i < OUTPUT_DIM; i++) {
        probabilities[i] = probabilities[i] / sum_exp;
    }
    
    // 找出最大概率类别
    fixed_t max_prob = probabilities[0];
    *predicted_class = 0;
    
    for(int i = 1; i < OUTPUT_DIM; i++) {
        if(probabilities[i] > max_prob) {
            max_prob = probabilities[i];
            *predicted_class = i;
        }
    }
}
