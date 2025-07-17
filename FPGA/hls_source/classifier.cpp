#include "classifier.h"
#include "weights.h"

// 完全手工控制的ECG分类器 - 零浮点依赖
void ecg_classify_manual_fixed(
    data_t features[46],      // 输入特征
    acc_t probabilities[6],   // 输出概率
    class_t* predicted_class  // 预测类别
) {
    #pragma HLS INTERFACE m_axi port=features offset=slave bundle=gmem0 depth=46
    #pragma HLS INTERFACE m_axi port=probabilities offset=slave bundle=gmem1 depth=6  
    #pragma HLS INTERFACE m_axi port=predicted_class offset=slave bundle=gmem2 depth=1
    #pragma HLS INTERFACE s_axilite port=features bundle=control
    #pragma HLS INTERFACE s_axilite port=probabilities bundle=control
    #pragma HLS INTERFACE s_axilite port=predicted_class bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 第一层: 46 -> 256
    data_t hidden1[256];
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete
    
    LAYER1_LOOP: for (int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        
        acc_t sum = 0;
        
        LAYER1_INNER: for (int j = 0; j < 46; j++) {
            #pragma HLS UNROLL factor=4
            sum += fixed_mult(features[j], dense_weights[i * 46 + j]);
        }
        
        sum = fixed_add(sum, dense_biases[i]);
        hidden1[i] = relu_fixed(sum);
    }
    
    // 第二层: 256 -> 128  
    data_t hidden2[128];
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete
    
    LAYER2_LOOP: for (int i = 0; i < 128; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4
        
        acc_t sum = 0;
        
        LAYER2_INNER: for (int j = 0; j < 256; j++) {
            #pragma HLS UNROLL factor=4
            sum += fixed_mult(hidden1[j], dense_1_weights[i * 256 + j]);
        }
        
        sum = fixed_add(sum, dense_1_biases[i]);
        hidden2[i] = relu_fixed(sum);
    }
    
    // 第三层: 128 -> 64
    data_t hidden3[64];
    #pragma HLS ARRAY_PARTITION variable=hidden3 complete
    
    LAYER3_LOOP: for (int i = 0; i < 64; i++) {
        #pragma HLS PIPELINE II=1  
        #pragma HLS UNROLL factor=4
        
        acc_t sum = 0;
        
        LAYER3_INNER: for (int j = 0; j < 128; j++) {
            #pragma HLS UNROLL factor=4
            sum += fixed_mult(hidden2[j], dense_2_weights[i * 128 + j]);
        }
        
        sum = fixed_add(sum, dense_2_biases[i]);
        hidden3[i] = relu_fixed(sum);
    }
    
    // 输出层: 64 -> 6
    acc_t outputs[6];
    
    OUTPUT_LOOP: for (int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=1
        
        acc_t sum = 0;
        
        OUTPUT_INNER: for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL factor=4
            sum += fixed_mult(hidden3[j], dense_3_weights[i * 64 + j]);
        }
        
        sum = fixed_add(sum, dense_3_biases[i]);
        outputs[i] = sum;  // 输出层不使用激活函数
    }
    
    // 写入输出概率
    PROB_COPY: for (int i = 0; i < 6; i++) {
        probabilities[i] = outputs[i];
    }
    
    // 找到最大概率对应的类别
    *predicted_class = find_max_class(outputs);
}
