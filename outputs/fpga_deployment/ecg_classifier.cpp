// ECG分类器HLS实现
#include "ap_int.h"
#include "ap_fixed.h"

#define INPUT_DIM 15
#define OUTPUT_DIM 1

typedef ap_fixed<16, 8> fixed_t;

// 简化的分类函数
void ecg_classify(fixed_t features[INPUT_DIM], int* result) {
    #pragma HLS INTERFACE m_axi port=features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=result bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return
    
    // 简单的阈值分类逻辑
    fixed_t peak_to_peak = features[2];
    fixed_t std_val = features[4];
    fixed_t energy = features[6];
    
    if (peak_to_peak > 2.5) {
        *result = 3;  // V类
    } else if (std_val > 1.2) {
        *result = 1;  // A类
    } else if (energy < 50) {
        *result = 2;  // L类
    } else {
        *result = 0;  // N类
    }
}
