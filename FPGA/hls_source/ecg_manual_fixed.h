#ifndef ECG_MANUAL_FIXED_H
#define ECG_MANUAL_FIXED_H

#include <ap_int.h>

// 完全手工控制的固定点类型 - 避免任何可能的浮点生成
typedef ap_int<32> acc_t;      // 32位累加器 (避免ap_fixed内部转换)
typedef ap_int<16> data_t;     // 16位数据 (输入特征)
typedef ap_int<16> weight_t;   // 16位权重
typedef ap_int<16> bias_t;     // 16位偏置
typedef ap_int<6>  class_t;    // 6位输出类别

// 定点数缩放因子 (2^8 = 256, 相当于8位小数)
const int SCALE_FACTOR = 8;
const acc_t SCALE_MASK = (1 << SCALE_FACTOR) - 1;

// 手工实现的定点乘法 (避免ap_fixed内部运算)
inline acc_t fixed_mult(data_t a, weight_t b) {
    return ((acc_t)a * (acc_t)b);
}

// 手工实现的定点加法
inline acc_t fixed_add(acc_t a, bias_t b) {
    return a + ((acc_t)b << SCALE_FACTOR);
}

// 手工实现的ReLU激活函数
inline data_t relu_fixed(acc_t x) {
    // 右移8位恢复定点格式，然后应用ReLU
    acc_t shifted = x >> SCALE_FACTOR;
    return (shifted > 0) ? (data_t)shifted : (data_t)0;
}

// 手工实现的最大值查找
inline class_t find_max_class(acc_t outputs[6]) {
    acc_t max_val = outputs[0];
    class_t max_idx = 0;
    
    for (int i = 1; i < 6; i++) {
        if (outputs[i] > max_val) {
            max_val = outputs[i];
            max_idx = i;
        }
    }
    return max_idx;
}

#endif
