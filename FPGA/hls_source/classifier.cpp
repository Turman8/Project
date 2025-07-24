#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "weights.h"

// 定点数定义：Q8.8格式 (16位，8位整数，8位小数)
typedef ap_fixed<16, 8> fixed_t;

// 定点数乘法函数
inline fixed_t fixed_mult(fixed_t a, weight_t b) {
    #pragma HLS INLINE
    return a * fixed_t(b) / 256.0f;  // Q8.8 格式调整
}

// ReLU激活函数
inline fixed_t relu(fixed_t x) {
    #pragma HLS INLINE
    return (x > 0) ? x : fixed_t(0);
}

void ecg_classifier(
    fixed_t input[46],
    fixed_t output[6]
) {
    #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return

    // 中间缓冲区
    fixed_t buffer_a[128];
    fixed_t buffer_b[64];
    fixed_t buffer_c[32];

    #pragma HLS ARRAY_PARTITION variable=buffer_a cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=buffer_b cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=buffer_c cyclic factor=2 dim=1
    
    // 权重分割优化
    #pragma HLS ARRAY_PARTITION variable=dense_0_weights cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=dense_1_weights cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=dense_2_weights cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=dense_3_weights cyclic factor=2 dim=1

    // 第一层：46 -> 128 (保持相对高效)
    LAYER1: for (int i = 0; i < 128; i++) {
        #pragma HLS PIPELINE II=4
        
        fixed_t sum = fixed_t(dense_0_bias[i]) / 256.0f;
        
        LAYER1_MAC: for (int j = 0; j < 46; j++) {
            #pragma HLS UNROLL factor=2
            sum += fixed_mult(input[j], dense_0_weights[i * 46 + j]);
        }
        
        buffer_a[i] = relu(sum);
    }

    // 第二层：128 -> 64 (显著降低DSP使用)
    LAYER2: for (int i = 0; i < 64; i++) {
        #pragma HLS PIPELINE II=10  // 从6增加到10，大幅减少DSP
        
        fixed_t sum = fixed_t(dense_1_bias[i]) / 256.0f;
        
        LAYER2_MAC: for (int j = 0; j < 128; j++) {
            #pragma HLS UNROLL factor=2  // 从4降到2
            sum += fixed_mult(buffer_a[j], dense_1_weights[i * 128 + j]);
        }
        
        buffer_b[i] = relu(sum);
    }

    // 第三层：64 -> 32 (中等保守)
    LAYER3: for (int i = 0; i < 32; i++) {
        #pragma HLS PIPELINE II=6  // 从4增加到6
        
        fixed_t sum = fixed_t(dense_2_bias[i]) / 256.0f;
        
        LAYER3_MAC: for (int j = 0; j < 64; j++) {
            #pragma HLS UNROLL factor=2  // 保持为2
            sum += fixed_mult(buffer_b[j], dense_2_weights[i * 64 + j]);
        }
        
        buffer_c[i] = relu(sum);
    }

    // 第四层：32 -> 6 (保持相对高效，输出层小)
    LAYER4: for (int i = 0; i < 6; i++) {
        #pragma HLS PIPELINE II=3  // 从2增加到3
        
        fixed_t sum = fixed_t(dense_3_bias[i]) / 256.0f;
        
        LAYER4_MAC: for (int j = 0; j < 32; j++) {
            #pragma HLS UNROLL factor=4  // 从8降到4
            sum += fixed_mult(buffer_c[j], dense_3_weights[i * 32 + j]);
        }
        
        output[i] = sum;  // 输出层不需要ReLU
    }
}

/*
实用优化版本设计说明：

1. DSP减少策略：
   - 第二层 II: 6→10，UNROLL: 4→2 (预期DSP减少60%)
   - 第三层 II: 4→6 (预期DSP减少33%)
   - 第四层 UNROLL: 8→4 (预期DSP减少50%)
   
2. 预期资源使用：
   - 第一层: ~50 DSP (46×2÷4 ≈ 23×2)
   - 第二层: ~65 DSP (128×2÷10 ≈ 26×2.5)
   - 第三层: ~22 DSP (64×2÷6 ≈ 21×1)
   - 第四层: ~25 DSP (32×4÷3 ≈ 43)
   - 总计: ~162 DSP (目标范围内)

3. 性能预期：
   - 第一层: 128×4 = 512 cycles
   - 第二层: 64×10 = 640 cycles
   - 第三层: 32×6 = 192 cycles
   - 第四层: 6×3 = 18 cycles
   - 总延迟: ~1,362 cycles (vs 极简版本19,507 cycles)
   - 改进倍数: ~14.3x

4. 关键优化点：
   - 重点优化第二层（最大计算瓶颈）
   - 平衡各层的计算负载分配
   - 保持输入和输出层的相对高效
   - 使用适度的数组分割策略
*/
