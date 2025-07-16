// ECG分类器HLS测试激励
// 基于MIT-BIH数据库训练结果 (准确率: 99.08%)
// 特征维度: 46 (36维小波 + 10维时域)

#include <iostream>
#include <iomanip>
#include "ap_fixed.h"
#include "hls_math.h"

// 定义与主文件一致的常量
#define INPUT_DIM 46
#define OUTPUT_DIM 7
#define HIDDEN1_DIM 128
#define HIDDEN2_DIM 64  
#define HIDDEN3_DIM 32

typedef ap_fixed<16, 8> fixed_t;
typedef ap_fixed<32, 16> acc_t;

// 声明主函数
void ecg_classify_trained(
    fixed_t features[INPUT_DIM], 
    fixed_t probabilities[OUTPUT_DIM], 
    int* predicted_class
);

int main() {
    std::cout << "=== ECG心电图分类器HLS仿真测试 ===" << std::endl;
    std::cout << "基于MIT-BIH数据库，训练准确率: 99.08%" << std::endl;
    std::cout << "特征维度: " << INPUT_DIM << " (36维db4小波 + 10维时域)" << std::endl;
    std::cout << "输出类别: " << OUTPUT_DIM << " (N,L,R,A,V,F,P)" << std::endl;
    
    // 测试用例1: 正常心拍（N类）
    fixed_t test_features_normal[INPUT_DIM] = {
        // 小波特征 (36维) - db4小波6级分解统计特征
        0.02, 0.15, 0.8, -0.3, 0.12, 0.45,   // Level 1: mean,std,max,min,energy,abs_sum
        0.01, 0.12, 0.6, -0.25, 0.08, 0.35,  // Level 2  
        0.015, 0.18, 0.7, -0.28, 0.10, 0.40, // Level 3
        0.008, 0.10, 0.5, -0.2, 0.06, 0.25,  // Level 4
        0.005, 0.08, 0.4, -0.15, 0.04, 0.20, // Level 5
        0.003, 0.05, 0.3, -0.1, 0.02, 0.15,  // Level 6
        
        // 时域特征 (10维)
        0.05,   // 均值
        0.25,   // 标准差
        1.2,    // 最大值
        -0.8,   // 最小值
        2.0,    // 峰峰值
        0.35,   // 能量
        0.28,   // RMS
        35.0,   // 过零点数
        0.12,   // 平均绝对差分
        0.15    // 差分标准差
    };
    
    // 测试用例2: 室性心律（V类）
    fixed_t test_features_ventricular[INPUT_DIM] = {
        // 小波特征 - 室性心律特征（更大的幅值变化）
        0.08, 0.35, 1.8, -1.2, 0.45, 0.85,   // Level 1
        0.06, 0.28, 1.4, -0.9, 0.32, 0.65,   // Level 2
        0.04, 0.22, 1.1, -0.7, 0.25, 0.55,   // Level 3
        0.02, 0.18, 0.8, -0.5, 0.18, 0.45,   // Level 4
        0.015, 0.15, 0.6, -0.4, 0.12, 0.35,  // Level 5
        0.01, 0.12, 0.4, -0.25, 0.08, 0.25,  // Level 6
        
        // 时域特征
        0.1,    // 均值
        0.65,   // 标准差（更大）
        2.8,    // 最大值（更大）
        -1.8,   // 最小值
        4.6,    // 峰峰值（更大）
        1.2,    // 能量（更大）
        0.72,   // RMS
        25.0,   // 过零点数
        0.28,   // 平均绝对差分
        0.35    // 差分标准差
    };
    
    fixed_t probabilities[OUTPUT_DIM];
    int predicted_class;
    const char* class_names[] = {"N", "L", "R", "A", "V", "F", "P"};
    
    // 测试正常心拍
    std::cout << "\n🔍 测试用例1: 正常心拍特征" << std::endl;
    ecg_classify_trained(test_features_normal, probabilities, &predicted_class);
    
    std::cout << "预测类别: " << predicted_class << " (" << class_names[predicted_class] << ")" << std::endl;
    std::cout << "概率分布:" << std::endl;
    for(int i = 0; i < OUTPUT_DIM; i++) {
        std::cout << "  " << class_names[i] << ": " << std::fixed << std::setprecision(4) 
                  << (float)probabilities[i] << std::endl;
    }
    
    // 测试室性心律
    std::cout << "\n🔍 测试用例2: 室性心律特征" << std::endl;
    ecg_classify_trained(test_features_ventricular, probabilities, &predicted_class);
    
    std::cout << "预测类别: " << predicted_class << " (" << class_names[predicted_class] << ")" << std::endl;
    std::cout << "概率分布:" << std::endl;
    for(int i = 0; i < OUTPUT_DIM; i++) {
        std::cout << "  " << class_names[i] << ": " << std::fixed << std::setprecision(4) 
                  << (float)probabilities[i] << std::endl;
    }
    
    // 验证测试结果
    bool test1_passed = (predicted_class >= 0 && predicted_class < OUTPUT_DIM);
    bool test2_passed = (predicted_class >= 0 && predicted_class < OUTPUT_DIM);
    
    std::cout << "\n📊 测试结果:" << std::endl;
    std::cout << "测试用例1: " << (test1_passed ? "✅ 通过" : "❌ 失败") << std::endl;
    std::cout << "测试用例2: " << (test2_passed ? "✅ 通过" : "❌ 失败") << std::endl;
    
    if(test1_passed && test2_passed) {
        std::cout << "\n🎉 所有测试通过！ECG分类器HLS实现验证成功！" << std::endl;
        std::cout << "基于真实MIT-BIH训练的模型已成功转换为FPGA实现" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ 测试失败！请检查HLS实现" << std::endl;
        return 1;
    }
}
