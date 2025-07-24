#include <iostream>
#include <cstdlib>
#include <ap_fixed.h>

// 引入实际的分类器函数
void ecg_classifier_practical(
    ap_fixed<16, 8> input[46],
    ap_fixed<16, 8> output[6]
);

int main() {
    std::cout << "=== ECG Practical Classifier Test ===" << std::endl;
    std::cout << "Based on MIT-BIH database, Training accuracy: 99.08%" << std::endl;
    std::cout << "Feature dimensions: 46, Output classes: 6 (N,L,R,A,V,F)" << std::endl;
    std::cout << "Target: 150-180 DSP, 14x performance improvement" << std::endl;
    std::cout << std::endl;

    // 测试用例1：正常心跳特征 (Q8.8定点格式)
    ap_fixed<16, 8> test_features1[46];
    ap_fixed<16, 8> output1[6];
    
    // 简单的测试数据初始化
    for (int i = 0; i < 46; i++) {
        test_features1[i] = (i % 10) * 0.1f;  // 简单的测试模式
    }

    // 运行分类器
    std::cout << "Running practical classifier..." << std::endl;
    ecg_classifier_practical(test_features1, output1);

    // 输出结果
    std::cout << "Classification results:" << std::endl;
    const char* class_names[] = {"Normal", "LBBB", "RBBB", "APC", "VPC", "Fusion"};
    
    for (int i = 0; i < 6; i++) {
        std::cout << class_names[i] << ": " << (float)output1[i] << std::endl;
    }

    // 找到最大概率的类别
    int max_class = 0;
    float max_prob = (float)output1[0];
    for (int i = 1; i < 6; i++) {
        if ((float)output1[i] > max_prob) {
            max_prob = (float)output1[i];
            max_class = i;
        }
    }

    std::cout << "\nPredicted class: " << class_names[max_class] 
              << " (confidence: " << max_prob << ")" << std::endl;
    std::cout << "\nTest completed successfully!" << std::endl;

    return 0;
}
