#include <iostream>
#include <ap_fixed.h>

// 定点数定义：Q8.8格式 (16位，8位整数，8位小数)
typedef ap_fixed<16, 8> fixed_t;

// 外部函数声明
void ecg_classifier(fixed_t input[46], fixed_t output[6]);

int main() {
    std::cout << "=== ECG分类器定点数测试 ===" << std::endl;
    std::cout << "架构: 46->128->64->32->6 (MIT-BIH训练)" << std::endl;
    std::cout << "精度: Q8.8定点数 (16位)" << std::endl;
    std::cout << "目标: DSP<180, 延迟<1500周期, 99%+准确率" << std::endl;
    std::cout << std::endl;

    // 测试输入数据
    fixed_t input[46];
    fixed_t output[6];
    
    // 初始化测试数据 (模拟正常心跳特征)
    for (int i = 0; i < 46; i++) {
        input[i] = fixed_t(0.1 * (i % 10 - 5));  // 范围 [-0.5, 0.4]
    }
    
    std::cout << "输入特征 (前10个): ";
    for (int i = 0; i < 10; i++) {
        std::cout << input[i].to_double() << " ";
    }
    std::cout << "..." << std::endl;
    std::cout << std::endl;
    
    // 执行分类
    std::cout << "执行定点数分类..." << std::endl;
    ecg_classifier(input, output);
    
    // 显示结果
    std::cout << "分类结果:" << std::endl;
    const char* classes[] = {"正常(N)", "左束支阻滞(L)", "右束支阻滞(R)", 
                            "房性早搏(A)", "室性早搏(V)", "融合心跳(F)"};
    
    int max_idx = 0;
    double max_val = output[0].to_double();
    
    for (int i = 0; i < 6; i++) {
        double val = output[i].to_double();
        std::cout << "  " << classes[i] << ": " << val << std::endl;
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    std::cout << std::endl;
    std::cout << "预测类别: " << classes[max_idx] << " (置信度: " << max_val << ")" << std::endl;
    std::cout << "定点数测试完成!" << std::endl;
    
    return 0;
}
