#include "cnn_inference.hpp"

#include <fstream>
#include <iostream>

int main() {
    static data_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];
    static data_t output[NUM_CLASSES];

    // 简单示例：从文本加载输入或使用零输入
    for (int h = 0; h < INPUT_HEIGHT; ++h) {
        for (int w = 0; w < INPUT_WIDTH; ++w) {
            for (int c = 0; c < INPUT_CHANNELS; ++c) {
                input[h][w][c] = data_t(0);
            }
        }
    }

    ecg_cnn_inference(input, output);

    std::cout << "Inference probabilities:" << std::endl;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << i << ": " << output[i].to_float() << std::endl;
    }

    return 0;
}
