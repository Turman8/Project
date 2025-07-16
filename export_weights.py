"""
从已训练的模型中导出权重给FPGA使用
"""

import numpy as np
import os
import tensorflow as tf
from datetime import datetime

def export_weights_for_hls(model_path, output_path='FPGA/hls_source/weights.h'):
    """
    从保存的模型导出权重到HLS头文件格式
    """
    print(f"🔄 加载模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("// 训练好的神经网络权重和偏置\n")
        f.write("// 精度: 32位浮点数 (后续可量化为16位定点)\n\n")
        
        # 获取所有层的权重
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                layer_name = layer.name.replace('/', '_').replace('-', '_')
                
                print(f"  导出层 {i+1}: {layer_name}")
                
                # 权重矩阵
                if len(weights) > 0:
                    w = weights[0]
                    f.write(f"// Layer {i+1}: {layer_name} weights [{w.shape}]\n")
                    f.write(f"const float {layer_name}_weights[{w.size}] = {{\n")
                    
                    # 展平权重并写入
                    w_flat = w.flatten()
                    for j, val in enumerate(w_flat):
                        if j % 8 == 0:
                            f.write("    ")
                        f.write(f"{val:.6f}f")
                        if j < len(w_flat) - 1:
                            f.write(", ")
                        if (j + 1) % 8 == 0 or j == len(w_flat) - 1:
                            f.write("\n")
                    f.write("};\n\n")
                    
                    # 权重维度信息
                    f.write(f"const int {layer_name}_weights_rows = {w.shape[0]};\n")
                    f.write(f"const int {layer_name}_weights_cols = {w.shape[1] if len(w.shape) > 1 else 1};\n\n")
                
                # 偏置向量
                if len(weights) > 1:
                    b = weights[1]
                    f.write(f"// Layer {i+1}: {layer_name} biases [{b.shape}]\n")
                    f.write(f"const float {layer_name}_biases[{b.size}] = {{\n")
                    
                    for j, val in enumerate(b):
                        if j % 8 == 0:
                            f.write("    ")
                        f.write(f"{val:.6f}f")
                        if j < len(b) - 1:
                            f.write(", ")
                        if (j + 1) % 8 == 0 or j == len(b) - 1:
                            f.write("\n")
                    f.write("};\n\n")
        
        # 添加网络结构信息
        f.write("// 网络结构信息\n")
        f.write(f"const int INPUT_DIM = 46;  // 输入特征维度\n")
        f.write(f"const int OUTPUT_DIM = 6;  // 输出类别数\n")
        f.write(f"const int NUM_LAYERS = {len([l for l in model.layers if hasattr(l, 'get_weights') and l.get_weights()])};\n\n")
        
        f.write("#endif // WEIGHTS_H\n")
    
    print(f"✅ 权重已导出到: {output_path}")
    
    # 显示模型摘要
    print(f"\n📊 模型结构:")
    model.summary()
    
    return output_path

def main():
    # 查找最新的模型文件
    model_files = []
    if os.path.exists('outputs'):
        for file in os.listdir('outputs'):
            if file.startswith('trained_ecg_model_') and file.endswith('.h5'):
                model_files.append(os.path.join('outputs', file))
    
    if not model_files:
        print("❌ 未找到训练好的模型文件")
        return
    
    # 使用最新的模型文件
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"🎯 使用模型: {latest_model}")
    
    # 导出权重
    weights_file = export_weights_for_hls(latest_model)
    
    print(f"\n✅ 权重导出完成！")
    print(f"📁 输出文件: {weights_file}")
    print(f"🔧 下一步: 更新FPGA/hls_source/ecg_trained_classifier.cpp")

if __name__ == "__main__":
    main()
