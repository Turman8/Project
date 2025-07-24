"""
将浮点数权重转换为Q8.8定点数格式
"""
import numpy as np
import tensorflow as tf
import os

def float_to_q8_8(value):
    """
    将浮点数转换为Q8.8定点数 (16位整数表示)
    Q8.8格式: 8位整数部分 + 8位小数部分
    范围: -128.0 到 +127.996
    """
    # 限制范围到[-128, 127.996]
    value = np.clip(value, -128.0, 127.996)
    # 转换为Q8.8: 乘以256 (2^8)
    return int(value * 256)

def export_fixed_point_weights():
    """导出定点数权重到weights.cpp"""
    
    # 查找模型文件
    model_files = []
    if os.path.exists('outputs'):
        for file in os.listdir('outputs'):
            if file.endswith('.h5'):
                model_files.append(os.path.join('outputs', file))
    
    if not model_files:
        print("❌ 未找到模型文件")
        return
    
    # 使用最新的模型
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"🔧 加载模型: {latest_model}")
    
    model = tf.keras.models.load_model(latest_model)
    
    # 生成weights.cpp文件
    with open('FPGA/hls_source/weights.cpp', 'w') as f:
        f.write('#include "weights.h"\n\n')
        f.write('// Q8.8定点数权重数据 - 从训练模型自动生成\n')
        f.write('// 每个值为16位整数，表示范围[-128.0, +127.996]\n\n')
        
        layer_names = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                layer_name = f"layer{i//2 + 1}"  # 跳过dropout层
                layer_names.append(layer_name)
                
                # 权重矩阵
                if len(weights) > 0:
                    w = weights[0]
                    w_flat = w.flatten()
                    
                    print(f"  转换层 {layer_name}: {w.shape} -> {len(w_flat)} 个定点数")
                    
                    f.write(f"// {layer_name} 权重 {w.shape}\n")
                    f.write(f"const weight_t {layer_name}_weights[{len(w_flat)}] = {{\n")
                    
                    # 转换为定点数
                    for j, val in enumerate(w_flat):
                        fixed_val = float_to_q8_8(val)
                        if j % 16 == 0:
                            f.write("    ")
                        f.write(f"{fixed_val}")
                        if j < len(w_flat) - 1:
                            f.write(", ")
                        if (j + 1) % 16 == 0 or j == len(w_flat) - 1:
                            f.write("\n")
                    f.write("};\n\n")
                
                # 偏置向量
                if len(weights) > 1:
                    b = weights[1]
                    
                    print(f"  转换偏置 {layer_name}: {b.shape} -> {len(b)} 个定点数")
                    
                    f.write(f"// {layer_name} 偏置 {b.shape}\n")
                    f.write(f"const weight_t {layer_name}_biases[{len(b)}] = {{\n")
                    
                    # 转换为定点数
                    for j, val in enumerate(b):
                        fixed_val = float_to_q8_8(val)
                        if j % 16 == 0:
                            f.write("    ")
                        f.write(f"{fixed_val}")
                        if j < len(b) - 1:
                            f.write(", ")
                        if (j + 1) % 16 == 0 or j == len(b) - 1:
                            f.write("\n")
                    f.write("};\n\n")
        
        # 添加注释
        f.write("/*\n")
        f.write("定点数转换说明:\n")
        f.write("- 格式: Q8.8 (8位整数 + 8位小数)\n")
        f.write("- 范围: -128.0 到 +127.996\n")
        f.write("- 精度: 1/256 ≈ 0.0039\n")
        f.write("- 转换公式: fixed_value = int(float_value * 256)\n")
        f.write(f"- 生成时间: {model.summary()}\n")
        f.write("*/\n")
    
    print(f"✅ 定点数权重已生成到: FPGA/hls_source/weights.cpp")
    
    # 显示统计信息
    total_params = sum([np.prod(layer.get_weights()[0].shape) + len(layer.get_weights()[1]) 
                       for layer in model.layers 
                       if hasattr(layer, 'get_weights') and layer.get_weights()])
    print(f"📊 总参数数量: {total_params:,}")
    print(f"💾 存储大小: {total_params * 2:,} 字节 ({total_params * 2 / 1024:.1f} KB)")

if __name__ == "__main__":
    export_fixed_point_weights()
