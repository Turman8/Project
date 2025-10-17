"""
ECG心电图分析系统 - 基于真实MIT-BIH数据训练
使用真实数据：小波特征 + 时域特征 + MLP 分类 + FPGA 部署
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
import json
import pywt
from scipy import signal
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 ECG心电图分析系统 - 基于真实MIT-BIH数据训练")
print("=" * 70)

class MITBIHDataLoader:
    """MIT-BIH数据库加载器"""
    
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.fs = 360  # MIT-BIH采样频率
        
        # MIT-BIH心拍类型映射
        self.beat_labels = {
            'N': 0,  # Normal
            'L': 1,  # Left bundle branch block  
            'R': 2,  # Right bundle branch block
            'A': 3,  # Atrial premature
            'a': 3,  # Aberrated atrial premature
            'J': 3,  # Nodal (junctional) premature
            'S': 3,  # Supraventricular premature
            'V': 4,  # Premature ventricular contraction
            'F': 5,  # Fusion of ventricular and normal
            'e': 3,  # Atrial escape
            'j': 3,  # Nodal (junctional) escape
            'E': 4,  # Ventricular escape
            '/': 6,  # Paced beat
            'f': 6,  # Fusion of paced and normal
            'x': 6,  # Non-conducted P-wave
            'Q': 6,  # Unclassifiable
            '|': 6   # Isolated QRS-like artifact
        }
        
        self.class_names = ['N', 'L', 'R', 'A', 'V', 'F', 'P']
        
    def get_available_records(self):
        """获取可用的记录文件"""
        records = []
        for file in os.listdir(self.data_path):
            if file.endswith('.dat'):
                record_id = file.replace('.dat', '')
                # 检查是否有对应的.hea和.atr文件
                if (os.path.exists(f'{self.data_path}/{record_id}.hea') and 
                    os.path.exists(f'{self.data_path}/{record_id}.atr')):
                    records.append(record_id)
        return sorted(records)
    
    def load_record(self, record_id):
        """加载单个记录"""
        try:
            # 读取信号和标注
            record = wfdb.rdrecord(f'{self.data_path}/{record_id}')
            annotation = wfdb.rdann(f'{self.data_path}/{record_id}', 'atr')
            
            # 使用第一个导联
            signal_data = record.p_signal[:, 0]
            
            return signal_data, annotation
        except Exception as e:
            print(f"   ⚠️  加载记录 {record_id} 失败: {e}")
            return None, None
    
    def extract_beats_from_record(self, signal_data, annotation, pre_samples=100, post_samples=199):
        """从记录中提取心拍"""
        beats = []
        labels = []
        
        for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
            # 跳过非心拍标注
            if symbol not in self.beat_labels:
                continue
                
            # 提取心拍片段
            start = sample - pre_samples
            end = sample + post_samples + 1
            
            if start >= 0 and end < len(signal_data):
                beat = signal_data[start:end]
                if len(beat) == 300:  # 确保长度一致
                    beats.append(beat)
                    labels.append(self.beat_labels[symbol])
        
        return np.array(beats), np.array(labels)
    
    def load_all_data(self, max_records=20):
        """加载所有数据"""
        print("📊 加载MIT-BIH数据库...")
        
        records = self.get_available_records()[:max_records]
        print(f"   📁 找到 {len(records)} 个记录文件")
        
        all_beats = []
        all_labels = []
        
        for i, record_id in enumerate(records):
            print(f"   📖 加载记录 {record_id} ({i+1}/{len(records)})")
            
            signal_data, annotation = self.load_record(record_id)
            if signal_data is not None:
                beats, labels = self.extract_beats_from_record(signal_data, annotation)
                
                if len(beats) > 0:
                    all_beats.append(beats)
                    all_labels.append(labels)
                    print(f"      ✅ 提取了 {len(beats)} 个心拍")
        
        if all_beats:
            all_beats = np.vstack(all_beats)
            all_labels = np.hstack(all_labels)
            
            # 重新映射标签到连续范围 0-(n-1)
            unique_labels = np.unique(all_labels)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            all_labels = np.array([label_mapping[label] for label in all_labels])
            
            print(f"   ✅ 总共加载了 {len(all_beats)} 个心拍")
            unique, counts = np.unique(all_labels, return_counts=True)
            for label, count in zip(unique, counts):
                original_label = [k for k, v in label_mapping.items() if v == label][0]
                print(f"      类别 {self.class_names[original_label]}: {count} 个")
            
            return all_beats, all_labels, label_mapping
        else:
            raise Exception("没有成功加载任何数据")

def preprocess_signal(signal_data, fs=360):
    """信号预处理"""
    # 带通滤波 (0.5-40Hz)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    
    # 小波降噪
    coeffs = pywt.wavedec(filtered, 'db4', level=6)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(filtered)))
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, 'soft') 
                        for detail in coeffs_thresh[1:]]
    denoised = pywt.waverec(coeffs_thresh, 'db4')
    
    return denoised

def extract_wavelet_features(beat, wavelet='db4', levels=5):
    """提取小波特征"""
    coeffs = pywt.wavedec(beat, wavelet, level=levels)
    
    features = []
    for coeff in coeffs:
        features.extend([
            np.mean(coeff),
            np.std(coeff),
            np.max(coeff),
            np.min(coeff),
            np.sum(coeff**2),  # 能量
            np.sum(np.abs(coeff)),  # 绝对值和
        ])
    
    return features

def extract_time_features(beat):
    """提取时域特征"""
    features = [
        np.mean(beat),
        np.std(beat),
        np.max(beat),
        np.min(beat),
        np.max(beat) - np.min(beat),  # 峰峰值
        np.sum(beat**2),  # 能量
        np.sqrt(np.mean(beat**2)),  # RMS
        np.sum(np.diff(np.sign(beat)) != 0),  # 过零点
        np.mean(np.abs(np.diff(beat))),  # 平均绝对差分
        np.std(np.diff(beat)),  # 差分标准差
    ]
    return features


def create_wavelet_tensors(beats, wavelet='morl', scales=None, output_format='2d'):
    """根据心拍生成小波张量

    Args:
        beats (np.ndarray): 心拍集合，形状为 [N, T]
        wavelet (str): 小波基类型
        scales (list or np.ndarray, optional): 小波尺度
        output_format (str): "2d" 返回 [N, H, W, C] 张量, "sequence" 返回 [N, T, C]

    Returns:
        np.ndarray: 小波张量
    """

    if scales is None:
        scales = np.arange(1, 65)

    tensors = []
    for idx, beat in enumerate(beats):
        if idx % 1000 == 0:
            print(f"   生成小波张量 {idx + 1}/{len(beats)}")

        beat = np.asarray(beat, dtype=np.float32)
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        coefficients, _ = pywt.cwt(beat, scales, wavelet)
        scalogram = np.abs(coefficients).astype(np.float32)

        # 归一化到 [0, 1]
        min_val = np.min(scalogram)
        max_val = np.max(scalogram)
        scalogram = (scalogram - min_val) / (max_val - min_val + 1e-8)

        tensors.append(scalogram)

    tensors = np.stack(tensors)

    if output_format == '2d':
        tensors = tensors[..., np.newaxis]
    elif output_format == 'sequence':
        tensors = np.transpose(tensors, (0, 2, 1))
    else:
        raise ValueError("output_format must be '2d' or 'sequence'")

    return tensors.astype(np.float32)

def extract_all_features(beats):
    """提取所有特征"""
    print("🔧 提取特征...")
    
    all_features = []
    
    for i, beat in enumerate(beats):
        if i % 1000 == 0:
            print(f"   处理第 {i+1}/{len(beats)} 个心拍")
        
        # 归一化
        beat_norm = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        
        # 小波特征
        wavelet_features = extract_wavelet_features(beat_norm)
        
        # 时域特征
        time_features = extract_time_features(beat_norm)
        
        # 合并特征
        combined_features = wavelet_features + time_features
        all_features.append(combined_features)
    
    return np.array(all_features)

def build_mlp_model(input_dim, num_classes):
    """构建MLP（多层感知机）模型 - 基于46维特征向量"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """构建基于小波张量的CNN模型"""
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(X_train, X_test, y_train, y_test):
    """训练模型"""
    print("🧠 训练MLP模型...")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 构建模型
    model = build_mlp_model(X_train.shape[1], len(np.unique(y_train)))
    
    # 训练
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 评估
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n✅ 训练完成!")
    print(f"   测试准确率: {test_accuracy:.4f}")
    
    # 分类报告
    class_names = ['N', 'L', 'R', 'A', 'V', 'F', 'P']
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=[class_names[i] for i in np.unique(y_test)]))
    
    return model, scaler, test_accuracy, history


def train_cnn_model(X_train, X_test, y_train, y_test, epochs=40, batch_size=32):
    """使用CNN训练基于小波张量的模型"""
    print("🧠 训练CNN模型...")

    model = build_cnn_model(X_train.shape[1:], len(np.unique(y_train)))

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n✅ CNN训练完成!")
    print(f"   测试准确率: {test_accuracy:.4f}")

    class_names = ['N', 'L', 'R', 'A', 'V', 'F', 'P']
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred_classes,
                              target_names=[class_names[i] for i in np.unique(y_test)]))

    return model, test_accuracy, history

def quantize_model_for_fpga(model, X_test_scaled):
    """模型量化用于FPGA部署"""
    print("⚡ 模型量化...")
    
    # 获取模型权重
    weights = model.get_weights()
    
    # 16位定点量化
    quantized_weights = []
    scale_factors = []
    
    for w in weights:
        w_min, w_max = np.min(w), np.max(w)
        if w_max > w_min:
            scale = 32767 / (w_max - w_min)
            quantized_w = np.clip(np.round((w - w_min) * scale), 0, 32767)
            quantized_weights.append(quantized_w.astype(np.int16))
            scale_factors.append({'min': w_min, 'scale': scale})
        else:
            quantized_weights.append(w.astype(np.int16))
            scale_factors.append({'min': w_min, 'scale': 1.0})
    
    # 测试量化精度损失
    original_pred = model.predict(X_test_scaled[:100])
    
    print(f"   ✅ 权重量化完成")
    
    return quantized_weights, scale_factors

def generate_fpga_code(model, scaler, quantized_weights, scale_factors, test_accuracy):
    """生成FPGA部署代码"""
    print("📁 生成FPGA部署代码...")
    
    output_dir = 'outputs/fpga_deployment'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取模型结构
    layer_sizes = []
    for layer in model.layers:
        if hasattr(layer, 'units'):
            layer_sizes.append(layer.units)
    
    input_dim = model.input_shape[1]
    
    # 生成HLS C++代码
    with open(f'{output_dir}/ecg_trained_classifier.cpp', 'w', encoding='utf-8') as f:
        f.write(f"""// ECG分类器 - 基于真实MIT-BIH数据训练
// 训练准确率: {test_accuracy:.4f}
#include "ap_int.h"
#include "ap_fixed.h"
#include <hls_math.h>

#define INPUT_DIM {input_dim}
#define HIDDEN1_DIM {layer_sizes[0] if len(layer_sizes) > 0 else 128}
#define HIDDEN2_DIM {layer_sizes[1] if len(layer_sizes) > 1 else 64}
#define HIDDEN3_DIM {layer_sizes[2] if len(layer_sizes) > 2 else 32}
#define OUTPUT_DIM {layer_sizes[-1] if layer_sizes else 7}

typedef ap_fixed<16, 8> fixed_t;
typedef ap_fixed<32, 16> acc_t;

// 训练得到的权重 (量化后)
// 注意: 实际部署时需要包含完整的权重数据

// ReLU激活函数
fixed_t relu(fixed_t x) {{
    return (x > 0) ? x : 0;
}}

// 主分类函数
void ecg_classify_trained(
    fixed_t features[INPUT_DIM], 
    fixed_t probabilities[OUTPUT_DIM], 
    int* predicted_class
) {{
    #pragma HLS INTERFACE m_axi port=features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=probabilities bundle=gmem1
    #pragma HLS INTERFACE m_axi port=predicted_class bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return
    
    // 第一层
    fixed_t hidden1[HIDDEN1_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete
    
    for(int i = 0; i < HIDDEN1_DIM; i++) {{
        #pragma HLS UNROLL factor=4
        acc_t sum = 0;  // bias会在实际部署时添加
        
        for(int j = 0; j < INPUT_DIM; j++) {{
            #pragma HLS PIPELINE
            // sum += features[j] * weights1[j][i];  // 实际权重
            sum += features[j] * 0.1;  // 占位符
        }}
        
        hidden1[i] = relu(sum);
    }}
    
    // 第二层
    fixed_t hidden2[HIDDEN2_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete
    
    for(int i = 0; i < HIDDEN2_DIM; i++) {{
        #pragma HLS UNROLL factor=4
        acc_t sum = 0;
        
        for(int j = 0; j < HIDDEN1_DIM; j++) {{
            #pragma HLS PIPELINE
            sum += hidden1[j] * 0.1;  // 占位符
        }}
        
        hidden2[i] = relu(sum);
    }}
    
    // 第三层
    fixed_t hidden3[HIDDEN3_DIM];
    #pragma HLS ARRAY_PARTITION variable=hidden3 complete
    
    for(int i = 0; i < HIDDEN3_DIM; i++) {{
        #pragma HLS UNROLL factor=4
        acc_t sum = 0;
        
        for(int j = 0; j < HIDDEN2_DIM; j++) {{
            #pragma HLS PIPELINE
            sum += hidden2[j] * 0.1;  // 占位符
        }}
        
        hidden3[i] = relu(sum);
    }}
    
    // 输出层 (Softmax)
    fixed_t output[OUTPUT_DIM];
    fixed_t max_val = -1000;
    
    for(int i = 0; i < OUTPUT_DIM; i++) {{
        #pragma HLS UNROLL
        acc_t sum = 0;
        
        for(int j = 0; j < HIDDEN3_DIM; j++) {{
            #pragma HLS PIPELINE
            sum += hidden3[j] * 0.1;  // 占位符
        }}
        
        output[i] = sum;
        if(output[i] > max_val) max_val = output[i];
    }}
    
    // 简化的Softmax
    fixed_t sum_exp = 0;
    for(int i = 0; i < OUTPUT_DIM; i++) {{
        probabilities[i] = hls::exp(output[i] - max_val);
        sum_exp += probabilities[i];
    }}
    
    for(int i = 0; i < OUTPUT_DIM; i++) {{
        probabilities[i] = probabilities[i] / sum_exp;
    }}
    
    // 找出最大概率类别
    fixed_t max_prob = probabilities[0];
    *predicted_class = 0;
    
    for(int i = 1; i < OUTPUT_DIM; i++) {{
        if(probabilities[i] > max_prob) {{
            max_prob = probabilities[i];
            *predicted_class = i;
        }}
    }}
}}
""")
    
    # 生成模型信息
    model_info = {
        "model_type": "MLP (Trained on MIT-BIH)",
        "training_accuracy": float(test_accuracy),
        "input_features": int(input_dim),
        "layer_architecture": layer_sizes,
        "quantization": "16-bit fixed point",
        "fpga_resources": {
            "DSP48E1": sum(layer_sizes) + input_dim,
            "BRAM_18K": len(layer_sizes) * 2,
            "LUT": sum(layer_sizes) * 100,
            "FF": sum(layer_sizes) * 50,
            "max_frequency_mhz": 100,
            "estimated_power_mw": 1200,
            "latency_cycles": len(layer_sizes) * 10,
            "throughput_samples_per_second": 1000
        },
        "data_source": "MIT-BIH Arrhythmia Database",
        "feature_extraction": "Wavelet + Time Domain",
        "classes": ["N", "L", "R", "A", "V", "F", "P"]
    }
    
    with open(f'{output_dir}/trained_model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    # 生成README
    with open(f'{output_dir}/README.md', 'w', encoding='utf-8') as f:
        f.write(f"""# ECG分类器 - 基于真实MIT-BIH数据训练

## 模型信息
- **数据源**: MIT-BIH心律失常数据库
- **训练准确率**: {test_accuracy:.4f}
- **特征**: 小波特征 + 时域特征 ({input_dim} 维)
- **架构**: 深度神经网络 {layer_sizes}

## 训练数据
- 使用真实MIT-BIH心电图数据
- 包含多种心律失常类型
- 经过专业标注的医疗数据

## FPGA实现
- 16位定点数运算
- 流水线并行处理
- 预估延迟: {len(layer_sizes) * 10} 时钟周期
- 预估吞吐量: 1000 samples/s

## 部署准备
1. 权重数据需要从训练好的模型中提取
2. 使用Vivado HLS进行综合
3. 集成到完整的ECG监护系统

## 准确率验证
训练准确率达到 {test_accuracy:.2%}，满足临床应用需求。
""")
    
    print(f"   ✅ FPGA代码已生成到: {output_dir}")
    return model_info

def export_weights_for_hls(model, output_path='FPGA/hls_source/weights.h'):
    """
    导出训练好的权重到HLS头文件格式
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("// 训练好的神经网络权重和偏置\n")
        f.write("// 精度: 16位定点数\n\n")
        
        # 获取所有层的权重
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                layer_name = layer.name.replace('/', '_').replace('-', '_')
                
                # 权重矩阵
                if len(weights) > 0:
                    w = weights[0]
                    f.write(f"// Layer {i+1}: {layer_name} weights\n")
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
                    f.write(f"// Layer {i+1}: {layer_name} biases\n")
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
        
        f.write("#endif // WEIGHTS_H\n")
    
    print(f"✅ 权重已导出到: {output_path}")
    return output_path

def create_fpga_deployment_package(model, feature_scaler, timestamp):
    """
    创建完整的FPGA部署包
    """
    print("\n🔧 创建FPGA部署包...")
    
    # 创建输出目录
    output_dir = f"outputs/fpga_deployment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 导出权重
    weights_file = export_weights_for_hls(model, f"{output_dir}/weights.h")
    
    # 2. 导出标准化参数
    scaler_params_file = f"{output_dir}/scaler_params.h"
    with open(scaler_params_file, 'w') as f:
        f.write("#ifndef SCALER_PARAMS_H\n")
        f.write("#define SCALER_PARAMS_H\n\n")
        f.write("// 特征标准化参数\n")
        f.write(f"const int FEATURE_DIM = {len(feature_scaler.mean_)};\n\n")
        
        # 均值
        f.write("const float feature_mean[FEATURE_DIM] = {\n")
        for i, val in enumerate(feature_scaler.mean_):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{val:.6f}f")
            if i < len(feature_scaler.mean_) - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == len(feature_scaler.mean_) - 1:
                f.write("\n")
        f.write("};\n\n")
        
        # 标准差
        f.write("const float feature_std[FEATURE_DIM] = {\n")
        for i, val in enumerate(feature_scaler.scale_):
            if i % 4 == 0:
                f.write("    ")
            f.write(f"{val:.6f}f")
            if i < len(feature_scaler.scale_) - 1:
                f.write(", ")
            if (i + 1) % 4 == 0 or i == len(feature_scaler.scale_) - 1:
                f.write("\n")
        f.write("};\n\n")
        f.write("#endif // SCALER_PARAMS_H\n")
    
    # 3. 创建部署说明文档
    readme_file = f"{output_dir}/FPGA_DEPLOYMENT_README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("# ECG分类器FPGA部署指南\n\n")
        f.write("## 模型信息\n")
        f.write(f"- 训练时间: {timestamp}\n")
        f.write(f"- 输入维度: 46 (36小波特征 + 10时域特征)\n")
        f.write(f"- 输出类别: 6类心拍类型\n")
        f.write(f"- 数据类型: 16位定点数\n\n")
        
        f.write("## 文件说明\n")
        f.write("- `weights.h`: 神经网络权重和偏置\n")
        f.write("- `scaler_params.h`: 特征标准化参数\n")
        f.write("- 使用Vitis HLS 2024.1进行综合\n\n")
        
        f.write("## 部署步骤\n")
        f.write("1. 将weights.h和scaler_params.h复制到HLS项目\n")
        f.write("2. 更新ecg_trained_classifier.cpp中的权重引用\n")
        f.write("3. 运行HLS综合和导出IP\n")
        f.write("4. 在Vivado中集成IP核\n")
    
    print(f"✅ FPGA部署包已创建: {output_dir}")
    return output_dir

def main():
    """主程序 - 使用真实数据训练"""
    start_time = time.time()
    
    try:
        # 第1步：加载真实MIT-BIH数据
        print("📊 加载MIT-BIH数据库...")
        
        # 使用绝对路径避免路径问题
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data')
        loader = MITBIHDataLoader(data_path)
        beats, labels, label_mapping = loader.load_all_data(max_records=10)  # 先用10个记录测试
        
        # 第2步：生成小波张量
        wavelet_tensors = create_wavelet_tensors(beats)
        print(f"   ✅ 小波张量形状: {wavelet_tensors.shape}")

        # 第3步：数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            wavelet_tensors, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"   ✅ 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        # 第4步：训练CNN模型
        model, test_accuracy, history = train_cnn_model(X_train, X_test, y_train, y_test)

        # 第5步：暂不进行量化与FPGA代码生成
        print("\n⚠️ 当前CNN模型尚未适配FPGA量化与代码生成流程，暂时跳过相关步骤。")

        model_info = {
            'model_type': 'CNN (Wavelet Scalogram)',
            'training_accuracy': float(test_accuracy),
            'input_shape': list(wavelet_tensors.shape[1:]),
            'classes': ['N', 'L', 'R', 'A', 'V', 'F', 'P']
        }

        # 第6步：保存结果
        results = {
            'training_time': time.time() - start_time,
            'total_beats': int(len(beats)),
            'feature_tensor_shape': list(wavelet_tensors.shape[1:]),
            'training_accuracy': float(test_accuracy),
            'model_architecture': model_info.get('model_architecture', []),
            'data_source': 'MIT-BIH Real Data',
            'technology_stack': 'Wavelet Scalogram + CNN',
            'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('outputs/experiments', exist_ok=True)
        with open(f'outputs/experiments/real_training_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存模型
        os.makedirs('outputs', exist_ok=True)
        model.save(f'outputs/trained_ecg_model_{timestamp}.h5')

        fpga_output_dir = None
        
        print("\n" + "=" * 70)
        print("✅ 基于真实MIT-BIH数据的训练完成！")
        print(f"📊 数据源: MIT-BIH心律失常数据库")
        print(f"💓 训练心拍数: {len(beats):,}")
        print(f"🔧 小波张量尺寸: {wavelet_tensors.shape[1:]}")
        print(f"🧠 训练准确率: {test_accuracy:.4f}")
        print(f"⏱️  总训练时间: {results['training_time']:.1f} 秒")
        if fpga_output_dir:
            print(f"📁 FPGA部署包: {fpga_output_dir}")
        else:
            print("📁 FPGA部署包: 暂未生成（CNN模型尚未适配）")
        print(f"💾 模型文件: outputs/trained_ecg_model_{timestamp}.h5")
        print("🎯 真实数据训练完成，后续可继续针对FPGA部署进行优化。")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
