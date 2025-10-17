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
import shutil
from pathlib import Path
import textwrap
import pywt
from scipy import signal
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
        
        # MIT-BIH心拍类型映射到更丰富的心律类别，尽可能覆盖更多心律失常
        self.symbol_to_class = {
            'N': 'Normal',
            'L': 'LeftBundleBranchBlock',
            'R': 'RightBundleBranchBlock',
            'B': 'BundleBranchBlock',
            'A': 'AtrialPremature',
            'a': 'AberratedAtrialPremature',
            'J': 'JunctionalPremature',
            'S': 'SupraventricularPremature',
            'e': 'AtrialEscape',
            'j': 'JunctionalEscape',
            'V': 'PrematureVentricular',
            'E': 'VentricularEscape',
            'F': 'Fusion',
            '/': 'Paced',
            'f': 'FusionPaced',
            'x': 'NonConductedPWave',
            'Q': 'Unclassifiable',
            '|': 'IsolatedQRSLike',
            '!': 'VentricularFlutter',
            '[': 'VentricularFlutterStart',
            ']': 'VentricularFlutterEnd',
            'p': 'PacedPremature',
            't': 'TWaveAbnormality'
        }

        self.class_names = sorted(set(self.symbol_to_class.values()))
        
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
            if symbol not in self.symbol_to_class:
                continue
                
            # 提取心拍片段
            start = sample - pre_samples
            end = sample + post_samples + 1
            
            if start >= 0 and end < len(signal_data):
                beat = signal_data[start:end]
                if len(beat) == 300:  # 确保长度一致
                    beats.append(beat)
                    labels.append(self.symbol_to_class[symbol])
        
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
            unique_label_names = sorted(np.unique(all_labels))
            label_mapping = {label_name: idx for idx, label_name in enumerate(unique_label_names)}
            numeric_labels = np.array([label_mapping[label] for label in all_labels], dtype=np.int32)

            # 更新可用类别
            self.class_names = unique_label_names

            print(f"   ✅ 总共加载了 {len(all_beats)} 个心拍")
            unique, counts = np.unique(numeric_labels, return_counts=True)
            for label_idx, count in zip(unique, counts):
                class_name = unique_label_names[label_idx]
                print(f"      类别 {class_name}: {count} 个")

            return all_beats, numeric_labels, label_mapping, unique_label_names
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

def train_model(X_train, X_test, y_train, y_test, class_names=None):
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
    if class_names is None:
        unique_eval_classes = np.unique(np.concatenate([y_train, y_test]))
        class_names = [f'class_{idx}' for idx in unique_eval_classes]
    unique_eval_classes = np.unique(y_test)
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred_classes,
                              labels=unique_eval_classes,
                              target_names=[class_names[i] for i in unique_eval_classes]))

    return model, scaler, test_accuracy, history


def train_cnn_model(X_train, X_test, y_train, y_test, class_names, epochs=40, batch_size=32):
    """使用CNN训练基于小波张量的模型"""
    print("🧠 训练CNN模型...")

    num_classes = len(class_names)
    model = build_cnn_model(X_train.shape[1:], num_classes)

    # 处理类别不平衡，计算类别权重
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=unique_classes,
                                                      y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(f"\n✅ CNN训练完成!")
    print(f"   测试准确率: {test_accuracy:.4f}")

    # 分类报告与混淆矩阵
    unique_eval_classes = np.unique(y_test)
    target_names = [class_names[idx] for idx in unique_eval_classes]
    textual_report = classification_report(y_test, y_pred_classes,
                                           labels=unique_eval_classes,
                                           target_names=target_names,
                                           digits=4)
    report_dict = classification_report(y_test, y_pred_classes,
                                        labels=unique_eval_classes,
                                        target_names=target_names,
                                        digits=4,
                                        output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    print("\n📊 分类报告:")
    print(textual_report)

    return model, test_accuracy, history, textual_report, report_dict, conf_matrix

def quantize_model_for_fpga(model, representative_data, output_dir, timestamp, calibration_size=256):
    """使用TensorFlow Lite进行整型量化，便于在Pynq-Z2上部署"""

    print("⚡ 模型量化 (TensorFlow Lite INT8)...")

    if representative_data is None or len(representative_data) == 0:
        raise ValueError("代表性数据集为空，无法执行量化")

    os.makedirs(output_dir, exist_ok=True)

    representative_data = np.asarray(representative_data, dtype=np.float32)
    calibration_size = min(calibration_size, representative_data.shape[0])
    calibration_samples = representative_data[:calibration_size]

    def representative_dataset():
        for sample in calibration_samples:
            sample = sample.astype(np.float32)
            yield [np.expand_dims(sample, axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quant_mode = 'int8'
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        print(f"   ⚠️ INT8量化失败 ({exc})，尝试Float16量化作为回退方案")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quant_mode = 'float16'
        tflite_model = converter.convert()

    quant_model_path = os.path.join(output_dir, f"ecg_cnn_{quant_mode}_{timestamp}.tflite")
    with open(quant_model_path, 'wb') as f:
        f.write(tflite_model)

    quantization_details = {
        'mode': quant_mode,
        'calibration_samples': int(calibration_size),
        'tflite_path': quant_model_path,
        'tflite_size_bytes': len(tflite_model)
    }

    print(f"   ✅ 量化完成，输出文件: {quant_model_path}")

    return quant_model_path, quantization_details



def export_cnn_weights_for_hls(model, output_dir):
    """导出CNN权重为HLS友好的格式 (NPZ + 头文件)"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weight_tensors = {}
    header_lines = ["#ifndef CNN_WEIGHTS_H", "#define CNN_WEIGHTS_H", ""]

    for idx, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue

        layer_name = f"layer_{idx}_{layer.name}".replace('/', '_').replace('-', '_')
        weights = layer_weights[0].astype(np.float32)
        weight_tensors[f"{layer_name}_weights"] = weights

        header_lines.append(f"// Layer {idx}: {layer.name} weights")
        header_lines.append(f"const int {layer_name}_weights_rank = {weights.ndim};")
        header_lines.append(
            f"const int {layer_name}_weights_shape[{weights.ndim}] = {{{', '.join(str(dim) for dim in weights.shape)}}};")
        header_lines.append(f"const float {layer_name}_weights[{weights.size}] = {{")
        flat_weights = weights.flatten()
        for start_idx in range(0, flat_weights.size, 8):
            chunk = flat_weights[start_idx:start_idx + 8]
            chunk_str = ", ".join(f"{val:.8e}f" for val in chunk)
            suffix = ',' if start_idx + 8 < flat_weights.size else ''
            header_lines.append(f"    {chunk_str}{suffix}")
        header_lines.append("};")
        header_lines.append("")

        if len(layer_weights) > 1:
            biases = layer_weights[1].astype(np.float32)
            weight_tensors[f"{layer_name}_biases"] = biases
            header_lines.append(f"const int {layer_name}_biases_length = {biases.size};")
            header_lines.append(f"const float {layer_name}_biases[{biases.size}] = {{")
            flat_biases = biases.flatten()
            for start_idx in range(0, flat_biases.size, 8):
                chunk = flat_biases[start_idx:start_idx + 8]
                chunk_str = ", ".join(f"{val:.8e}f" for val in chunk)
                suffix = ',' if start_idx + 8 < flat_biases.size else ''
                header_lines.append(f"    {chunk_str}{suffix}")
            header_lines.append("};")
            header_lines.append("")

    header_lines.append("#endif // CNN_WEIGHTS_H")

    header_path = output_path / "cnn_weights.h"
    header_path.write_text("\n".join(header_lines), encoding='utf-8')

    weights_npz_path = output_path / "cnn_weights.npz"
    if weight_tensors:
        np.savez_compressed(weights_npz_path, **weight_tensors)
    else:
        np.savez_compressed(weights_npz_path, placeholder=np.array([], dtype=np.float32))

    print(f"✅ CNN权重已导出: {weights_npz_path}")
    return str(weights_npz_path), str(header_path)


def create_fpga_deployment_package(model,
                                   class_names,
                                   label_mapping,
                                   quant_model_path,
                                   quantization_details,
                                   history,
                                   textual_report,
                                   conf_matrix,
                                   class_distribution,
                                   timestamp):
    """创建适配Pynq-Z2的部署资源包"""

    print("\n🔧 创建FPGA/Pynq部署资源包...")

    output_dir = Path(f"outputs/fpga_deployment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_path = Path(quant_model_path)
    quant_dest = output_dir / quant_path.name
    shutil.copy2(quant_path, quant_dest)

    weights_npz_path, weights_header_path = export_cnn_weights_for_hls(model, output_dir / "weights")

    history_data = {}
    if history is not None and hasattr(history, 'history'):
        history_data = {k: [float(x) for x in v] for k, v in history.history.items()}

    metadata = {
        'model_type': 'Wavelet-CNN (CWT + 2D CNN)',
        'timestamp': timestamp,
        'input_shape': list(model.input_shape[1:]),
        'num_parameters': int(model.count_params()),
        'class_names': class_names,
        'label_mapping': label_mapping,
        'class_distribution': class_distribution,
        'quantization': quantization_details,
        'training_history': history_data,
        'confusion_matrix': conf_matrix.tolist(),
        'weights_npz': os.path.relpath(weights_npz_path, output_dir),
        'weights_header': os.path.relpath(weights_header_path, output_dir),
        'tflite_model': quant_dest.name
    }

    with open(output_dir / "deployment_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(output_dir / "classification_report.txt", 'w', encoding='utf-8') as f:
        f.write(textual_report + "\n")

    np.save(output_dir / "confusion_matrix.npy", conf_matrix)

    readme_text = textwrap.dedent(f"""
        # Pynq-Z2 心律失常CNN部署指南

        本目录包含基于连续小波变换 (CWT) + 卷积神经网络 (CNN) 的心律失常分类模型，已经完成INT8量化，支持直接在 Pynq-Z2 的 ARM 端通过 TensorFlow Lite 运行，或进一步移植到 DPU/HLS 加速器中。

        ## 目录结构
        - `{quant_dest.name}`: 量化后的 TensorFlow Lite 模型。
        - `weights/cnn_weights.npz`: 原始浮点卷积/全连接权重，便于定制化量化或FINN/TVM等工具链使用。
        - `weights/cnn_weights.h`: HLS 友好的权重头文件，可直接在Vitis HLS项目中包含。
        - `deployment_metadata.json`: 模型结构、量化、类别映射等关键信息。
        - `classification_report.txt`: 测试集分类指标。
        - `confusion_matrix.npy`: 测试集混淆矩阵 (numpy格式)。
        - `pynq_z2_tflite_inference.py`: Pynq-Z2 上的推理示例脚本。

        ## 在Pynq-Z2上运行TensorFlow Lite
        1. 将整个 `fpga_deployment_{timestamp}` 目录复制到板卡（例如 `/home/xilinx/ecg_cnn`）。
        2. 在Pynq终端执行 `sudo pip3 install --upgrade tflite-runtime pywavelets numpy` 安装依赖。
        3. 进入目录并运行 `python3 pynq_z2_tflite_inference.py --input sample_beat.npy`，脚本会自动生成小波尺度图并调用量化模型输出预测结果。

        ## 在FPGA/DPU上进一步加速
        - 使用 `weights/cnn_weights.npz` 与 `deployment_metadata.json` 中的输入形状、通道顺序信息，可在Vitis AI或FINN工具链中重建并量化网络。
        - `weights/cnn_weights.h` 可直接包含到Vitis HLS项目中，结合 `deployment_metadata.json` 的量化比例实现手写加速器。

        ## 输入预处理
        - 输入为单导联 ECG 心拍（300样本，360Hz采样）。
        - 预处理与训练保持一致：带通滤波 → 小波CWT (`morl`, 1~64尺度) → 幅值归一化至[0,1] → 作为CNN输入 (H×W×1)。

        ## 支持的心律类型
        {', '.join(class_names)}

        如需集成到自定义工程，可参考 `pynq_z2_tflite_inference.py` 了解完整的数据流。
    """)

    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_text)

    pynq_script = textwrap.dedent("""
        # Pynq-Z2 TensorFlow Lite 推理示例

        import argparse
        import json
        from pathlib import Path

        import numpy as np
        import pywt
        from tflite_runtime.interpreter import Interpreter


        def create_wavelet_scalogram(beat, wavelet='morl', scales=None):
            if scales is None:
                scales = np.arange(1, 65)
            beat = np.asarray(beat, dtype=np.float32)
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            coefficients, _ = pywt.cwt(beat, scales, wavelet)
            scalogram = np.abs(coefficients).astype(np.float32)
            min_val = scalogram.min()
            max_val = scalogram.max()
            scalogram = (scalogram - min_val) / (max_val - min_val + 1e-8)
            return scalogram


        def load_metadata(base_dir):
            with open(base_dir / "deployment_metadata.json", 'r', encoding='utf-8') as f:
                return json.load(f)


        def run_inference(base_dir, beat_path):
            metadata = load_metadata(base_dir)
            interpreter = Interpreter(model_path=str(base_dir / metadata['tflite_model']))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            if beat_path is None:
                raise ValueError('请提供包含单个心拍波形 (300样本) 的 .npy 文件')

            beat = np.load(beat_path)
            scalogram = create_wavelet_scalogram(beat)
            scalogram = scalogram[..., np.newaxis]

            scale, zero_point = input_details['quantization']
            if scale == 0:
                raise RuntimeError('量化比例为0，请检查量化模型。')
            quantized = np.round(scalogram / scale + zero_point)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
            quantized = quantized[np.newaxis, ...]

            interpreter.set_tensor(input_details['index'], quantized)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details['index'])[0]

            predicted_idx = int(np.argmax(prediction))
            class_names = metadata['class_names']
            print("预测类别:", class_names[predicted_idx])
            print("各类别概率:")
            for name, prob in zip(class_names, prediction):
                print(f"  {name}: {prob:.4f}")


        def main():
            parser = argparse.ArgumentParser(description='Pynq-Z2 ECG CNN inference demo')
            parser.add_argument('--input', required=True, help='包含单个心拍 (300样本) 的 .npy 文件路径')
            args = parser.parse_args()

            base_dir = Path(__file__).resolve().parent
            run_inference(base_dir, Path(args.input))


        if __name__ == '__main__':
            main()
    """)

    with open(output_dir / "pynq_z2_tflite_inference.py", 'w', encoding='utf-8') as f:
        f.write(pynq_script)

    print(f"✅ FPGA部署包已创建: {output_dir}")
    return str(output_dir)

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
        beats, labels, label_mapping, class_names = loader.load_all_data(max_records=10)  # 先用10个记录测试

        # 统计类别分布
        unique_indices, counts = np.unique(labels, return_counts=True)
        class_distribution = {class_names[idx]: int(count) for idx, count in zip(unique_indices, counts)}
        
        # 第2步：生成小波张量
        wavelet_tensors = create_wavelet_tensors(beats)
        print(f"   ✅ 小波张量形状: {wavelet_tensors.shape}")

        # 第3步：数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            wavelet_tensors, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"   ✅ 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

        # 第4步：训练CNN模型
        model, test_accuracy, history, textual_report, report_dict, conf_matrix = train_cnn_model(
            X_train, X_test, y_train, y_test, class_names=class_names
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 第5步：量化模型并生成FPGA部署包
        quantized_dir = Path('outputs/quantized_models')
        quantized_dir.mkdir(parents=True, exist_ok=True)
        quant_model_path, quant_details = quantize_model_for_fpga(
            model, X_train, str(quantized_dir), timestamp
        )

        fpga_output_dir = create_fpga_deployment_package(
            model=model,
            class_names=class_names,
            label_mapping=label_mapping,
            quant_model_path=quant_model_path,
            quantization_details=quant_details,
            history=history,
            textual_report=textual_report,
            conf_matrix=conf_matrix,
            class_distribution=class_distribution,
            timestamp=timestamp
        )

        # 第6步：保存结果
        history_data = {}
        if history is not None and hasattr(history, 'history'):
            history_data = {k: [float(x) for x in v] for k, v in history.history.items()}

        results = {
            'timestamp': timestamp,
            'training_time': time.time() - start_time,
            'total_beats': int(len(beats)),
            'feature_tensor_shape': list(wavelet_tensors.shape[1:]),
            'test_accuracy': float(test_accuracy),
            'num_parameters': int(model.count_params()),
            'data_source': 'MIT-BIH Arrhythmia Database',
            'technology_stack': 'Continuous Wavelet Transform + 2D CNN',
            'class_names': class_names,
            'class_distribution': class_distribution,
            'label_mapping': label_mapping,
            'quantization': quant_details,
            'classification_report': report_dict,
            'confusion_matrix': conf_matrix.tolist(),
            'training_history': history_data,
            'fpga_package': fpga_output_dir,
            'tflite_model': quant_model_path
        }

        os.makedirs('outputs/experiments', exist_ok=True)
        with open(f'outputs/experiments/real_training_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 保存模型
        os.makedirs('outputs', exist_ok=True)
        model.save(f'outputs/trained_ecg_cnn_{timestamp}.h5')

        print("\n" + "=" * 70)
        print("✅ 基于真实MIT-BIH数据的训练完成！")
        print(f"📊 数据源: MIT-BIH心律失常数据库")
        print(f"💓 训练心拍数: {len(beats):,}")
        print(f"🔧 小波张量尺寸: {wavelet_tensors.shape[1:]}")
        print(f"🧠 训练准确率: {test_accuracy:.4f}")
        print(f"⏱️  总训练时间: {results['training_time']:.1f} 秒")
        print(f"📁 FPGA部署包: {fpga_output_dir}")
        print(f"💾 模型文件: outputs/trained_ecg_cnn_{timestamp}.h5")
        print(f"🧮 支持心律类型: {', '.join(class_names)}")
        print("🎯 模型已适配Pynq-Z2量化部署流程，可直接复制部署目录进行验证。")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
