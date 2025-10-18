"""
ECG心电图分析系统 - 基于真实MIT-BIH数据训练
使用真实数据：小波特征 + 时域特征 + MLP 分类 + FPGA 部署
"""

import argparse
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pywt
from scipy import signal
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    SpatialDropout2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


CARDIAC_WAVELET_BANDS = [
    {
        'name': 'atrial_low',
        'range_hz': (0.5, 5.0),
        'description': '低频段(0.5-5Hz)，覆盖P波/T波等缓慢成分',
    },
    {
        'name': 'qrs_mid',
        'range_hz': (5.0, 15.0),
        'description': '中频段(5-15Hz)，聚焦于QRS复合波主频',
    },
    {
        'name': 'high_freq',
        'range_hz': (15.0, 40.0),
        'description': '高频段(15-40Hz)，捕获室性波群/快速病理特征',
    },
]

def _serialize_wavelet_bands(bands):
    return [
        {
            'name': band['name'],
            'range_hz': [float(band['range_hz'][0]), float(band['range_hz'][1])],
            'description': band.get('description', ''),
        }
        for band in bands
    ]


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


def _build_frequency_masks(scales, wavelet, fs, bands):
    """根据频段定义生成尺度掩码"""

    freqs = pywt.scale2frequency(wavelet, scales)
    freqs = freqs * fs

    masks = []
    for band in bands:
        low, high = band['range_hz']
        mask = (freqs >= low) & (freqs < high)

        # 若严格范围内无尺度，放宽 10% 以确保至少包含一个尺度
        if not np.any(mask):
            tolerance = max(0.5, 0.1 * (high - low))
            mask = (freqs >= max(0.0, low - tolerance)) & (freqs < high + tolerance)

        masks.append(mask.astype(np.float32))

    return np.stack(masks, axis=0)


def create_wavelet_tensors(
    beats,
    wavelet='morl',
    scales=None,
    output_format='2d',
    fs=360,
    channel_strategy='cardiac_band',
    cache_dir='outputs/cache',
    dtype=np.float16,
):
    """根据心拍生成小波张量并可选落盘缓存以降低内存峰值

    Args:
        beats (np.ndarray): 心拍集合，形状为 [N, T]
        wavelet (str): 小波基类型
        scales (list or np.ndarray, optional): 小波尺度
        output_format (str): "2d" 返回 [N, H, W, C] 张量, "sequence" 返回 [N, T, C]
        fs (int): 采样频率
        channel_strategy (str): 生成通道的策略，默认为根据P/QRS/T频段生成多通道
        cache_dir (str or Path): 若提供则使用内存映射文件缓存结果，避免一次性占满内存
        dtype (np.dtype): 存储小波张量的精度，默认为 float16 以压缩磁盘/内存占用

    Returns:
        tuple[np.memmap, dict]: 生成的小波张量内存映射及相关元信息
    """

    if scales is None:
        scales = np.arange(1, 65)

    if channel_strategy not in {'cardiac_band', 'single'}:
        raise ValueError("channel_strategy must be 'cardiac_band' or 'single'")

    beats = np.asarray(beats, dtype=np.float32)
    if beats.ndim != 2:
        raise ValueError("beats 必须是形状为 [N, T] 的二维数组")

    num_samples, signal_length = beats.shape
    if num_samples == 0:
        raise ValueError("传入的心拍数量为0，无法生成小波张量")

    frequency_masks = None
    if channel_strategy == 'cardiac_band':
        frequency_masks = _build_frequency_masks(scales, wavelet, fs, CARDIAC_WAVELET_BANDS)

    num_channels = len(frequency_masks) if frequency_masks is not None else 1
    dtype = np.dtype(dtype)

    memmap = None
    memmap_path = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_shape = (num_samples, len(scales), signal_length, num_channels)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        memmap_path = cache_dir / f'wavelet_tensors_{timestamp}.dat'
        memmap = np.memmap(memmap_path, mode='w+', dtype=dtype, shape=target_shape)
        tensors = memmap
    else:
        tensors = np.empty(target_shape, dtype=dtype)

    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(num_channels, dtype=np.float64)
    channel_counts = np.zeros(num_channels, dtype=np.int64)

    for idx, beat in enumerate(beats):
        if idx % 1000 == 0:
            print(f"   生成小波张量 {idx + 1}/{len(beats)}")

        beat_norm = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        coefficients, _ = pywt.cwt(beat_norm, scales, wavelet)
        scalogram = np.log1p(np.abs(coefficients).astype(np.float32))

        if channel_strategy == 'cardiac_band' and frequency_masks is not None:
            stacked = np.zeros((len(scales), signal_length, num_channels), dtype=np.float32)
            for channel_idx, mask in enumerate(frequency_masks):
                active_rows = mask > 0
                band_slice = scalogram[active_rows, :]
                if band_slice.size == 0:
                    continue
                channel_sum[channel_idx] += float(np.sum(band_slice))
                channel_sq_sum[channel_idx] += float(np.sum(np.square(band_slice)))
                channel_counts[channel_idx] += band_slice.size
                stacked[:, :, channel_idx] = scalogram * mask[:, np.newaxis]
        else:
            stacked = scalogram[..., np.newaxis]
            channel_sum[0] += float(np.sum(scalogram))
            channel_sq_sum[0] += float(np.sum(np.square(scalogram)))
            channel_counts[0] += scalogram.size

        tensors[idx] = stacked.astype(dtype, copy=False)

    clip_min, clip_max = -5.0, 5.0
    channel_mean = np.divide(
        channel_sum,
        channel_counts,
        out=np.zeros_like(channel_sum),
        where=channel_counts > 0,
    )
    channel_var = np.divide(
        channel_sq_sum,
        channel_counts,
        out=np.zeros_like(channel_sq_sum),
        where=channel_counts > 0,
    ) - np.square(channel_mean)
    channel_var = np.maximum(channel_var, 1e-6)
    channel_std = np.sqrt(channel_var)

    if isinstance(tensors, np.memmap):
        for idx in range(num_samples):
            sample = np.asarray(tensors[idx], dtype=np.float32)
            normalized = (sample - channel_mean.reshape((1, 1, num_channels))) / channel_std.reshape((1, 1, num_channels))
            normalized = np.clip(normalized, clip_min, clip_max)
            tensors[idx] = normalized.astype(dtype, copy=False)

        tensors.flush()
        info = {
            'path': str(memmap_path),
            'shape': list(target_shape),
            'dtype': tensors.dtype.str,
            'timestamp': timestamp,
            'wavelet': wavelet,
            'channel_strategy': channel_strategy,
            'normalization': {
                'type': 'log1p_zscore',
                'clip_range': [clip_min, clip_max],
                'channel_mean': channel_mean.astype(np.float32).tolist(),
                'channel_std': channel_std.astype(np.float32).tolist(),
            },
        }
        readonly_memmap = np.memmap(memmap_path, mode='r', dtype=dtype, shape=target_shape)
        return readonly_memmap, info

    tensors = tensors.astype(np.float32)
    tensors = (tensors - channel_mean.reshape((1, 1, 1, num_channels))) / channel_std.reshape((1, 1, 1, num_channels))
    tensors = np.clip(tensors, clip_min, clip_max)

    info = {
        'shape': list(target_shape),
        'dtype': tensors.dtype.str,
        'wavelet': wavelet,
        'channel_strategy': channel_strategy,
        'normalization': {
            'type': 'log1p_zscore',
            'clip_range': [clip_min, clip_max],
            'channel_mean': channel_mean.astype(np.float32).tolist(),
            'channel_std': channel_std.astype(np.float32).tolist(),
        },
    }

    if output_format == '2d':
        return tensors, info
    if output_format == 'sequence':
        reshaped = tensors.reshape(tensors.shape[0], tensors.shape[2], -1)
        seq_info = info.copy()
        seq_info['shape'] = list(reshaped.shape)
        seq_info['dtype'] = reshaped.dtype.str
        return reshaped.astype(np.float32), seq_info

    raise ValueError("output_format must be '2d' or 'sequence'")


def filter_rare_classes(beats, labels, label_mapping, class_names, min_count=2):
    """移除样本数量不足的类别以保证分层抽样的稳定性"""

    unique_labels, counts = np.unique(labels, return_counts=True)
    if not len(unique_labels):
        raise ValueError("标签数组为空，无法继续训练")

    inverse_mapping = {index: name for name, index in label_mapping.items()}

    rare_indices = [label for label, count in zip(unique_labels, counts) if count < min_count]
    if not rare_indices:
        return beats, labels, label_mapping, class_names, {}

    rare_summary = {inverse_mapping[idx]: int(count)
                    for idx, count in zip(unique_labels, counts)
                    if idx in rare_indices}

    keep_indices = [label for label, count in zip(unique_labels, counts) if count >= min_count]
    if not keep_indices:
        raise ValueError(
            "所有类别的样本数量都不足以进行训练，请检查数据集或放宽最小阈值。"
        )

    keep_mask = np.isin(labels, keep_indices)
    filtered_beats = beats[keep_mask]
    filtered_labels = labels[keep_mask]

    keep_indices_sorted = sorted(keep_indices)
    reindex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices_sorted)}

    remapped_labels = np.array([reindex_map[idx] for idx in filtered_labels], dtype=np.int32)
    new_class_names = [inverse_mapping[idx] for idx in keep_indices_sorted]
    new_label_mapping = {name: reindex_map[idx]
                         for name, idx in label_mapping.items()
                         if idx in reindex_map}

    return filtered_beats, remapped_labels, new_label_mapping, new_class_names, rare_summary


def compute_class_distribution(labels, class_names):
    """统计标签分布，返回类别名到数量的映射"""

    unique_labels, counts = np.unique(labels, return_counts=True)
    distribution = {}
    for label, count in zip(unique_labels, counts):
        if 0 <= label < len(class_names):
            distribution[class_names[label]] = int(count)
        else:
            distribution[str(label)] = int(count)
    return distribution


def stratified_train_val_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """分层拆分训练/验证/测试集"""

    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 范围内")
    if not 0 < val_size < 1:
        raise ValueError("val_size 必须在 (0, 1) 范围内")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    remaining = 1.0 - test_size
    adjusted_val = val_size / remaining

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _augment_cwt_samples(samples, rng, noise_std=0.01, max_shift=4, scale_range=(0.9, 1.1)):
    """在过采样阶段对小波张量进行轻量扰动"""

    augmented = samples.astype(np.float32).copy()

    if max_shift > 0:
        shifts = rng.integers(-max_shift, max_shift + 1, size=samples.shape[0])
        for i, shift in enumerate(shifts):
            if shift == 0:
                continue
            augmented[i] = np.roll(augmented[i], shift=shift, axis=1)

    if scale_range is not None:
        low, high = scale_range
        scales = rng.uniform(low, high, size=(samples.shape[0], 1, 1, samples.shape[-1]))
        augmented = augmented * scales.astype(np.float32)

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=samples.shape).astype(np.float32)
        augmented += noise

    return np.clip(augmented, 0.0, 1.0)


def rebalance_training_data(
    X,
    y,
    min_samples_per_class=32,
    noise_std=0.01,
    random_state=42,
    adaptive_target=True,
):
    """对训练数据进行过采样，避免极端类不平衡"""

    unique_labels, counts = np.unique(y, return_counts=True)
    augmentations = {}
    augmented_sets = [X]
    augmented_labels = [y]

    rng = np.random.default_rng(random_state)

    target_per_class = min_samples_per_class
    if adaptive_target and counts.size:
        percentile_75 = np.percentile(counts, 75)
        target_per_class = max(min_samples_per_class, int(np.ceil(percentile_75)))

    for label, count in zip(unique_labels, counts):
        if count >= target_per_class:
            continue

        label_indices = np.where(y == label)[0]
        if label_indices.size == 0:
            continue

        needed = int(target_per_class - count)
        sampled_indices = rng.choice(label_indices, size=needed, replace=True)
        samples = X[sampled_indices]
        samples = _augment_cwt_samples(samples, rng, noise_std=noise_std)

        augmented_sets.append(samples)
        augmented_labels.append(np.full(needed, label, dtype=y.dtype))
        augmentations[int(label)] = {
            'original': int(count),
            'added': int(needed),
            'final': int(count + needed),
            'target': int(target_per_class),
        }

    if len(augmented_sets) == 1:
        return X, y, augmentations

    X_augmented = np.concatenate(augmented_sets, axis=0)
    y_augmented = np.concatenate(augmented_labels, axis=0)

    permutation = rng.permutation(len(y_augmented))
    return X_augmented[permutation], y_augmented[permutation], augmentations


class WaveletTensorSequence(Sequence):
    """基于内存映射小波张量的批量加载器"""

    def __init__(
        self,
        memmap_info,
        indices,
        labels,
        batch_size=32,
        shuffle=False,
        augment=False,
        noise_std=0.01,
        max_shift=4,
        scale_range=(0.9, 1.1),
        oversample_target=None,
        adaptive_target=True,
        seed=42,
    ):
        if not isinstance(memmap_info, dict):
            raise ValueError('memmap_info 必须为字典类型')

        self.memmap_path = memmap_info.get('path')
        self.memmap_shape = tuple(memmap_info.get('shape', ()))
        self.memmap_dtype = np.dtype(memmap_info.get('dtype', np.float16))

        if not self.memmap_path or not self.memmap_shape:
            raise ValueError('memmap_info 缺少路径或形状信息')

        self._memmap = np.memmap(
            self.memmap_path,
            mode='r',
            dtype=self.memmap_dtype,
            shape=self.memmap_shape,
        )

        self.base_indices = np.asarray(indices, dtype=np.int64)
        self.base_labels = np.asarray(labels, dtype=np.int32)

        if self.base_indices.size != self.base_labels.size:
            raise ValueError('indices 与 labels 长度不一致')

        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.max_shift = int(max_shift)
        self.scale_range = scale_range
        self.adaptive_target = bool(adaptive_target)
        self.rng = np.random.default_rng(seed)
        self.sample_shape = tuple(self.memmap_shape[1:])

        self.indices = self.base_indices.copy()
        self.labels = self.base_labels.copy()
        self.augmentations = {}
        self.min_samples_per_class = int(oversample_target) if oversample_target is not None else None

        if oversample_target is not None and self.base_labels.size:
            self._apply_oversampling(int(oversample_target))

        self.on_epoch_end()

    def _apply_oversampling(self, min_samples_per_class):
        unique_labels, counts = np.unique(self.base_labels, return_counts=True)
        if unique_labels.size == 0:
            return

        target = min_samples_per_class
        if self.adaptive_target and counts.size:
            percentile_75 = np.percentile(counts, 75)
            target = max(min_samples_per_class, int(np.ceil(percentile_75)))

        augmented_indices = [self.base_indices]
        augmented_labels = [self.base_labels]
        augmentations = {}

        for label, count in zip(unique_labels, counts):
            if count >= target:
                continue

            mask = self.base_labels == label
            label_indices = self.base_indices[mask]
            if label_indices.size == 0:
                continue

            needed = int(target - count)
            sampled = self.rng.choice(label_indices, size=needed, replace=True)

            augmented_indices.append(sampled.astype(np.int64))
            augmented_labels.append(np.full(needed, label, dtype=np.int32))
            augmentations[int(label)] = {
                'original': int(count),
                'added': int(needed),
                'final': int(count + needed),
                'target': int(target),
            }

        if len(augmented_indices) > 1:
            self.indices = np.concatenate(augmented_indices)
            self.labels = np.concatenate(augmented_labels)
        else:
            self.indices = self.base_indices.copy()
            self.labels = self.base_labels.copy()

        self.augmentations = augmentations

    def __len__(self):
        if self.batch_size <= 0:
            raise ValueError('batch_size 必须大于0')
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        batch_labels = self.labels[start:end]

        batch = np.asarray(self._memmap[batch_indices], dtype=np.float32)
        if self.augment:
            batch = _augment_cwt_samples(
                batch,
                rng=self.rng,
                noise_std=self.noise_std,
                max_shift=self.max_shift,
                scale_range=self.scale_range,
            )

        return batch, batch_labels

    def on_epoch_end(self):
        if self.shuffle and len(self.indices) > 1:
            permutation = self.rng.permutation(len(self.indices))
            self.indices = self.indices[permutation]
            self.labels = self.labels[permutation]

    def get_base_distribution(self, class_names):
        return compute_class_distribution(self.base_labels, class_names)

    def get_balanced_distribution(self, class_names):
        return compute_class_distribution(self.labels, class_names)

    def get_effective_size(self):
        return int(len(self.indices))

    def get_original_size(self):
        return int(len(self.base_labels))

    def get_augmentations(self):
        return self.augmentations

    def get_labels(self):
        return self.base_labels.copy()

    def get_numpy_subset(self, subset_indices):
        subset_indices = np.asarray(subset_indices, dtype=np.int64)
        return np.asarray(self._memmap[subset_indices], dtype=np.float32)


def sample_wavelet_representatives(memmap_info, indices, sample_size=512, seed=42):
    """从小波张量内存映射中抽取代表性样本，用于量化或调试"""

    if not isinstance(memmap_info, dict):
        raise ValueError('memmap_info 必须为字典类型')

    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0,) + tuple(memmap_info.get('shape', (0,))[1:]), dtype=np.float32)

    rng = np.random.default_rng(seed)
    sample_size = min(int(sample_size), indices.size)
    chosen = rng.choice(indices, size=sample_size, replace=False)

    memmap = np.memmap(
        memmap_info['path'],
        mode='r',
        dtype=np.dtype(memmap_info.get('dtype', np.float16)),
        shape=tuple(memmap_info.get('shape', ())),
    )
    samples = np.asarray(memmap[chosen], dtype=np.float32)

    return samples


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


def build_cnn_model(input_shape, num_classes, learning_rate=3e-4, weight_decay=1e-4):
    """构建正则化增强的CWT-CNN模型，提高稳定性与泛化能力"""

    kernel_regularizer = regularizers.l2(weight_decay)

    def conv_block(x, filters, kernel_size=(3, 3), pool=True, dropout_rate=0.2, dilation_rate=1):
        x = Conv2D(
            filters,
            kernel_size,
            padding='same',
            dilation_rate=dilation_rate,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_regularizer,
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = SpatialDropout2D(dropout_rate)(x)
        if pool:
            x = MaxPooling2D((2, 2))(x)
        return x

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = conv_block(inputs, 48, kernel_size=(5, 5), dropout_rate=0.1)
    x = conv_block(x, 96, kernel_size=(3, 3), dropout_rate=0.15, dilation_rate=2)
    x = conv_block(x, 160, kernel_size=(3, 3), dropout_rate=0.2)
    x = Flatten()(x)
    x = Dense(
        256,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer,
    )(x)
    x = Dropout(0.45)(x)
    x = Dense(
        128,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=kernel_regularizer,
    )(x)
    x = Dropout(0.35)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='wavelet_regularized_cnn')

    optimizer = Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(name='top3_accuracy', k=3),
        ]
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
    textual_report = classification_report(
        y_test,
        y_pred_classes,
        labels=unique_eval_classes,
        target_names=[class_names[i] for i in unique_eval_classes],
        digits=4,
    )
    report_dict = classification_report(
        y_test,
        y_pred_classes,
        labels=unique_eval_classes,
        target_names=[class_names[i] for i in unique_eval_classes],
        digits=4,
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    print("\n📊 分类报告:")
    print(textual_report)

    return model, scaler, test_accuracy, history, textual_report, report_dict, conf_matrix


def train_cnn_model(
    train_sequence,
    val_sequence,
    test_sequence,
    class_names,
    epochs=40,
    class_weight_strategy='balanced',
    augmentation_strategy=None,
):
    """使用CNN训练基于小波张量的模型，并提供更丰富的调试信息"""

    print("🧠 训练CNN模型...")

    train_distribution = train_sequence.get_base_distribution(class_names)
    val_distribution = val_sequence.get_base_distribution(class_names)
    test_distribution = test_sequence.get_base_distribution(class_names)

    print("   📈 训练集类别分布:")
    for name, count in train_distribution.items():
        print(f"      - {name}: {count}")

    print("   📊 验证集类别分布:")
    for name, count in val_distribution.items():
        print(f"      - {name}: {count}")

    print("   📦 测试集类别分布:")
    for name, count in test_distribution.items():
        print(f"      - {name}: {count}")

    augmentations = train_sequence.get_augmentations()
    balanced_distribution = train_sequence.get_balanced_distribution(class_names)

    if augmentations:
        print("   ♻️  对以下类别执行了过采样:")
        for label_idx, stats in augmentations.items():
            class_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)
            print(
                f"      - {class_name}: 原始 {stats['original']} 个 → 增补 {stats['added']} 个 → 最终 {stats['final']} 个"
            )

    if balanced_distribution:
        print("   🔄 过采样后训练集分布:")
        for name, count in balanced_distribution.items():
            print(f"      - {name}: {count}")
        if train_sequence.augment:
            print("   （小波张量在批次中应用随机平移/缩放/噪声增强）")

    num_classes = len(class_names)
    model = build_cnn_model(train_sequence.sample_shape, num_classes)

    base_labels = train_sequence.get_labels()
    unique_classes = np.unique(base_labels)
    unique_classes = np.sort(unique_classes)
    class_weight_dict = {}
    if class_weight_strategy:
        weights = class_weight.compute_class_weight(
            class_weight=class_weight_strategy,
            classes=unique_classes,
            y=base_labels,
        )
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        train_sequence,
        epochs=epochs,
        validation_data=val_sequence,
        class_weight=class_weight_dict if class_weight_dict else None,
        callbacks=callbacks,
        verbose=1,
    )

    val_metrics = model.evaluate(val_sequence, verbose=0, return_dict=True)
    test_metrics = model.evaluate(test_sequence, verbose=0, return_dict=True)

    val_loss = float(val_metrics.get('loss', 0.0))
    val_accuracy = float(val_metrics.get('accuracy', 0.0))
    val_top3 = float(val_metrics.get('top3_accuracy', 0.0))

    test_loss = float(test_metrics.get('loss', 0.0))
    test_accuracy = float(test_metrics.get('accuracy', 0.0))
    test_top3 = float(test_metrics.get('top3_accuracy', 0.0))
    y_pred = model.predict(test_sequence, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test = test_sequence.get_labels()

    print(f"\n✅ CNN训练完成!")
    print(f"   验证准确率: {val_accuracy:.4f} (Top-3={val_top3:.4f}, loss={val_loss:.4f})")
    print(f"   测试准确率: {test_accuracy:.4f} (Top-3={test_top3:.4f}, loss={test_loss:.4f})")

    unique_eval_classes = np.unique(y_test)
    target_names = [class_names[idx] for idx in unique_eval_classes]
    textual_report = classification_report(
        y_test,
        y_pred_classes,
        labels=unique_eval_classes,
        target_names=target_names,
        digits=4,
    )
    report_dict = classification_report(
        y_test,
        y_pred_classes,
        labels=unique_eval_classes,
        target_names=target_names,
        digits=4,
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    print("\n📊 分类报告:")
    print(textual_report)

    evaluation_summary = {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_top3_accuracy': val_top3,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_top3_accuracy': test_top3,
        'train_distribution': train_distribution,
        'validation_distribution': val_distribution,
        'test_distribution': test_distribution,
        'class_weight': class_weight_dict,
        'augmentations': augmentations,
        'effective_train_size': train_sequence.get_effective_size(),
        'original_train_size': train_sequence.get_original_size(),
        'balanced_train_distribution': balanced_distribution,
        'augmentation_strategy': augmentation_strategy or {
            'noise_std': train_sequence.noise_std,
            'max_time_shift': train_sequence.max_shift,
            'scale_range': list(train_sequence.scale_range) if isinstance(train_sequence.scale_range, (list, tuple)) else train_sequence.scale_range,
            'adaptive_target': train_sequence.adaptive_target,
            'target_min_samples': train_sequence.min_samples_per_class,
        },
    }

    return model, evaluation_summary, history, textual_report, report_dict, conf_matrix

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



def export_cnn_weights_for_hls(model, output_dir, fixed_point_total_bits=16, fixed_point_integer_bits=6):
    """导出CNN权重到HLS模板所需的头文件与NPZ权重包"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(model.input_shape) != 4:
        raise ValueError("模型输入必须是四维张量 [B, H, W, C]")

    input_height, input_width, input_channels = [int(dim) for dim in model.input_shape[1:]]

    conv_layers = []
    dense_layers = []
    flatten_size = None

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            weights, biases = layer.get_weights()
            weights = np.transpose(weights.astype(np.float32), (3, 2, 0, 1))  # OC, IC, KH, KW
            conv_layers.append({
                'name': layer.name,
                'weights': weights,
                'biases': biases.astype(np.float32),
                'kernel': (int(weights.shape[2]), int(weights.shape[3])),
                'in_channels': int(weights.shape[1]),
                'out_channels': int(weights.shape[0]),
                'output_shape': [int(dim) for dim in layer.output_shape[1:4]]
            })

        elif isinstance(layer, BatchNormalization):
            if not conv_layers:
                continue
            gamma, beta, moving_mean, moving_variance = layer.get_weights()
            epsilon = getattr(layer, 'epsilon', 1e-3)
            scale = gamma / np.sqrt(moving_variance + epsilon)
            offset = beta - moving_mean * scale
            conv_layers[-1]['bn_scale'] = scale.astype(np.float32)
            conv_layers[-1]['bn_offset'] = offset.astype(np.float32)

        elif isinstance(layer, MaxPooling2D):
            if conv_layers:
                conv_layers[-1]['pool_output_shape'] = [
                    int(dim) if dim is not None else None for dim in layer.output_shape[1:4]
                ]

        elif isinstance(layer, Flatten):
            flatten_size = int(np.prod([dim for dim in layer.output_shape[1:] if dim is not None]))

        elif isinstance(layer, Dense):
            weights, biases = layer.get_weights()
            dense_layers.append({
                'name': layer.name,
                'weights': weights.astype(np.float32).T,  # OUT, IN
                'biases': biases.astype(np.float32),
                'units': int(layer.units),
                'input_dim': int(weights.shape[0]),
                'activation': getattr(layer.activation, '__name__', 'linear')
            })

    if not conv_layers or not dense_layers:
        raise ValueError("模型中未找到卷积或全连接层，无法导出HLS权重")

    if len(conv_layers) != 3:
        raise ValueError(f"当前HLS模板假定3个卷积块，检测到 {len(conv_layers)} 个。请调整模型或扩展模板。")
    if len(dense_layers) < 2:
        raise ValueError(f"当前HLS模板假定至少2个全连接层，检测到 {len(dense_layers)} 个。")

    for conv in conv_layers:
        if 'bn_scale' not in conv:
            conv['bn_scale'] = np.ones((conv['out_channels'],), dtype=np.float32)
            conv['bn_offset'] = np.zeros((conv['out_channels'],), dtype=np.float32)
        if 'pool_output_shape' not in conv:
            conv['pool_output_shape'] = conv['output_shape']

    if flatten_size is None:
        raise ValueError("模型未包含Flatten层，无法确定全连接输入维度")

    def format_array(name, values, values_per_line=8):
        values = np.asarray(values, dtype=np.float32).ravel()
        lines = [f"static const float {name}[{values.size}] = {{"]
        for start in range(0, values.size, values_per_line):
            chunk = values[start:start + values_per_line]
            chunk_str = ", ".join(f"{val:.8e}f" for val in chunk)
            suffix = ',' if start + values_per_line < values.size else ''
            lines.append(f"    {chunk_str}{suffix}")
        lines.append("};")
        lines.append("")
        return lines

    num_classes = int(dense_layers[-1]['units'])

    header_lines = [
        "#ifndef CNN_WEIGHTS_H",
        "#define CNN_WEIGHTS_H",
        "",
        "#include <cstddef>",
        "",
        f"static constexpr int CNN_HLS_TOTAL_BITS = {int(fixed_point_total_bits)};",
        f"static constexpr int CNN_HLS_INTEGER_BITS = {int(fixed_point_integer_bits)};",
        "",
        f"static constexpr int INPUT_HEIGHT = {input_height};",
        f"static constexpr int INPUT_WIDTH = {input_width};",
        f"static constexpr int INPUT_CHANNELS = {input_channels};",
        f"static constexpr int NUM_CLASSES = {num_classes};",
        ""
    ]

    hls_manifest = {
        'input_shape': [input_height, input_width, input_channels],
        'conv_layers': [],
        'dense_layers': [],
        'flatten_size': int(flatten_size),
        'num_classes': num_classes,
        'weight_statistics': {
            'conv_layers': [],
            'dense_layers': []
        }
    }

    npz_tensors = {}

    for idx, conv in enumerate(conv_layers, start=1):
        kernel_h, kernel_w = conv['kernel']
        out_h, out_w, out_c = conv['output_shape'][0], conv['output_shape'][1], conv['output_shape'][2]
        pool_h, pool_w, pool_c = conv['pool_output_shape'][0], conv['pool_output_shape'][1], conv['pool_output_shape'][2]

        header_lines.extend([
            f"// Conv block {idx}: {conv['name']}",
            f"static constexpr int CONV{idx}_KERNEL_H = {kernel_h};",
            f"static constexpr int CONV{idx}_KERNEL_W = {kernel_w};",
            f"static constexpr int CONV{idx}_IN_CHANNELS = {conv['in_channels']};",
            f"static constexpr int CONV{idx}_OUT_CHANNELS = {conv['out_channels']};",
            f"static constexpr int CONV{idx}_OUTPUT_HEIGHT = {out_h};",
            f"static constexpr int CONV{idx}_OUTPUT_WIDTH = {out_w};",
            f"static constexpr int POOL{idx}_OUTPUT_HEIGHT = {pool_h};",
            f"static constexpr int POOL{idx}_OUTPUT_WIDTH = {pool_w};",
            f"static constexpr int POOL{idx}_OUTPUT_CHANNELS = {pool_c};",
            ""
        ])

        weight_key = f"conv{idx}_weights"
        bias_key = f"conv{idx}_biases"
        scale_key = f"conv{idx}_bn_scale"
        offset_key = f"conv{idx}_bn_offset"

        npz_tensors[weight_key] = conv['weights']
        npz_tensors[bias_key] = conv['biases']
        npz_tensors[scale_key] = conv['bn_scale']
        npz_tensors[offset_key] = conv['bn_offset']

        header_lines.extend(format_array(f"CONV{idx}_WEIGHTS", conv['weights']))
        header_lines.extend(format_array(f"CONV{idx}_BIASES", conv['biases']))
        header_lines.extend(format_array(f"CONV{idx}_BN_SCALE", conv['bn_scale']))
        header_lines.extend(format_array(f"CONV{idx}_BN_OFFSET", conv['bn_offset']))

        hls_manifest['conv_layers'].append({
            'name': conv['name'],
            'kernel': [kernel_h, kernel_w],
            'in_channels': conv['in_channels'],
            'out_channels': conv['out_channels'],
            'output_shape': conv['output_shape'],
            'pool_output_shape': conv['pool_output_shape']
        })

        hls_manifest['weight_statistics']['conv_layers'].append({
            'name': conv['name'],
            'weights': {
                'min': float(np.min(conv['weights'])),
                'max': float(np.max(conv['weights'])),
                'mean': float(np.mean(conv['weights'])),
                'std': float(np.std(conv['weights']))
            },
            'biases': {
                'min': float(np.min(conv['biases'])),
                'max': float(np.max(conv['biases'])),
                'mean': float(np.mean(conv['biases'])),
                'std': float(np.std(conv['biases']))
            },
            'batch_norm': {
                'scale_min': float(np.min(conv['bn_scale'])),
                'scale_max': float(np.max(conv['bn_scale'])),
                'offset_min': float(np.min(conv['bn_offset'])),
                'offset_max': float(np.max(conv['bn_offset']))
            }
        })

    header_lines.append(f"static constexpr int FLATTEN_SIZE = {flatten_size};")
    header_lines.append("")

    for idx, dense in enumerate(dense_layers, start=1):
        npz_tensors[f"dense{idx}_weights"] = dense['weights']
        npz_tensors[f"dense{idx}_biases"] = dense['biases']

        header_lines.extend([
            f"// Dense layer {idx}: {dense['name']} ({dense['activation']})",
            f"static constexpr int DENSE{idx}_INPUTS = {dense['input_dim']};",
            f"static constexpr int DENSE{idx}_OUTPUTS = {dense['units']};",
            ""
        ])
        header_lines.extend(format_array(f"DENSE{idx}_WEIGHTS", dense['weights']))
        header_lines.extend(format_array(f"DENSE{idx}_BIASES", dense['biases']))

        hls_manifest['dense_layers'].append({
            'name': dense['name'],
            'inputs': dense['input_dim'],
            'units': dense['units'],
            'activation': dense['activation']
        })

        hls_manifest['weight_statistics']['dense_layers'].append({
            'name': dense['name'],
            'weights': {
                'min': float(np.min(dense['weights'])),
                'max': float(np.max(dense['weights'])),
                'mean': float(np.mean(dense['weights'])),
                'std': float(np.std(dense['weights']))
            },
            'biases': {
                'min': float(np.min(dense['biases'])),
                'max': float(np.max(dense['biases'])),
                'mean': float(np.mean(dense['biases'])),
                'std': float(np.std(dense['biases']))
            }
        })

    header_lines.append("#endif // CNN_WEIGHTS_H")

    header_path = output_path / "cnn_weights.h"
    header_path.write_text("\n".join(header_lines), encoding='utf-8')

    weights_npz_path = output_path / "cnn_weights.npz"
    np.savez_compressed(weights_npz_path, **npz_tensors)

    with open(output_path / "hls_manifest.json", 'w', encoding='utf-8') as f:
        json.dump(hls_manifest, f, indent=2, ensure_ascii=False)

    print(f"✅ CNN权重已导出: {weights_npz_path}")
    return str(weights_npz_path), str(header_path), hls_manifest


def create_fpga_deployment_package(model,
                                   class_names,
                                   label_mapping,
                                   quant_model_path,
                                   quantization_details,
                                   history,
                                   textual_report,
                                   conf_matrix,
                                   class_distribution,
                                   timestamp,
                                   per_class_metrics=None,
                                   excluded_classes=None,
                                   split_distributions=None,
                                   validation_metrics=None,
                                   wavelet_band_info=None,
                                   cwt_settings=None):
    """创建适配Pynq-Z2的部署资源包"""

    print("\n🔧 创建FPGA/Pynq部署资源包...")

    output_dir = Path(f"outputs/fpga_deployment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_path = Path(quant_model_path)
    quant_dest = output_dir / quant_path.name
    shutil.copy2(quant_path, quant_dest)

    weights_npz_path, weights_header_path, hls_manifest = export_cnn_weights_for_hls(model, output_dir / "weights")
    weights_npz_path = Path(weights_npz_path)
    weights_header_path = Path(weights_header_path)
    weights_dir = weights_npz_path.parent

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
        'hls_manifest': os.path.relpath(weights_dir / "hls_manifest.json", output_dir),
        'weight_statistics': hls_manifest.get('weight_statistics', {}),
        'tflite_model': quant_dest.name,
        'per_class_metrics': per_class_metrics or {},
        'excluded_classes': excluded_classes or {}
    }

    if validation_metrics is not None:
        metadata['validation_metrics'] = validation_metrics

    if split_distributions is not None:
        metadata['dataset_split_distribution'] = split_distributions

    if wavelet_band_info is not None:
        metadata['wavelet_bands'] = wavelet_band_info

    if cwt_settings is not None:
        metadata['cwt_settings'] = cwt_settings

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
        - `hls/`: 结合 `cnn_weights.h` 的Vitis HLS推理模板源码，可直接综合生成Pynq-Z2加速核。
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
        - 预处理与训练保持一致：带通滤波 → 小波CWT (`morl`, 1~64尺度) → 幅值归一化至[0,1] → 构造包含低频/中频/高频三个通道的CNN输入 (H×W×3)。

        ## 支持的心律类型
        {', '.join(class_names)}

        如需集成到自定义工程，可参考 `pynq_z2_tflite_inference.py` 了解完整的数据流。
    """)

    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_text)

    hls_template_dir = Path('FPGA/hls_cnn')
    if hls_template_dir.exists():
        target_hls_dir = output_dir / 'hls'
        shutil.copytree(hls_template_dir, target_hls_dir, dirs_exist_ok=True)
        print(f"   ✅ 已复制HLS推理模板: {target_hls_dir}")

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
    return str(output_dir), hls_manifest.get('weight_statistics', {})


def _collect_per_class_metrics(report_dict):
    metrics = {}
    for class_name, values in report_dict.items():
        if class_name in {'accuracy', 'macro avg', 'weighted avg'}:
            continue
        metrics[class_name] = {
            'precision': float(values.get('precision', 0.0)),
            'recall': float(values.get('recall', 0.0)),
            'f1_score': float(values.get('f1-score', 0.0)),
            'support': int(values.get('support', 0)),
        }
    return metrics


def _format_top_classes(metrics_dict, top_k=5):
    if not metrics_dict:
        return ""
    sorted_items = sorted(metrics_dict.items(), key=lambda kv: kv[1]['f1_score'], reverse=True)
    lines = []
    for name, stats in sorted_items[:top_k]:
        lines.append(
            f"      • {name}: F1={stats['f1_score']:.4f}, 精确率={stats['precision']:.4f}, 召回率={stats['recall']:.4f}, 样本数={stats['support']}"
        )
    return "\n".join(lines)


def _format_weight_summary(weight_stats):
    if not weight_stats:
        return ""
    lines = []
    for layer in weight_stats.get('conv_layers', []):
        w = layer['weights']
        b = layer['biases']
        lines.append(
            f"      • {layer['name']} 卷积权重范围[{w['min']:.4f}, {w['max']:.4f}] (μ={w['mean']:.4f}, σ={w['std']:.4f}); 偏置范围[{b['min']:.4f}, {b['max']:.4f}]"
        )
    for layer in weight_stats.get('dense_layers', []):
        w = layer['weights']
        b = layer['biases']
        lines.append(
            f"      • {layer['name']} 全连接权重范围[{w['min']:.4f}, {w['max']:.4f}] (μ={w['mean']:.4f}, σ={w['std']:.4f}); 偏置范围[{b['min']:.4f}, {b['max']:.4f}]"
        )
    return "\n".join(lines)


def _save_scaler(scaler, path):
    np.savez_compressed(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=getattr(scaler, 'var_', np.square(scaler.scale_)),
    )


def main():
    """主程序 - 使用真实数据训练"""

    parser = argparse.ArgumentParser(description='MIT-BIH ECG training pipeline')
    parser.add_argument(
        '--model',
        choices=['cnn', 'mlp', 'both'],
        default='cnn',
        help='选择训练CNN、MLP或同时训练二者',
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=10,
        help='从MIT-BIH数据集中加载的记录数量（默认10）',
    )
    args = parser.parse_args()

    try:
        print("📊 加载MIT-BIH数据库...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data')
        loader = MITBIHDataLoader(data_path)
        beats, labels, label_mapping, class_names = loader.load_all_data(max_records=args.max_records)

        beats, labels, label_mapping, class_names, dropped_classes = filter_rare_classes(
            beats,
            labels,
            label_mapping,
            class_names,
            min_count=2,
        )

        if dropped_classes:
            print("\n⚠️ 以下类别的样本数不足2个，已从本次训练中移除：")
            for name, count in dropped_classes.items():
                print(f"   - {name}: {count} 个样本")

        unique_indices, counts = np.unique(labels, return_counts=True)
        class_distribution = {class_names[idx]: int(count) for idx, count in zip(unique_indices, counts)}

        executed_models = []

        if args.model in {'cnn', 'both'}:
            print("\n=== Wavelet-CNN 训练流程 ===")
            cnn_start = time.time()

            cwt_scales = np.arange(1, 65)
            _wavelet_memmap, wavelet_info = create_wavelet_tensors(
                beats,
                scales=cwt_scales,
                fs=loader.fs,
                wavelet='morl',
                channel_strategy='cardiac_band',
                cache_dir='outputs/cache',
                dtype=np.float16,
            )
            tensor_shape = tuple(wavelet_info['shape'])
            print(f"   ✅ 小波张量形状: {tensor_shape}")
            print(f"   💾 已写入小波缓存文件: {wavelet_info['path']} (dtype={wavelet_info['dtype']})")
            print("   🎯 小波通道覆盖频段:")
            for band in CARDIAC_WAVELET_BANDS:
                print(
                    f"      - {band['name']}: {band['range_hz'][0]:.1f}-{band['range_hz'][1]:.1f} Hz ({band['description']})"
                )

            all_indices = np.arange(labels.shape[0])
            (
                train_indices,
                val_indices,
                test_indices,
                y_train,
                y_val,
                y_test,
            ) = stratified_train_val_test_split(
                all_indices,
                labels,
                test_size=0.2,
                val_size=0.1,
                random_state=42,
            )

            print(
                f"   ✅ 数据集划分: 训练 {train_indices.shape[0]} / 验证 {val_indices.shape[0]} / 测试 {test_indices.shape[0]}"
            )

            augmentation_config = {
                'noise_std': 0.02,
                'max_time_shift': 4,
                'scale_range': (0.9, 1.1),
                'target_min_samples': 32,
                'adaptive_target': True,
            }

            train_sequence = WaveletTensorSequence(
                memmap_info=wavelet_info,
                indices=train_indices,
                labels=y_train,
                batch_size=32,
                shuffle=True,
                augment=True,
                noise_std=augmentation_config['noise_std'],
                max_shift=augmentation_config['max_time_shift'],
                scale_range=augmentation_config['scale_range'],
                oversample_target=augmentation_config['target_min_samples'],
                adaptive_target=augmentation_config['adaptive_target'],
            )
            val_sequence = WaveletTensorSequence(
                memmap_info=wavelet_info,
                indices=val_indices,
                labels=y_val,
                batch_size=32,
                shuffle=False,
                augment=False,
            )
            test_sequence = WaveletTensorSequence(
                memmap_info=wavelet_info,
                indices=test_indices,
                labels=y_test,
                batch_size=32,
                shuffle=False,
                augment=False,
            )

            del _wavelet_memmap

            (
                model,
                evaluation_summary,
                history,
                textual_report,
                report_dict,
                conf_matrix,
            ) = train_cnn_model(
                train_sequence,
                val_sequence,
                test_sequence,
                class_names=class_names,
                epochs=40,
                class_weight_strategy='balanced',
                augmentation_strategy={
                    'noise_std': augmentation_config['noise_std'],
                    'max_time_shift': augmentation_config['max_time_shift'],
                    'scale_range': list(augmentation_config['scale_range']),
                    'target_min_samples': augmentation_config['target_min_samples'],
                    'adaptive_target': augmentation_config['adaptive_target'],
                },
            )

            per_class_metrics = _collect_per_class_metrics(report_dict)

            augmentations_named = {}
            for idx, stats in evaluation_summary.get('augmentations', {}).items():
                key = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
                augmentations_named[key] = stats

            timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cnn"

            quantized_dir = Path('outputs/quantized_models')
            quantized_dir.mkdir(parents=True, exist_ok=True)
            representative_data = sample_wavelet_representatives(
                wavelet_info,
                np.concatenate([train_indices, val_indices]),
                sample_size=512,
            )
            quant_model_path, quant_details = quantize_model_for_fpga(
                model,
                representative_data,
                str(quantized_dir),
                timestamp,
            )

            cwt_settings = {
                'wavelet': 'morl',
                'scales': [int(x) for x in cwt_scales.tolist()],
                'sampling_rate': loader.fs,
                'channel_strategy': 'cardiac_band',
                'bands': _serialize_wavelet_bands(CARDIAC_WAVELET_BANDS),
                'normalization': wavelet_info.get('normalization'),
            }

            fpga_output_dir, weight_statistics = create_fpga_deployment_package(
                model=model,
                class_names=class_names,
                label_mapping=label_mapping,
                quant_model_path=quant_model_path,
                quantization_details=quant_details,
                history=history,
                textual_report=textual_report,
                conf_matrix=conf_matrix,
                class_distribution=class_distribution,
                timestamp=timestamp,
                per_class_metrics=per_class_metrics,
                excluded_classes=dropped_classes,
                split_distributions={
                    'train': evaluation_summary.get('train_distribution', {}),
                    'validation': evaluation_summary.get('validation_distribution', {}),
                    'test': evaluation_summary.get('test_distribution', {}),
                },
                validation_metrics={
                    'val_accuracy': evaluation_summary.get('val_accuracy'),
                    'val_loss': evaluation_summary.get('val_loss'),
                    'val_top3_accuracy': evaluation_summary.get('val_top3_accuracy'),
                    'test_accuracy': evaluation_summary.get('test_accuracy'),
                    'test_top3_accuracy': evaluation_summary.get('test_top3_accuracy'),
                    'test_loss': evaluation_summary.get('test_loss'),
                },
                wavelet_band_info=_serialize_wavelet_bands(CARDIAC_WAVELET_BANDS),
                cwt_settings=cwt_settings,
            )

            history_data = {}
            if history is not None and hasattr(history, 'history'):
                history_data = {k: [float(x) for x in v] for k, v in history.history.items()}

            results = {
                'timestamp': timestamp,
                'training_time': time.time() - cnn_start,
                'total_beats': int(len(beats)),
                'feature_tensor_shape': list(tensor_shape[1:]),
                'test_accuracy': float(evaluation_summary.get('test_accuracy', 0.0)),
                'test_top3_accuracy': float(evaluation_summary.get('test_top3_accuracy', 0.0)),
                'test_loss': float(evaluation_summary.get('test_loss', 0.0)),
                'validation_accuracy': float(evaluation_summary.get('val_accuracy', 0.0)),
                'validation_top3_accuracy': float(evaluation_summary.get('val_top3_accuracy', 0.0)),
                'validation_loss': float(evaluation_summary.get('val_loss', 0.0)),
                'train_distribution': evaluation_summary.get('train_distribution', {}),
                'validation_distribution': evaluation_summary.get('validation_distribution', {}),
                'test_distribution': evaluation_summary.get('test_distribution', {}),
                'balanced_train_distribution': evaluation_summary.get('balanced_train_distribution', {}),
                'augmentations': augmentations_named,
                'class_weight': evaluation_summary.get('class_weight', {}),
                'effective_train_size': evaluation_summary.get('effective_train_size'),
                'original_train_size': evaluation_summary.get('original_train_size'),
                'augmentation_strategy': evaluation_summary.get('augmentation_strategy'),
                'num_parameters': int(model.count_params()),
                'data_source': 'MIT-BIH Arrhythmia Database',
                'technology_stack': 'Continuous Wavelet Transform + 2D CNN',
                'class_names': class_names,
                'class_distribution': class_distribution,
                'label_mapping': label_mapping,
                'wavelet_bands': _serialize_wavelet_bands(CARDIAC_WAVELET_BANDS),
                'cwt_settings': cwt_settings,
                'quantization': quant_details,
                'classification_report': report_dict,
                'confusion_matrix': conf_matrix.tolist(),
                'training_history': history_data,
                'fpga_package': fpga_output_dir,
                'tflite_model': quant_model_path,
                'per_class_metrics': per_class_metrics,
                'weight_statistics': weight_statistics,
                'excluded_classes': dropped_classes,
                'wavelet_cache': {
                    'path': wavelet_info.get('path'),
                    'dtype': wavelet_info.get('dtype'),
                    'shape': tensor_shape,
                    'normalization': wavelet_info.get('normalization'),
                },
            }

            os.makedirs('outputs/experiments', exist_ok=True)
            with open(f'outputs/experiments/cnn_training_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            os.makedirs('outputs', exist_ok=True)
            model_path = f'outputs/trained_ecg_cnn_{timestamp}.h5'
            model.save(model_path)

            print("\n" + "=" * 70)
            print("✅ Wavelet-CNN 训练完成！")
            print(f"📊 数据源: MIT-BIH心律失常数据库")
            print(f"💓 训练心拍数: {len(beats):,}")
            print(f"🔧 小波张量尺寸: {tensor_shape[1:]}")
            print(
                "🧠 验证准确率: "
                f"{evaluation_summary.get('val_accuracy', 0.0):.4f} (Top-3 {evaluation_summary.get('val_top3_accuracy', 0.0):.4f}) / "
                f"测试准确率: {evaluation_summary.get('test_accuracy', 0.0):.4f} (Top-3 {evaluation_summary.get('test_top3_accuracy', 0.0):.4f})"
            )
            print(f"⏱️  训练耗时: {results['training_time']:.1f} 秒")
            print(f"📁 FPGA部署包: {fpga_output_dir}")
            print(f"💾 模型文件: {model_path}")
            print(f"🧮 支持心律类型: {', '.join(class_names)}")
            if per_class_metrics:
                print("📈 各类别F1评分:")
                print(_format_top_classes(per_class_metrics, top_k=min(10, len(per_class_metrics))))
            if weight_statistics:
                print("⚖️ 权重统计:")
                print(_format_weight_summary(weight_statistics))
            print("🎯 模型已适配Pynq-Z2量化部署流程，可直接复制部署目录进行验证。")

            executed_models.append('cnn')

        if args.model in {'mlp', 'both'}:
            print("\n=== Wavelet+Time 特征 + MLP 训练流程 ===")
            mlp_start = time.time()

            feature_vectors = extract_all_features(beats)
            print(f"   ✅ 特征向量形状: {feature_vectors.shape}")

            X_train, X_test, y_train, y_test = train_test_split(
                feature_vectors, labels, test_size=0.2, random_state=42, stratify=labels
            )
            print(f"   ✅ 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

            (
                model,
                scaler,
                test_accuracy,
                history,
                textual_report,
                report_dict,
                conf_matrix,
            ) = train_model(X_train, X_test, y_train, y_test, class_names=class_names)

            per_class_metrics = _collect_per_class_metrics(report_dict)

            timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mlp"

            history_data = {}
            if history is not None and hasattr(history, 'history'):
                history_data = {k: [float(x) for x in v] for k, v in history.history.items()}

            scaler_path = Path(f'outputs/mlp_scaler_{timestamp}.npz')
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            _save_scaler(scaler, scaler_path)

            results = {
                'timestamp': timestamp,
                'training_time': time.time() - mlp_start,
                'total_beats': int(len(beats)),
                'feature_vector_dim': int(feature_vectors.shape[1]),
                'test_accuracy': float(test_accuracy),
                'num_parameters': int(model.count_params()),
                'data_source': 'MIT-BIH Arrhythmia Database',
                'technology_stack': 'Wavelet Statistical Features + Time-domain Features + MLP',
                'class_names': class_names,
                'class_distribution': class_distribution,
                'label_mapping': label_mapping,
                'classification_report': report_dict,
                'confusion_matrix': conf_matrix.tolist(),
                'training_history': history_data,
                'scaler_path': str(scaler_path),
                'per_class_metrics': per_class_metrics,
                'excluded_classes': dropped_classes,
            }

            os.makedirs('outputs/experiments', exist_ok=True)
            with open(f'outputs/experiments/mlp_training_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            model_path = f'outputs/trained_ecg_mlp_{timestamp}.h5'
            model.save(model_path)

            print("\n" + "=" * 70)
            print("✅ Wavelet+Time 特征 + MLP 训练完成！")
            print(f"📊 数据源: MIT-BIH心律失常数据库")
            print(f"💓 训练心拍数: {len(beats):,}")
            print(f"🧾 特征维度: {feature_vectors.shape[1]}")
            print(f"🧠 测试准确率: {test_accuracy:.4f}")
            print(f"⏱️  训练耗时: {results['training_time']:.1f} 秒")
            print(f"💾 模型文件: {model_path}")
            print(f"📊 分类报告已写入: outputs/experiments/mlp_training_{timestamp}.json")
            print(f"🧮 支持心律类型: {', '.join(class_names)}")
            if per_class_metrics:
                print("📈 各类别F1评分:")
                print(_format_top_classes(per_class_metrics, top_k=min(10, len(per_class_metrics))))
            print("🎯 已保留原始MLP特征工程与模型，可用于对比或迁移部署。")

            executed_models.append('mlp')

        if not executed_models:
            raise RuntimeError('未选择任何模型进行训练。')

    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
