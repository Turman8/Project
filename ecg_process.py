import numpy as np
from signal_read import load_mitdb_record
from ecg_filter import apply_ecg_filters
from ecg_segmenter import segment_heartbeats
from ecg_visualizer import visualize_segmentation
from ecg_wavelet_denoising import wavelet_denoise
from ecg_normalize import normalize_beats
from ecg_feature_extraction import extract_temporal_features

def process_ecg_record(record_path):
    """
    完整处理ECG记录
    :param record_path: MIT-BIH记录路径
    :return: 滤波后信号, 心拍列表, 标签列表, 小波降噪后信号, 标准化心拍, 特征, 特征名
    """
    # 1. 读取原始信号
    signals, fs, r_peaks, beat_types = load_mitdb_record(record_path)
    
    # 2. 应用滤波
    filtered_signals = apply_ecg_filters(signals, fs)
    
    # 3. 小波降噪
    denoised_signals = wavelet_denoise(filtered_signals, wavelet='db4', level=4, threshold_type='soft')
    
    # 4. 用降噪后的信号分割心拍
    beats, labels = segment_heartbeats(denoised_signals, r_peaks, beat_types)
    
    # 5. 标准化心拍
    normalized_beats = normalize_beats(beats)
    
    # 6. 提取心拍特征
    # 需要传递r_peaks和beat_indices
    # 计算每个beats对应的R峰在r_peaks中的索引
    beat_indices = []
    valid_labels = ['N', 'L', 'R', 'V', 'A', 'F', 'E', 'J']
    for i, peak in enumerate(r_peaks):
        if beat_types[i] in valid_labels:
            beat_indices.append(i)
    features, feature_names = extract_temporal_features(
        np.array(normalized_beats), r_peaks, beat_indices, fs=fs
    )

    return filtered_signals, beats, labels, denoised_signals, normalized_beats, features, feature_names

if __name__ == "__main__":
    # 测试处理MIT-BIH记录100
    record_path = 'data/100'
    
    # 处理记录
    filtered_signals, beats, labels, denoised_signals, normalized_beats, features, feature_names = process_ecg_record(record_path)
    
    print(f"处理完成! 采样率: 360Hz")
    print(f"总心拍数: {len(beats)}")
    print(f"首5个标签: {labels[:5]}")
    print(f"首段心拍形状: {np.array(beats[0]).shape}")
    print(f"提取了 {features.shape[0]} 个心拍的 {features.shape[1]} 个特征")
    print("特征名称:", feature_names)

    # 重新读取信号用于可视化（保证R峰和标签与beats一致）
    signals, fs, r_peaks, _ = load_mitdb_record(record_path)

    # 可视化小波降噪后的信号
    visualize_segmentation(
        signals=denoised_signals,
        r_peaks=r_peaks,
        beats=beats,
        labels=labels,
        fs=fs
    )
    # 可视化滤波后信号（可选）
    visualize_segmentation(
        signals=filtered_signals,
        r_peaks=r_peaks,
        beats=beats,
        labels=labels,
        fs=fs
    )