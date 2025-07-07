import numpy as np
from signal_reader import load_mitdb_record
from ecg_filter import apply_ecg_filters

def segment_heartbeats(signals, r_peaks, beat_types, window=(-100, 199)):
    """
    分割心拍信号
    :param signals: 滤波后信号
    :param r_peaks: R峰位置数组
    :param beat_types: 心拍标签数组
    :param window: 截取窗口（R峰前后点数）
    :return: 心拍列表, 标签列表
    """
    beats = []
    labels = []
    valid_labels = ['N','L','R','V','A','F','E','J']  # 有效心拍类型
    
    start_offset, end_offset = window
    
    for i, peak in enumerate(r_peaks):
        start_idx = peak + start_offset
        end_idx = peak + end_offset + 1
        
        # 检查边界
        if start_idx >= 0 and end_idx < len(signals):
            beat = signals[start_idx:end_idx]
            label = beat_types[i]
            
            if label in valid_labels:
                beats.append(beat)
                labels.append(label)
    
    return beats, labels

def process_ecg_record(record_path):
    """
    完整处理ECG记录
    :param record_path: MIT-BIH记录路径
    :return: 滤波后信号, 心拍列表, 标签列表
    """
    # 1. 读取原始信号
    signals, fs, r_peaks, beat_types = load_mitdb_record(record_path)
    
    # 2. 应用滤波
    filtered_signals = apply_ecg_filters(signals, fs)
    
    # 3. 分割心拍
    beats, labels = segment_heartbeats(filtered_signals, r_peaks, beat_types)
    
    return filtered_signals, beats, labels

if __name__ == "__main__":
    # 测试处理MIT-BIH记录100
    record_path = 'data/100'
    
    # 处理记录
    filtered_signals, beats, labels = process_ecg_record(record_path)
    
    print(f"处理完成! 采样率: 360Hz")
    print(f"总心拍数: {len(beats)}")
    print(f"首5个标签: {labels[:5]}")
    print(f"首段心拍形状: {np.array(beats[0]).shape}")
