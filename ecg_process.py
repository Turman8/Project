import numpy as np
from signal_read import load_mitdb_record
from ecg_filter import apply_ecg_filters
from ecg_segmenter import segment_heartbeats

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
