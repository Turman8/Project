import numpy as np

def segment_heartbeats(signals, r_peaks, beat_types, valid_labels=None, window=(-100, 199)):
    """
    分割心拍，返回beats, labels, beat_indices
    
    参数:
        signals (np.array): 滤波后的ECG信号
        r_peaks (list): R峰位置索引数组
        beat_types (list): 对应R峰的心拍类型标签
        valid_labels (list): 需要保留的心拍类型列表
        window (tuple): 以R峰为中心的时间窗口(前,后)
    
    返回:
        beats (np.array): 分割后的心拍信号数组
        labels (np.array): 对应心拍的标签数组
        beat_indices (list): 有效心拍在原始R峰列表中的索引
    """
    # 设置默认有效标签
    if valid_labels is None:
        valid_labels = ['N', 'L', 'R', 'V', 'A', 'F', 'E', 'J']  # MIT-BIH定义的8类有效心拍
    
    beats, labels, beat_indices = [], [], []
    start_offset, end_offset = window
    
    for i, (peak, label) in enumerate(zip(r_peaks, beat_types)):
        # 只保留指定类型的心拍
        if label in valid_labels:
            # 计算心拍起止位置
            start_idx = peak + start_offset
            end_idx = peak + end_offset + 1
            
            # 检查边界有效性
            if start_idx >= 0 and end_idx <= len(signals):
                # 提取心拍信号段
                beat_segment = signals[start_idx:end_idx]
                
                beats.append(beat_segment)
                labels.append(label)
                beat_indices.append(i)  # 记录该beat对应的r_peaks索引
    
    return np.array(beats), np.array(labels), beat_indices

# 测试用例
if __name__ == "__main__":
    # 模拟数据
    sample_signals = np.random.rand(1000)  # 1000点模拟信号
    sample_r_peaks = [100, 300, 500, 700]  # R峰位置
    sample_beat_types = ['N', 'V', 'A', 'N']  # 对应标签
    
    # 调用分割函数
    beats, labels, beat_indices = segment_heartbeats(
        signals=sample_signals,
        r_peaks=sample_r_peaks,
        beat_types=sample_beat_types,
        window=(-100, 199)  # 300点心拍
    )
    
    print(f"分割出有效心拍数量: {len(beats)}")
    print(f"心拍形状: {beats[0].shape if len(beats) > 0 else '无'}")
    print(f"心拍标签: {labels}")
    print(f"心拍索引: {beat_indices}")
