import numpy as np
from scipy import signal, interpolate


def extract_temporal_features(beats, r_peaks_all, beat_indices, fs=360):
    """
    从标准化心拍中提取时域特征
    
    参数:
        beats (np.array): 标准化心拍数组 (n_beats, n_points)
        r_peaks_all (list): 所有R峰位置索引（原始信号中的位置）
        beat_indices (list): 每个心拍对应的R峰在r_peaks_all中的索引
        fs (int): 采样率 (Hz)
        
    返回:
        features (np.array): 特征矩阵 (n_beats, n_features)
        feature_names (list): 特征名称列表
    """
    n_beats = beats.shape[0]
    features = np.zeros((n_beats, 8))  # 8个特征
    feature_names = [
        'RR_interval', 'QRS_duration', 'QT_interval',
        'R_amplitude', 'Q_amplitude', 'S_amplitude',
        'T_amplitude', 'ST_slope'
    ]
    
    # 计算所有RR间期（秒）
    rr_intervals = np.diff(r_peaks_all) / fs
    
    for i, beat in enumerate(beats):
        beat_index = beat_indices[i]
        r_peak = int(len(beat) / 2)  # 心拍中心点为R峰
        
        try:
            # 1. RR间期 (秒)
            if beat_index > 0:
                features[i, 0] = rr_intervals[beat_index - 1]
            elif len(rr_intervals) > 0:
                features[i, 0] = rr_intervals[0]
            else:
                features[i, 0] = 0.8  # 默认值
            
            # 2. 关键点检测
            q_point = find_q_point(beat, r_peak)
            s_point = find_s_point(beat, r_peak)
            t_peak = find_t_peak(beat, r_peak)
            
            # 3. QRS波群持续时间 (ms)
            qrs_duration = (s_point - q_point) * (1000 / fs)
            features[i, 1] = qrs_duration
            
            # 4. QT间期 (ms)
            qt_interval = (t_peak - q_point) * (1000 / fs)
            features[i, 2] = qt_interval
            
            # 5. 幅度特征
            features[i, 3] = beat[r_peak]  # R波幅度
            features[i, 4] = beat[q_point]  # Q波幅度
            features[i, 5] = beat[s_point]  # S波幅度
            features[i, 6] = beat[t_peak]   # T波幅度
            
            # 6. ST段斜率 (mV/s)
            st_start = s_point + int(0.02 * fs)  # J点后20ms
            st_end = st_start + int(0.08 * fs)    # 80ms段
            
            if st_end < len(beat):
                st_segment = beat[st_start:st_end]
                if len(st_segment) > 1:
                    x = np.arange(len(st_segment))
                    slope, _ = np.polyfit(x, st_segment, 1)
                    features[i, 7] = slope * fs * 1000  # 转换为mV/s
                else:
                    features[i, 7] = 0
            else:
                features[i, 7] = 0
                
        except Exception as e:
            print(f"心拍 {i} 特征提取失败: {str(e)}")
            # 设置默认值避免中断处理
            features[i, :] = [0.8, 100, 400, 1.0, -0.2, -0.1, 0.5, 0]
    
    return features, feature_names

def find_q_point(beat, r_peak, search_range=50):
    """定位Q点位置"""
    start_idx = max(0, r_peak - search_range)
    end_idx = r_peak
    
    # 使用导数检测Q点
    derivative = np.gradient(beat[start_idx:end_idx])
    q_candidates = np.where(derivative < -0.05)[0]  # 负斜率阈值
    
    if len(q_candidates) > 0:
        return start_idx + q_candidates[-1]  # 选择最接近R峰的Q点
    else:
        return r_peak - int(0.04 * len(beat))  # 默认位置

def find_s_point(beat, r_peak, search_range=50):
    """定位S点位置"""
    start_idx = r_peak
    end_idx = min(len(beat), r_peak + search_range)
    
    # 使用导数检测S点
    derivative = np.gradient(beat[start_idx:end_idx])
    s_candidates = np.where(derivative > 0.05)[0]  # 正斜率阈值
    
    if len(s_candidates) > 0:
        return start_idx + s_candidates[0]  # 选择最接近R峰的S点
    else:
        return r_peak + int(0.04 * len(beat))  # 默认位置

def find_t_peak(beat, r_peak, search_range=150):
    """定位T波峰值位置"""
    start_idx = r_peak + int(0.1 * len(beat))
    end_idx = min(len(beat), start_idx + search_range)
    
    if end_idx <= start_idx:
        return r_peak + int(0.2 * len(beat))  # 默认位置
    
    t_region = beat[start_idx:end_idx]
    t_peak_rel = np.argmax(t_region)
    return start_idx + t_peak_rel

# 测试用例
if __name__ == "__main__":
    # 生成更真实的模拟心拍数据
    fs = 360
    n_points = 300
    t = np.linspace(0, n_points/fs, n_points)
    
    # 创建心拍模板
    def create_beat_template(t):
        # Q波
        q = -0.3 * np.exp(-((t-0.22)*40)**2)
        # R波
        r = 1.5 * np.exp(-((t-0.25)*50)**2)
        # S波
        s = -0.4 * np.exp(-((t-0.28)*50)**2)
        # T波
        t_wave = 0.6 * np.exp(-((t-0.35)*20)**2)
        return q + r + s + t_wave
    
    # 生成5个心拍
    beats = np.zeros((5, n_points))
    for i in range(5):
        base_template = create_beat_template(t)
        # 添加个体变异
        noise = np.random.normal(0, 0.05, n_points)
        beats[i] = base_template + noise
    
    # 模拟R峰位置和索引
    r_peaks_all = [100, 400, 700, 1000, 1300]  # 原始信号中的R峰位置
    beat_indices = [0, 1, 2, 3, 4]  # 每个心拍对应的索引
    
    # 提取特征
    features, feature_names = extract_temporal_features(beats, r_peaks_all, beat_indices, fs)
    
    # 打印结果
    print("提取的特征名称:", feature_names)
    print("\n5个心拍的特征值:")
    for i in range(5):
        print(f"心拍 {i+1}:")
        for j, name in enumerate(feature_names):
            print(f"  {name}: {features[i, j]:.4f}")
    
    # 可视化一个心拍及其特征点
    import matplotlib.pyplot as plt
    beat_idx = 0
    plt.figure(figsize=(10, 6))
    plt.plot(t, beats[beat_idx], label='ECG信号')
    
    # 标记特征点
    r_peak = int(len(beats[beat_idx]) / 2)
    q_point = find_q_point(beats[beat_idx], r_peak)
    s_point = find_s_point(beats[beat_idx], r_peak)
    t_peak = find_t_peak(beats[beat_idx], r_peak)
    
    plt.plot(t[r_peak], beats[beat_idx][r_peak], 'ro', label='R峰')
    plt.plot(t[q_point], beats[beat_idx][q_point], 'go', label='Q点')
    plt.plot(t[s_point], beats[beat_idx][s_point], 'bo', label='S点')
    plt.plot(t[t_peak], beats[beat_idx][t_peak], 'mo', label='T峰')
    
    plt.title('心拍波形与特征点')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅值 (mV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ecg_beat_with_features.png', dpi=300)
    plt.show()
