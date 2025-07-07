import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def remove_baseline_wander(signals, fs):
    """
    消除ECG信号的基线漂移
    :param signals: 原始心电信号
    :param fs: 采样率 (Hz)
    :return: 滤波后信号
    """
    # 设计0.5Hz高通滤波器 (Butterworth, 3阶)
    nyq = 0.5 * fs  # 奈奎斯特频率
    cutoff = 0.5    # 截止频率0.5Hz
    normal_cutoff = cutoff / nyq
    b, a = butter(3, normal_cutoff, btype='high', analog=False)
    
    # 双向滤波避免相位失真
    return filtfilt(b, a, signals)

def remove_powerline_noise(signals, fs, notch_freq=50, Q=30):
    """
    消除工频干扰
    :param signals: 输入信号
    :param fs: 采样率
    :param notch_freq: 陷波频率 (默认50Hz)
    :param Q: 品质因数 (控制带宽)
    :return: 滤波后信号
    """
    # 设计陷波滤波器
    wo = notch_freq / (fs/2)
    b, a = iirnotch(wo, Q)
    
    # 应用滤波器
    return filtfilt(b, a, signals)

def apply_ecg_filters(signals, fs):
    """
    应用完整的滤波流程
    :param signals: 原始ECG信号
    :param fs: 采样率
    :return: 滤波后信号
    """
    # 1. 消除基线漂移
    baseline_removed = remove_baseline_wander(signals, fs)
    
    # 2. 消除工频干扰
    powerline_removed = remove_powerline_noise(baseline_removed, fs)
    
    return powerline_removed

if __name__ == "__main__":
    # 测试代码
    import numpy as np
    # 生成测试信号 (1Hz正弦波 + 50Hz干扰 + 基线漂移)
    fs = 360
    t = np.arange(0, 1, 1/fs)
    signal = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*50*t) + 0.2*t
    
    # 应用滤波
    filtered = apply_ecg_filters(signal, fs)
    
    print("原始信号:", signal[:5])
    print("滤波后信号:", filtered[:5])
