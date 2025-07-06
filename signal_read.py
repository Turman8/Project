import wfdb

# 核心函数：加载单条MIT-BIH记录
def load_mitdb_record(record_path):
    """
    参数:
        record_path: 记录路径 (e.g. 'data/100')
    返回:
        signals: 心电信号值 (numpy数组)
        fs: 采样率 (int)
        r_peaks: R峰位置索引 (numpy数组)
        beat_types: 心拍标签 (list)
    """
    # 读取信号数据 (第一导联)
    record = wfdb.rdrecord(record_path, channels=[0])
    signals = record.p_signal[:, 0]
    fs = record.fs

    # 读取注释文件
    annotation = wfdb.rdann(record_path, 'atr')
    r_peaks = annotation.sample
    beat_types = annotation.symbol

    return signals, fs, r_peaks, beat_types

# 调用示例
if __name__ == "__main__":
    signals, fs, r_peaks, beat_types = load_mitdb_record('data/100')
    print(f"采样率: {fs}Hz | 信号长度: {len(signals)}")
    print(f"首5个R峰位置: {r_peaks[:5]} | 标签: {beat_types[:5]}")