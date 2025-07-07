import wfdb

def load_mitdb_record(record_path, channels=[0]):
    """
    读取MIT-BIH记录
    """
    record = wfdb.rdrecord(record_path, channels=channels)
    signals = record.p_signal[:, 0]
    fs = record.fs
    annotation = wfdb.rdann(record_path, 'atr')
    r_peaks = annotation.sample
    beat_types = annotation.symbol
    return signals, fs, r_peaks, beat_types

if __name__ == "__main__":
    signals, fs, r_peaks, beat_types = load_mitdb_record('data/100')
    print(f"采样率: {fs}Hz | 信号长度: {len(signals)}")
    print(f"首5个R峰位置: {r_peaks[:5]} | 标签: {beat_types[:5]}")