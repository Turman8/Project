import numpy as np
from signal_read import load_mitdb_record
from ecg_filter import apply_ecg_filters
from ecg_wavelet_denoising import wavelet_denoise
from ecg_segmenter import segment_heartbeats
from ecg_normalize import normalize_beats

def preprocess_ecg_signal(record_path):
    """
    完成ECG信号的读取、滤波、小波去噪、分割、归一化
    :param record_path: MIT-BIH记录路径
    :return: 归一化心拍、心拍标签、R峰索引、beats对应的R峰索引、采样率
    """
    # 1. 读取原始信号及R峰、标签
    signals, fs, r_peaks, beat_types = load_mitdb_record(record_path)
    # 2. 滤波去除基线漂移和工频干扰
    filtered_signals = apply_ecg_filters(signals, fs)
    # 3. 小波降噪
    denoised_signals = wavelet_denoise(filtered_signals, wavelet='db4', level=4, threshold_type='soft')
    # 4. 基于降噪信号分割心拍
    beats, labels, beat_indices = segment_heartbeats(denoised_signals, r_peaks, beat_types)
    # 5. 归一化心拍
    normalized_beats = normalize_beats(beats)
    return normalized_beats, labels, r_peaks, beat_indices, fs