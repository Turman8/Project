import numpy as np
from ecg_feature_extraction import extract_temporal_features
from extract_wavelet_features import extract_wavelet_wavelet_feature

def extract_combined_features(beats, r_peaks, beat_indices, fs):
    # 时域特征
    temporal_features, temporal_names = extract_temporal_features(
        np.array(beats), r_peaks, beat_indices, fs
    )
    # 小波特征
    wavelet_features, wavelet_names = extract_wavelet_wavelet_feature(
        np.array(beats)
    )
    # 合并
    combined_features = np.concatenate([temporal_features, wavelet_features], axis=1)
    feature_names = temporal_names + wavelet_names
    return combined_features, feature_names
