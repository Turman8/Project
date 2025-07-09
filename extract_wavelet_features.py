import numpy as np
import pywt

def extract_wavelet_wavelet_feature(beats, wavelet='db4', level=4):
    n_beats = beats.shape[0]
    n_wavelet_feature = level * 3
    wavelet_feature = np.zeros((n_beats, n_wavelet_feature))
    wavelet_featurename = []
    for lv in range(level):
        wavelet_featurename.append(f'detail_{lv+1}_entropy')
        wavelet_featurename.append(f'detail_{lv+1}_energy_ratio')
        wavelet_featurename.append(f'detail_{lv+1}_std')
    for i, beat in enumerate(beats):
        coeffs = pywt.wavedec(beat, wavelet, level=level)
        cA = coeffs[0]
        for j in range(1, level + 1):
            cD = coeffs[j]
            entropy = np.sum(-cD**2 * np.log(np.clip(cD**2, 1e-10, None)))
            total_energy = np.sum(cA**2) + np.sum(cD**2)
            energy_ratio = np.sum(cD**2) / (total_energy + 1e-10)
            std_dev = np.std(cD)
            idx = (j-1)*3
            wavelet_feature[i, idx] = entropy
            wavelet_feature[i, idx+1] = energy_ratio
            wavelet_feature[i, idx+2] = std_dev
    return wavelet_feature, wavelet_featurename