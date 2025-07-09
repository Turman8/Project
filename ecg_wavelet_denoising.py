import numpy as np
import pywt

def wavelet_denoise(signal, wavelet='db4', level=4, threshold_type='soft'):
    """
    使用小波变换对ECG信号进行降噪处理
    
    参数:
        signal (np.array): 原始ECG信号
        wavelet (str): 小波基名称，默认为'db4'
        level (int): 小波分解层数,默认为4
        threshold_type (str): 阈值类型，'soft'或'hard'
    
    返回:
        denoised_signal (np.array): 降噪后的ECG信号
    """
    # 1. 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 2. 噪声估计与阈值计算
    # 使用最高频细节系数估计噪声标准差
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    # 通用阈值公式
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # 3. 阈值处理
    thresholded_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:  # 保留近似系数（低频成分）
            thresholded_coeffs.append(c)
        else:  # 处理细节系数（高频成分）
            if threshold_type == 'soft':
                thresholded_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
            else:
                thresholded_coeffs.append(pywt.threshold(c, threshold, mode='hard'))
    
    # 4. 小波重构
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    
    # 确保输出长度与输入一致
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'constant')
    
    return denoised_signal

if __name__ == "__main__":
    # 测试用例：生成模拟ECG信号
    fs = 360  # 采样率360Hz
    t = np.linspace(0, 10, 10*fs)  # 10秒信号
    clean_ecg = 0.5 * np.sin(2 * np.pi * 1 * t)  # 低频基线
    clean_ecg += np.sin(2 * np.pi * 10 * t)  # QRS波群
    clean_ecg += 0.3 * np.sin(2 * np.pi * 5 * t)  # T波
    
    # 添加噪声
    noise = 0.2 * np.random.randn(len(t))  # 高斯噪声
    noisy_ecg = clean_ecg + noise
    
    # 应用小波降噪
    denoised_ecg = wavelet_denoise(noisy_ecg)
    
    # 计算信噪比改进
    def calculate_snr(signal, reference):
        noise = signal - reference
        return 10 * np.log10(np.sum(reference**2) / np.sum(noise**2))
    
    input_snr = calculate_snr(noisy_ecg, clean_ecg)
    output_snr = calculate_snr(denoised_ecg, clean_ecg)
    
    print(f"输入信号SNR: {input_snr:.2f} dB")
    print(f"输出信号SNR: {output_snr:.2f} dB")
    print(f"SNR改进: {output_snr - input_snr:.2f} dB")
