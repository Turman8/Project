# ecg_main.py - ECG处理主流程
from ecg_preprocess import preprocess_ecg_signal
from ecg_feature_extractor import extract_combined_features
from ecg_cnn_model import build_and_train_model

def main(record_path):
    """
    心电图处理主流程
    
    参数:
        record_path: MIT-BIH记录路径
    """
    # 1. 预处理：降噪+分割+标准化
    beats, labels, r_peaks, beat_indices, fs = preprocess_ecg_signal(record_path)
    
    # 2. 特征提取
    features, feature_names = extract_combined_features(
        beats, r_peaks, beat_indices, fs
    )
    
    # 3. CNN模型训练
    model, history = build_and_train_model(features, labels)
    
    return model, features, labels

if __name__ == "__main__":
    # 测试记录100
    model, features, labels = main('data/100')
    print("模型训练完成 | 特征维度:", features.shape)
