import numpy as np

def normalize_beat(beat):
    """
    归一化单个心拍信号到[-1, 1]范围
    
    参数:
        beat (np.array): 单个心拍信号
        
    返回:
        np.array: 归一化后的心拍信号
    """
    min_val = np.min(beat)
    max_val = np.max(beat)
    
    # 防止除零错误
    if max_val - min_val < 1e-6:
        return np.zeros_like(beat)
    
    # 归一化到[0,1]再转换到[-1,1]
    normalized = (beat - min_val) / (max_val - min_val)
    return 2 * normalized - 1

def normalize_beats(beats):
    """
    批量归一化心拍信号
    
    参数:
        beats (list of np.array): 心拍信号列表
        
    返回:
        list of np.array: 归一化后的心拍信号列表
    """
    return [normalize_beat(beat) for beat in beats]

# 测试用例
if __name__ == "__main__":
    # 模拟心拍数据
    beat1 = np.array([0.1, 0.5, 1.0, 0.8, 0.3])
    beat2 = np.array([-0.2, 0.0, 0.5, 0.2, -0.1])
    beat3 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # 常数信号测试
    
    beats = [beat1, beat2, beat3]
    
    # 归一化处理
    normalized_beats = normalize_beats(beats)
    
    # 打印结果
    print("原始心拍信号:")
    for i, beat in enumerate(beats):
        print(f"心拍 {i+1}: {beat}")
    
    print("\n归一化后心拍信号:")
    for i, norm_beat in enumerate(normalized_beats):
        print(f"心拍 {i+1}: {norm_beat}")
    
    # 验证归一化范围
    for i, norm_beat in enumerate(normalized_beats):
        min_val = np.min(norm_beat)
        max_val = np.max(norm_beat)
        print(f"心拍 {i+1} 范围: [{min_val:.2f}, {max_val:.2f}]")
