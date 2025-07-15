#!/usr/bin/env python3
# module_test.py - 全模块测试脚本

import sys
import traceback

def test_module(module_name, description):
    """测试单个模块的导入和基本功能"""
    print(f"\n{'='*50}")
    print(f"测试模块: {module_name}")
    print(f"描述: {description}")
    print(f"{'='*50}")
    
    try:
        # 尝试导入模块
        module = __import__(module_name)
        print(f"✅ {module_name} 导入成功")
        
        # 检查主要函数是否存在
        if hasattr(module, '__all__'):
            print(f"📋 模块导出函数: {module.__all__}")
        
        # 获取所有公共函数
        functions = [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
        if functions:
            print(f"📋 可用函数: {functions}")
        
        return True
        
    except ImportError as e:
        print(f"❌ {module_name} 导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} 测试异常: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始ECG项目全模块测试")
    print(f"Python版本: {sys.version}")
    
    # 定义所有模块
    modules = [
        ("signal_read", "MIT-BIH数据读取模块"),
        ("ecg_filter", "ECG信号滤波模块"),
        ("ecg_wavelet_denoising", "小波降噪模块"),
        ("ecg_segmenter", "心拍分割模块"),
        ("ecg_normalize", "信号归一化模块"),
        ("ecg_feature_extraction", "时域特征提取模块"),
        ("extract_wavelet_features", "小波特征提取模块"),
        ("ecg_feature_extractor", "特征提取集成模块"),
        ("ecg_cnn_model", "CNN模型训练模块"),
        ("ecg_preprocess", "预处理集成模块"),
        ("ecg_visualizer", "可视化模块"),
        ("ecg_main", "主流程模块"),
    ]
    
    # 测试结果统计
    success_count = 0
    total_count = len(modules)
    
    # 逐个测试模块
    for module_name, description in modules:
        if test_module(module_name, description):
            success_count += 1
    
    # 输出测试总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print(f"{'='*60}")
    print(f"总模块数: {total_count}")
    print(f"成功模块: {success_count}")
    print(f"失败模块: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 所有模块测试通过！")
    else:
        print("⚠️ 部分模块存在问题，请检查上述错误信息")
    
    # 测试核心依赖
    print(f"\n{'='*60}")
    print("🔍 核心依赖检查")
    print(f"{'='*60}")
    
    dependencies = [
        "numpy", "scipy", "tensorflow", "sklearn", 
        "matplotlib", "wfdb", "pywt", "pandas"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 可用")
        except ImportError:
            print(f"❌ {dep} 缺失")

if __name__ == "__main__":
    main()
