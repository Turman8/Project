#!/usr/bin/env python3
"""
ECG项目快速开发启动工具
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def load_project_config():
    """加载项目配置"""
    with open('project_config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def check_project_status():
    """检查项目状态"""
    print("🔍 项目状态检查...")
    
    # 运行安全检查
    result = subprocess.run([sys.executable, 'project_safety_check.py'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ 项目状态正常")
        return True
    else:
        print("❌ 项目状态异常")
        print(result.stderr)
        return False

def show_development_options():
    """显示开发选项"""
    config = load_project_config()
    
    print(f"\n🚀 ECG项目开发环境")
    print("=" * 50)
    print(f"📦 项目版本: {config['project']['version']}")
    print(f"🌿 当前分支: ", end="")
    
    # 获取当前分支
    result = subprocess.run(['git', 'branch', '--show-current'], 
                          capture_output=True, text=True)
    print(result.stdout.strip())
    
    print(f"🏛️ 稳定基线: {config['development']['stable_branch']}")
    print(f"🎯 模型准确率: {config['model']['accuracy']}")
    
    print("\n📋 可用操作:")
    print("1. 🧪 运行ECG训练和测试")
    print("2. 🔧 启动FPGA开发环境")
    print("3. 📊 查看数据分析")
    print("4. 🌿 创建新的功能分支")
    print("5. 🛡️ 运行安全检查")
    print("6. 📚 查看项目文档")
    print("0. 退出")

def create_feature_branch():
    """创建新功能分支"""
    print("\n🌿 创建新功能分支")
    feature_name = input("请输入功能名称 (如: improved-visualization): ")
    
    if not feature_name:
        print("❌ 功能名称不能为空")
        return
    
    branch_name = f"feature/{feature_name}"
    
    try:
        # 确保从稳定基线创建
        subprocess.run(['git', 'checkout', 'baseline-stable'], check=True)
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        
        print(f"✅ 成功创建功能分支: {branch_name}")
        print(f"💡 现在可以开始开发新功能: {feature_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 创建分支失败: {e}")

def run_ecg_system():
    """运行ECG系统"""
    print("\n🧪 启动ECG分析系统...")
    
    choice = input("选择运行模式:\n1. 完整训练\n2. 快速测试\n3. 数据分析\n请选择 (1-3): ")
    
    if choice == "1":
        print("🚀 启动完整训练...")
        subprocess.run([sys.executable, 'main.py'])
    elif choice == "2":
        print("⚡ 启动快速测试...")
        subprocess.run([sys.executable, 'main.py', '--quick-test'])
    elif choice == "3":
        print("📊 启动数据分析...")
        subprocess.run([sys.executable, 'main.py', '--analyze-only'])
    else:
        print("❌ 无效选择")

def main():
    """主函数"""
    if not os.path.exists('project_config.json'):
        print("❌ 项目配置文件不存在！请确保在项目根目录运行。")
        return
    
    print("🎯 ECG项目开发快速启动")
    print("=" * 70)
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查项目状态
    if not check_project_status():
        response = input("⚠️ 项目状态异常，是否继续？(y/N): ")
        if response.lower() != 'y':
            return
    
    while True:
        show_development_options()
        choice = input("\n请选择操作 (0-6): ")
        
        if choice == "0":
            print("👋 退出开发环境")
            break
        elif choice == "1":
            run_ecg_system()
        elif choice == "2":
            print("\n🔧 FPGA开发环境")
            print("💡 请使用 Vivado 2024.1.2 打开 FPGA/hls_project")
            print("📁 源码位置: FPGA/hls_source/")
        elif choice == "3":
            print("\n📊 数据分析工具")
            subprocess.run([sys.executable, '-c', 
                          'import main; main.analyze_mitbih_data()'])
        elif choice == "4":
            create_feature_branch()
        elif choice == "5":
            subprocess.run([sys.executable, 'project_safety_check.py'])
        elif choice == "6":
            print("\n📚 项目文档:")
            print("- PROJECT_REPORT.md: 技术报告")
            print("- PROJECT_SAFETY_STATUS.md: 安全状态")  
            print("- DEVELOPMENT_WORKFLOW.md: 开发流程")
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()
