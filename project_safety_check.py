#!/usr/bin/env python3
"""
项目安全性检查脚本
验证ECG项目的完整性和安全性
"""

import os
import hashlib
import json
from datetime import datetime

def check_file_integrity():
    """检查关键文件的完整性"""
    critical_files = {
        'main.py': 'ECG主程序',
        'FPGA/hls_source/classifier.cpp': 'FPGA分类器源码',
        'FPGA/hls_source/weights.h': '神经网络权重',
        'PROJECT_REPORT.md': '项目报告',
        'export_weights.py': '权重导出工具',
        '.gitignore': 'Git忽略规则'
    }
    
    print("🔍 项目完整性检查")
    print("=" * 50)
    
    status = {}
    
    for file_path, description in critical_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            status[file_path] = {
                'status': '✅ 存在',
                'size': f'{file_size} bytes',
                'hash': file_hash,
                'description': description
            }
            print(f"✅ {file_path:<40} ({file_size:>8} bytes) - {description}")
        else:
            status[file_path] = {
                'status': '❌ 缺失',
                'description': description
            }
            print(f"❌ {file_path:<40} - 文件缺失！")
    
    return status

def check_data_integrity():
    """检查MIT-BIH数据完整性"""
    print("\n📊 MIT-BIH数据完整性检查")
    print("=" * 50)
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print("❌ data目录不存在！")
        return False
    
    # 统计MIT-BIH记录数量
    records = set()
    for file in os.listdir(data_dir):
        if file.endswith('.dat'):
            record_id = file.split('.')[0]
            records.add(record_id)
    
    print(f"📁 发现 {len(records)} 个MIT-BIH记录")
    
    # 检查关键记录是否存在
    key_records = ['100', '101', '102', '103', '104', '105']
    missing_records = []
    
    for record in key_records:
        dat_file = f'{data_dir}/{record}.dat'
        atr_file = f'{data_dir}/{record}.atr'
        hea_file = f'{data_dir}/{record}.hea'
        
        if all(os.path.exists(f) for f in [dat_file, atr_file, hea_file]):
            print(f"✅ 记录 {record}: 完整")
        else:
            print(f"❌ 记录 {record}: 不完整")
            missing_records.append(record)
    
    return len(missing_records) == 0

def check_git_status():
    """检查Git状态"""
    print("\n🔧 Git状态检查")
    print("=" * 50)
    
    try:
        import subprocess
        
        # 检查当前分支
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True)
        current_branch = result.stdout.strip()
        print(f"🌿 当前分支: {current_branch}")
        
        # 检查未提交的更改
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  有未提交的更改:")
            print(result.stdout)
        else:
            print("✅ 工作目录干净")
        
        # 检查提交历史
        result = subprocess.run(['git', 'log', '--oneline', '-5'], 
                              capture_output=True, text=True)
        print("\n📝 最近5次提交:")
        print(result.stdout)
        
        return True
        
    except Exception as e:
        print(f"❌ Git检查失败: {e}")
        return False

def generate_safety_report():
    """生成安全报告"""
    print("\n📋 生成安全报告...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'file_integrity': check_file_integrity(),
        'data_integrity': check_data_integrity(),
        'git_status': check_git_status()
    }
    
    # 保存报告
    report_file = f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 安全报告已保存到: {report_file}")
    
    return report

def main():
    """主函数"""
    print("🛡️  ECG项目安全性检查工具")
    print("=" * 70)
    print(f"⏰ 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 执行各项检查
    file_ok = check_file_integrity()
    data_ok = check_data_integrity()
    git_ok = check_git_status()
    
    # 生成报告
    report = generate_safety_report()
    
    # 总结
    print("\n🎯 安全检查总结")
    print("=" * 50)
    
    all_critical_files_exist = all(
        info.get('status') == '✅ 存在' 
        for info in report['file_integrity'].values()
    )
    
    if all_critical_files_exist and data_ok and git_ok:
        print("🎉 项目状态: 完全安全")
        print("✅ 所有关键文件完整")
        print("✅ MIT-BIH数据完整")  
        print("✅ Git状态正常")
    else:
        print("⚠️  项目状态: 需要注意")
        if not all_critical_files_exist:
            print("❌ 有关键文件缺失")
        if not data_ok:
            print("❌ MIT-BIH数据不完整")
        if not git_ok:
            print("❌ Git状态异常")
    
    # 获取最新报告文件名
    import glob
    report_files = glob.glob("safety_report_*.json")
    latest_report = max(report_files) if report_files else "safety_report.json"
    
    print(f"\n📊 详细报告: {latest_report}")
    print("🔒 项目本地备份建议: 定期复制整个项目文件夹到安全位置")

if __name__ == "__main__":
    main()
