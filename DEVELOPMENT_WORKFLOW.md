# 🚀 ECG项目开发工作流指南
**基线版本**: v1.0-baseline  
**创建日期**: 2025年7月18日

## 📋 分支管理策略

### 🏛️ 核心分支
| 分支名 | 用途 | 保护级别 |
|--------|------|----------|
| `baseline-stable` | 🔒 稳定基线，不可直接修改 | **严格保护** |
| `fpga` | 🏭 FPGA相关开发 | 主要开发分支 |
| `main` | 📦 发布分支 | 保护分支 |
| `version_1` | 📚 历史版本 | 归档分支 |

### 🌿 开发分支命名规范
- `feature/功能名` - 新功能开发
- `fix/问题描述` - 错误修复  
- `experiment/实验名` - 实验性功能
- `optimize/优化内容` - 性能优化

## 🔄 推荐开发流程

### 开始新功能开发
```bash
# 1. 确保在最新的稳定版本
git checkout baseline-stable
git pull origin baseline-stable

# 2. 创建功能分支
git checkout -b feature/新功能名

# 3. 开发和提交
git add .
git commit -m "功能描述"

# 4. 完成后合并回主开发分支
git checkout fpga
git merge feature/新功能名
```

### 紧急修复流程
```bash
# 1. 从稳定基线创建修复分支
git checkout baseline-stable
git checkout -b fix/修复描述

# 2. 修复并测试
git add .
git commit -m "修复: 具体问题描述"

# 3. 合并到所有相关分支
git checkout fpga
git merge fix/修复描述
git checkout baseline-stable
git merge fix/修复描述
```

## 🛡️ 安全备份策略

### 自动备份检查点
- **每日检查**: 运行 `python project_safety_check.py`
- **功能完成后**: 创建新的标签版本
- **重大更改前**: 创建备份分支

### 版本标签规范
- `v1.x-baseline` - 稳定基线版本
- `v1.x-feature` - 功能版本  
- `v1.x-release` - 发布版本

## 📊 当前基线版本特性

### ✅ 核心功能 (v1.0-baseline)
- **ECG分析系统**: 691行完整实现
- **MIT-BIH数据支持**: 48个记录完整
- **FPGA部署**: AXI3兼容的定点化实现
- **模型准确率**: 99.08%
- **文档完整**: 技术报告和安全检查

### 🎯 下一步开发建议
1. **功能增强**: `feature/improved-visualization`
2. **性能优化**: `optimize/training-speed`  
3. **新算法**: `experiment/transformer-ecg`
4. **硬件优化**: `feature/fpga-optimization`

## 🚨 重要原则

### 🔒 基线保护
- **baseline-stable** 分支只能通过经过测试的合并请求更新
- 所有基线更改必须通过安全检查
- 保持基线版本始终可运行

### 📝 提交规范
- 使用清晰的提交信息
- 每个提交只包含一个逻辑更改
- 重要更改必须包含测试

### 🧪 测试要求
- 新功能必须验证不破坏现有功能
- FPGA代码更改需要硬件验证
- 数据处理更改需要准确率验证

---
**记住: 稳定的基线是创新的基础！** 🚀
