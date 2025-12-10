# CS461 Final Project - 讽刺检测（Sarcasm Detection）

## 项目概述

本项目旨在构建一个机器学习模型来检测短文本中的讽刺意味。这是CS461课程的期末项目，要求在不使用Transformer架构的前提下，使用传统机器学习或经典神经网络方法完成二分类任务。

## 项目结构

```
CS461_FinalProject/
├── notebooks/                     # Jupyter笔记本（数据探索、实验）
├── src/                           # 源代码
├── models/                        # 保存的模型
├── results/                       # 实验结果
├── report/                        # 报告
├── predict_sarcasm.py             # 推理脚本（提交用）
├── requirements.txt               # Python依赖
└── README.md                      # 本文件
```

详细文件结构请查看 `project_structure.md`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据探索

打开并运行 `notebooks/01_data_exploration.ipynb` 来了解数据集。

### 3. 训练模型

```bash
python src/train.py
```

### 4. 进行预测

```bash
python predict_sarcasm.py --input test.csv --output predictions.csv
```

## 项目要求

### ✅ 允许使用的模型
- 传统机器学习（逻辑回归、SVM、随机森林等）
- 经典神经网络（MLP、CNN、RNN、LSTM、BiLSTM）
- 预训练词嵌入（Word2Vec、GloVe）
- 集成方法

### ❌ 禁止使用的模型
- Transformer架构（BERT、GPT、RoBERTa等）
- 注意力机制模型

## 实施步骤（14-20天）

1. **数据探索** (1-2天) - 统计分析、可视化
2. **数据预处理** (1天) - 文本清洗、规范化
3. **特征工程** (2-3天) - TF-IDF、Word2Vec/GloVe、手工特征
4. **模型训练** (3-4天) - 基线→进阶→集成
5. **模型优化** (2-3天) - 超参数调优、交叉验证
6. **评估与分析** (1-2天) - 性能评估、错误分析
7. **报告撰写** (2-3天) - 完成9个章节

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)

## 提交要求

最终ZIP文件应包含：
1. `report.pdf` - 详细报告（9个章节）
2. `predict_sarcasm.py` - 推理脚本
3. `models/` - 模型权重文件
4. `requirements.txt` - 依赖列表

⚠️ **重要**: 使用相对路径，确保跨平台兼容性！

---

**截止日期**: 2025年12月17日 23:59 ET
