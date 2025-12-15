# CS461 Final Project - 讽刺检测 (Sarcasm Detection)

## 📋 项目要求

训练一个分类模型检测文本中的讽刺。

**限制**：❌ 不能使用 Transformer (BERT, GPT, RoBERTa等)  
**允许**：✅ 传统ML、LSTM、CNN、集成方法

**截止日期**：2025年12月17日

---

## 📁 当前文件

```
CS461_FinalProject/
├── train.csv              # 训练数据 (21,465条)
├── valid.csv              # 验证数据 (717条)
├── test.csv               # 测试数据 (967条)
├── requirements.txt       # Python依赖
├── predict_sarcasm.py     # 推理脚本（需完善）
├── models/                # 保存模型的地方
└── README.md              # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### 2. 开发和训练
建议使用 Jupyter Notebook 或 Google Colab 进行实验：
```bash
jupyter notebook
```

### 3. 训练流程建议
1. **数据探索** - 了解数据分布
2. **基线模型** - 逻辑回归 + TF-IDF（目标：F1 > 0.70）
3. **进阶模型** - SVM、LSTM、集成方法（目标：F1 > 0.75）
4. **完善推理脚本** - 确保 `predict_sarcasm.py` 能运行
5. **撰写报告** - 9个必需章节

---

## 📤 最终提交

需要提交一个ZIP文件，包含：

```
cs-461_final_project.zip
├── report.pdf              # 详细报告（9个章节）
├── predict_sarcasm.py      # 推理脚本
├── requirements.txt        # Python依赖
└── models/
    ├── model_weights.pkl   # 模型权重
    └── vectorizer.pkl      # 特征提取器
```

### 推理脚本使用方法
```bash
python predict_sarcasm.py --input test.csv --output predictions.csv
```

输出格式：CSV文件，包含 `text` 和 `prediction` 两列

---

## 📊 报告要求（9个章节）

1. **Introduction** - 问题描述、方法概述
2. **Data Exploration & Preprocessing** - 数据统计、预处理步骤
3. **Feature Engineering** - TF-IDF、词嵌入等
4. **Model Architecture & Selection** - 模型选择理由
5. **Training Methodology** - 训练过程、优化方法
6. **Experiments & Results** - ⚠️ 必须报告测试集的 F1/Precision/Recall/Accuracy
7. **Discussion** - 有效方法、局限性
8. **Conclusion** - 总结
9. **References** - 引用资源

---

## 💡 提示

- 从简单开始：逻辑回归 + TF-IDF是很好的基线
- 尝试不同特征：unigram, bigram, Word2Vec
- 考虑集成多个模型
- 注意防止过拟合（使用验证集、正则化）
- 所有路径使用相对路径

---

**评分**：报告15% + 公开测试5% + 隐藏测试10% = 30%
