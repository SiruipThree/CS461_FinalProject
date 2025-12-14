# CS461 Final Project - 文件结构

```
CS461_FinalProject/
│
├── data/                          # 数据文件夹
│   ├── train.csv                  # 训练数据（已有）
│   ├── valid.csv                  # 验证数据（已有）
│   └── test.csv                   # 测试数据（已有）
│
├── notebooks/                     # Jupyter笔记本（开发用）
│   ├── 01_data_exploration.ipynb  # 数据探索分析
│   ├── 02_preprocessing.ipynb     # 数据预处理实验
│   ├── 03_feature_engineering.ipynb # 特征工程
│   ├── 04_baseline_models.ipynb   # 基线模型
│   ├── 05_advanced_models.ipynb   # 进阶模型
│   └── 06_final_ensemble.ipynb    # 最终集成模型
│
├── src/                           # 源代码
│   ├── __init__.py
│   ├── data_loader.py             # 数据加载
│   ├── preprocessing.py           # 文本预处理
│   ├── feature_extraction.py      # 特征提取
│   ├── models.py                  # 模型定义
│   ├── train.py                   # 训练脚本
│   ├── evaluate.py                # 评估脚本
│   └── utils.py                   # 工具函数
│
├── models/                        # 保存的模型（最终提交）
│   ├── model_weights.pkl          # 主模型权重
│   ├── vectorizer.pkl             # 向量化器
│   └── config.json                # 模型配置
│
├── results/                       # 实验结果
│   ├── figures/                   # 图表
│   ├── metrics/                   # 指标记录
│   └── predictions/               # 预测结果
│
├── report/                        # 报告相关
│   ├── report.tex                 # LaTeX源文件（可选）
│   ├── figures/                   # 报告用图
│   └── report.pdf                 # 最终PDF报告
│
├── predict_sarcasm.py             # 推理脚本（提交用）
├── requirements.txt               # Python依赖
├── README.md                      # 项目说明
└── .gitignore                     # Git忽略文件

# 最终提交的ZIP文件结构：
cs-461_final_project.zip
├── report.pdf
├── requirements.txt
├── predict_sarcasm.py
├── models/
│   ├── model_weights.pkl
│   └── vectorizer.pkl
└── src/                          # (可选)
    └── utils.py
```


