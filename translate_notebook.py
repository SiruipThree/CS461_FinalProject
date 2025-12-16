#!/usr/bin/env python3
"""
Translate Chinese text in Jupyter Notebook to English
"""

import json
import re

# Translation dictionary for common terms and phrases
TRANSLATIONS = {
    # Headers and titles
    "讽刺检测": "Sarcasm Detection",
    "数据探索": "Data Exploration",
    "基线模型": "Baseline Model",
    "模型训练": "Model Training",
    "评估": "Evaluation",
    "预测": "Prediction",
    "准确率": "Accuracy",
    "精确率": "Precision",
    "召回率": "Recall",
    "混淆矩阵": "Confusion Matrix",
    
    # Common phrases
    "训练集": "Training set",
    "验证集": "Validation set",
    "测试集": "Test set",
    "样本": "samples",
    "特征": "features",
    "标签": "labels",
    "模型": "model",
    "结果": "results",
    "性能": "performance",
    "提升": "improvement",
    "对比": "comparison",
    
    # Status messages
    "加载数据": "Loading data",
    "训练中": "Training",
    "完成": "Completed",
    "成功": "Success",
    "失败": "Failed",
    "错误": "Error",
    
    # Model names
    "逻辑回归": "Logistic Regression",
    "支持向量机": "Support Vector Machine",
    "随机森林": "Random Forest",
    "梯度提升": "Gradient Boosting",
    "神经网络": "Neural Network",
    "深度学习": "Deep Learning",
    "集成方法": "Ensemble Method",
    
    # Technical terms
    "超参数": "hyperparameters",
    "正则化": "regularization",
    "过拟合": "overfitting",
    "欠拟合": "underfitting",
    "交叉验证": "cross-validation",
    "早停": "early stopping",
    "学习率": "learning rate",
    "批次大小": "batch size",
    "迭代": "iteration",
    "轮次": "epoch",
    
    # Analysis
    "分析": "Analysis",
    "总结": "Summary",
    "结论": "Conclusion",
    "建议": "Recommendation",
    "改进": "Improvement",
    "优化": "Optimization",
}

# Comprehensive translation mappings
PHRASE_TRANSLATIONS = {
    # Complete sentences and phrases
    "# CS461 Final Project - 讽刺检测": "# CS461 Final Project - Sarcasm Detection",
    "训练一个分类模型检测文本中的讽刺": "Train a classification model to detect sarcasm in text",
    "数据加载和探索": "Data Loading and Exploration",
    "特征工程": "Feature Engineering",
    "模型架构": "Model Architecture",
    "训练方法": "Training Methodology",
    "实验结果": "Experimental Results",
    "讨论": "Discussion",
    "参考文献": "References",
    
    # Common output messages
    "模型加载成功": "Model loaded successfully",
    "开始训练": "Starting training",
    "训练完成": "Training completed",
    "保存模型": "Saving model",
    "模型已保存": "Model saved",
    "正在预测": "Making predictions",
    "预测完成": "Prediction completed",
    
    # Performance descriptions
    "在测试集上": "On test set",
    "在训练集上": "On training set",
    "在验证集上": "On validation set",
    "最佳模型": "Best model",
    "最终结果": "Final results",
    "性能指标": "Performance metrics",
    
    # Comparisons
    "相比": "compared to",
    "优于": "better than",
    "提升了": "improved by",
    "降低了": "decreased by",
    "达到": "achieved",
    "超过": "exceeded",
    
    # Instructions
    "请运行": "Please run",
    "请注意": "Please note",
    "注意": "Note",
    "提示": "Tip",
    "警告": "Warning",
    "重要": "Important",
    
    # Time and process
    "需要": "requires",
    "大约": "approximately",
    "分钟": "minutes",
    "小时": "hours",
    "正在": "currently",
    "已完成": "completed",
    
    # Results and metrics
    "准确率：": "Accuracy:",
    "精确率：": "Precision:",
    "召回率：": "Recall:",
    "F1分数：": "F1 Score:",
    "损失：": "Loss:",
    
    # Data descriptions
    "条数据": "samples",
    "个特征": "features",
    "维": "dimensions",
    "类别": "classes",
    
    # Model components
    "输入层": "Input layer",
    "隐藏层": "Hidden layer",
    "输出层": "Output layer",
    "激活函数": "Activation function",
    "损失函数": "Loss function",
    "优化器": "Optimizer",
    
    # File operations
    "读取": "Reading",
    "写入": "Writing",
    "保存到": "Saving to",
    "加载自": "Loading from",
    "文件": "file",
    "路径": "path",
}

def translate_text(text):
    """Translate Chinese text to English"""
    if not isinstance(text, str):
        return text
    
    # First apply phrase translations (longer matches first)
    for chinese, english in sorted(PHRASE_TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        text = text.replace(chinese, english)
    
    # Then apply word-level translations
    for chinese, english in TRANSLATIONS.items():
        text = text.replace(chinese, english)
    
    return text

def translate_notebook(input_file, output_file):
    """Translate Chinese content in Jupyter Notebook to English"""
    print(f"Reading notebook: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_translated = 0
    total_cells = len(notebook['cells'])
    
    print(f"Translating {total_cells} cells...")
    
    for cell in notebook['cells']:
        if cell['cell_type'] in ['markdown', 'code']:
            # Translate source content
            if 'source' in cell:
                original_source = cell['source']
                if isinstance(original_source, list):
                    cell['source'] = [translate_text(line) for line in original_source]
                else:
                    cell['source'] = translate_text(original_source)
                
                if cell['source'] != original_source:
                    cells_translated += 1
            
            # Translate outputs (for code cells)
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        if isinstance(output['text'], list):
                            output['text'] = [translate_text(line) for line in output['text']]
                        else:
                            output['text'] = translate_text(output['text'])
                    
                    if 'data' in output and 'text/plain' in output['data']:
                        if isinstance(output['data']['text/plain'], list):
                            output['data']['text/plain'] = [translate_text(line) for line in output['data']['text/plain']]
                        else:
                            output['data']['text/plain'] = translate_text(output['data']['text/plain'])
    
    print(f"Translated content in {cells_translated} cells")
    print(f"Writing translated notebook: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("✅ Translation completed successfully!")
    print(f"\nOriginal notebook: {input_file}")
    print(f"Translated notebook: {output_file}")

if __name__ == '__main__':
    input_notebook = 'SarcasmDetection.ipynb'
    output_notebook = 'SarcasmDetection_EN.ipynb'
    
    translate_notebook(input_notebook, output_notebook)

