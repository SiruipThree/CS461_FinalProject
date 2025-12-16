#!/usr/bin/env python3
"""
Complete translation of Chinese to English in Jupyter Notebook
"""

import json
import re

# Comprehensive translation dictionary
COMPLETE_TRANSLATIONS = {
    # Main headers
    "CS461 讽刺检测项目": "CS461 Sarcasm Detection Project",
    "这个笔记本用于所有实验": "This notebook contains all experiments",
    "流程": "Workflow",
    "第一步": "Step 1",
    "第二步": "Step 2",
    "第三步": "Step 3",
    "第四步": "Step 4",
    "第五步": "Step 5",
    "第六步": "Step 6",
    
    # Steps and sections
    "加载库和数据": "Import Libraries and Load Data",
    "数据探索": "Data Exploration",
    "数据预处理": "Data Preprocessing",
    "特征工程": "Feature Engineering",
    "模型训练": "Model Training",
    "模型评估": "Model Evaluation",
    "结果分析": "Results Analysis",
    "保存模型": "Save Model",
    
    # Common phrases
    "训练Baseline Model": "Train Baseline Model",
    "尝试进阶model": "Try advanced models",
    "保存Best model": "Save best model",
    "记录所有results": "Record all results",
    "用于报告": "for the report",
    
    # Data loading
    "加载数据": "Loading data",
    "读取数据": "Reading data",
    "数据加载完成": "Data loading completed",
    
    # Labels and metrics
    "labels分布": "Label Distribution",
    "标签分布": "Label Distribution",
    "查看前几行": "View first few rows",
    "查看样例": "View examples",
    "样本": "samples",
    "条": "samples",
    
    # Text analysis
    "文本长度": "Text length",
    "平均文本长度": "Average text length",
    "字符": "characters",
    
    # Examples
    "讽刺样例": "Sarcastic examples",
    "非讽刺样例": "Non-sarcastic examples",
    "例子": "examples",
    
    # Model descriptions
    "基线模型": "Baseline Model",
    "逻辑回归": "Logistic Regression",
    "支持向量机": "Support Vector Machine",
    "梯度提升": "Gradient Boosting",
    "神经网络": "Neural Network",
    "深度学习模型": "Deep Learning Model",
    "集成模型": "Ensemble Model",
    
    # Training related
    "训练中": "Training in progress",
    "训练完成": "Training completed",
    "开始训练": "Starting training",
    "正在训练": "Currently training",
    "训练时间": "Training time",
    
    # Performance
    "性能评估": "Performance Evaluation",
    "测试集性能": "Test Set Performance",
    "验证集性能": "Validation Set Performance",
    "训练集性能": "Training Set Performance",
    
    # Metrics names
    "准确率": "Accuracy",
    "精确率": "Precision",
    "召回率": "Recall",
    "F1分数": "F1 Score",
    "损失": "Loss",
    "混淆矩阵": "Confusion Matrix",
    
    # Comparisons
    "对比": "Comparison",
    "比较": "Comparison",
    "最佳": "Best",
    "最优": "Optimal",
    "最终": "Final",
    "改进": "Improvement",
    "提升": "Improvement",
    "降低": "Reduction",
    
    # Status
    "成功": "Success",
    "完成": "Completed",
    "失败": "Failed",
    "错误": "Error",
    "警告": "Warning",
    "注意": "Note",
    "提示": "Tip",
    
    # Sets
    "训练集": "Training set",
    "验证集": "Validation set",
    "测试集": "Test set",
    "数据集": "Dataset",
    
    # Results
    "结果": "Results",
    "输出": "Output",
    "预测": "Prediction",
    "预测结果": "Prediction results",
    
    # File operations
    "保存到": "Saved to",
    "加载自": "Loaded from",
    "文件": "file",
    "路径": "path",
    "模型已保存": "Model saved",
    "模型加载成功": "Model loaded successfully",
    
    # Technical terms
    "超参数": "Hyperparameters",
    "正则化": "Regularization",
    "过拟合": "Overfitting",
    "欠拟合": "Underfitting",
    "交叉验证": "Cross-validation",
    "早停": "Early Stopping",
    "学习率": "Learning rate",
    "批次大小": "Batch size",
    "迭代次数": "Number of iterations",
    "轮次": "Epochs",
    
    # Numbers and units
    "分钟": "minutes",
    "小时": "hours",
    "秒": "seconds",
    "个": "",
    "次": "times",
    
    # Actions
    "运行": "Run",
    "执行": "Execute",
    "计算": "Calculate",
    "显示": "Display",
    "打印": "Print",
    
    # Additional common phrases
    "如下": "as follows",
    "详细": "Detailed",
    "总结": "Summary",
    "概述": "Overview",
    "分析": "Analysis",
    "讨论": "Discussion",
    "结论": "Conclusion",
    "建议": "Recommendations",
}

def translate_text(text):
    """
    Translate Chinese text to English with comprehensive dictionary
    """
    if not isinstance(text, str):
        return text
    
    # Sort by length (longest first) for better matching
    for chinese, english in sorted(COMPLETE_TRANSLATIONS.items(), key=lambda x: -len(x[0])):
        text = text.replace(chinese, english)
    
    return text

def translate_notebook(input_file, output_file):
    """Translate all Chinese content in Jupyter Notebook"""
    print(f"📖 Reading notebook: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells_modified = 0
    total_cells = len(notebook['cells'])
    
    print(f"🔄 Processing {total_cells} cells...")
    
    for i, cell in enumerate(notebook['cells'], 1):
        cell_modified = False
        
        if cell['cell_type'] in ['markdown', 'code']:
            # Translate source
            if 'source' in cell and cell['source']:
                original = cell['source']
                if isinstance(original, list):
                    translated = [translate_text(line) for line in original]
                else:
                    translated = translate_text(original)
                
                if translated != original:
                    cell['source'] = translated
                    cell_modified = True
            
            # Translate code cell outputs
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    # Translate text output
                    if 'text' in output:
                        original_text = output['text']
                        if isinstance(original_text, list):
                            output['text'] = [translate_text(line) for line in original_text]
                        else:
                            output['text'] = translate_text(original_text)
                        
                        if output['text'] != original_text:
                            cell_modified = True
                    
                    # Translate data/text/plain
                    if 'data' in output and 'text/plain' in output['data']:
                        original_data = output['data']['text/plain']
                        if isinstance(original_data, list):
                            output['data']['text/plain'] = [translate_text(line) for line in original_data]
                        else:
                            output['data']['text/plain'] = translate_text(original_data)
                        
                        if output['data']['text/plain'] != original_data:
                            cell_modified = True
        
        if cell_modified:
            cells_modified += 1
            if cells_modified % 10 == 0:
                print(f"  Processed {cells_modified}/{total_cells} cells...")
    
    print(f"✅ Modified {cells_modified} cells with Chinese content")
    print(f"💾 Writing translated notebook: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("\n" + "="*70)
    print("🎉 TRANSLATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Original notebook:   {input_file}")
    print(f"Translated notebook: {output_file}")
    print(f"Cells modified:      {cells_modified}/{total_cells}")
    print("="*70)
    print("\n✅ The English version is ready for your American classmate!")

if __name__ == '__main__':
    translate_notebook('SarcasmDetection.ipynb', 'SarcasmDetection_EN.ipynb')

