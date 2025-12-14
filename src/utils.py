"""
工具函数模块
包含常用的辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import joblib
import json


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    评估模型性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
    
    Returns:
        metrics: 指标字典
    """
    print(f"\n{'='*50}")
    print(f"{model_name} 性能评估")
    print(f"{'='*50}")
    
    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print()
    
    # 详细分类报告
    print("详细分类报告:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Non-Sarcastic', 'Sarcastic']))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics


def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    保存模型和向量化器
    
    Args:
        model: 训练好的模型
        vectorizer: 特征提取器
        model_path: 模型保存路径
        vectorizer_path: 向量化器保存路径
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"模型已保存至: {model_path}")
    print(f"向量化器已保存至: {vectorizer_path}")


def load_model(model_path, vectorizer_path):
    """
    加载模型和向量化器
    
    Args:
        model_path: 模型路径
        vectorizer_path: 向量化器路径
    
    Returns:
        model, vectorizer: 加载的模型和向量化器
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"模型已从 {model_path} 加载")
    print(f"向量化器已从 {vectorizer_path} 加载")
    return model, vectorizer


def save_config(config_dict, config_path):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"配置已保存至: {config_path}")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


