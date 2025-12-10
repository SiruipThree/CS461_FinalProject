#!/usr/bin/env python3
"""
CS461 Final Project - Sarcasm Detection
推理脚本：用于对新数据进行预测

使用方法:
    python predict_sarcasm.py --input test_data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import joblib
import sys
from pathlib import Path


def load_models():
    """
    加载训练好的模型和向量化器
    
    Returns:
        model: 训练好的分类模型
        vectorizer: 特征向量化器
    """
    model_dir = Path(__file__).parent / 'models'
    
    try:
        model = joblib.load(model_dir / 'model_weights.pkl')
        vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
        print("模型加载成功！")
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"错误: 找不到模型文件 - {e}")
        sys.exit(1)


def preprocess_text(text):
    """
    文本预处理函数
    根据训练时使用的预处理方法进行相同处理
    
    Args:
        text: 原始文本
    
    Returns:
        processed_text: 处理后的文本
    """
    # TODO: 根据你的训练代码实现相同的预处理逻辑
    # 示例: 转小写
    text = text.lower()
    
    # 更多预处理步骤...
    
    return text


def predict(input_path, output_path):
    """
    对输入文件进行预测并保存结果
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
    """
    # 读取输入数据
    try:
        df = pd.read_csv(input_path)
        print(f"成功读取 {len(df)} 条数据")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_path}")
        sys.exit(1)
    
    # 检查是否包含'text'列
    if 'text' not in df.columns:
        print("错误: 输入文件必须包含'text'列")
        sys.exit(1)
    
    # 加载模型
    model, vectorizer = load_models()
    
    # 预处理文本
    print("正在预处理文本...")
    texts = df['text'].apply(preprocess_text).tolist()
    
    # 特征提取
    print("正在提取特征...")
    X = vectorizer.transform(texts)
    
    # 预测
    print("正在进行预测...")
    predictions = model.predict(X)
    
    # 保存结果
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"预测完成！结果已保存至: {output_path}")
    print(f"预测标签分布: {output_df['prediction'].value_counts().to_dict()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Sarcasm Detection - 讽刺检测推理脚本'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入CSV文件路径（必须包含text列）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出CSV文件路径（包含text和prediction两列）'
    )
    
    args = parser.parse_args()
    
    # 执行预测
    predict(args.input, args.output)


if __name__ == '__main__':
    main()

