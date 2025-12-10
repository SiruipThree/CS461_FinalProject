"""
数据加载模块
负责读取和初步处理CSV数据文件
"""

import pandas as pd
from typing import Tuple


def load_data(train_path: str, valid_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载训练、验证和测试数据
    
    Args:
        train_path: 训练数据路径
        valid_path: 验证数据路径
        test_path: 测试数据路径
    
    Returns:
        train_df, valid_df, test_df: 三个数据框
    """
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(valid_df)}")
    print(f"测试集大小: {len(test_df)}")
    
    return train_df, valid_df, test_df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """
    获取数据集的基本统计信息
    
    Args:
        df: 数据框
    
    Returns:
        stats: 统计信息字典
    """
    stats = {
        'total_samples': len(df),
        'label_distribution': df['label'].value_counts().to_dict(),
        'avg_text_length': df['text'].str.len().mean(),
        'max_text_length': df['text'].str.len().max(),
        'min_text_length': df['text'].str.len().min(),
    }
    
    return stats


if __name__ == "__main__":
    # 测试数据加载
    train_df, valid_df, test_df = load_data(
        '../data/train.csv',
        '../data/valid.csv',
        '../data/test.csv'
    )
    
    print("\n训练集统计:")
    stats = get_basic_stats(train_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")

