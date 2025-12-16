#!/usr/bin/env python3
"""
CS461 Final Project - Sarcasm Detection
推理脚本：用于对新数据进行预测

使用方法:
    python predict_sarcasm.py --input test_data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import pickle
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_ensemble_models():
    """
    加载所有ensemble模型和配置
    
    Returns:
        models: 字典，包含所有模型
        vectorizer: TF-IDF向量化器
        tokenizer: Keras tokenizer
        config: Ensemble配置
    """
    model_dir = Path(__file__).parent / 'models'
    
    try:
        # 加载传统ML模型（用于TF-IDF特征）
        print("⏳ Loading models...")
        
        with open(model_dir / 'model_weights.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        
        with open(model_dir / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(model_dir / 'xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        
        with open(model_dir / 'svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        # 加载深度学习模型
        from tensorflow import keras
        glove_model = keras.models.load_model(model_dir / 'final_glove_bilstm.h5')
        
        with open(model_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # 加载ensemble配置
        with open(model_dir / 'ensemble_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        models = {
            'lr': lr_model,
            'xgboost': xgb_model,
            'svm': svm_model,
            'glove_bilstm': glove_model
        }
        
        print("✅ All models loaded successfully!")
        print(f"   Ensemble weights: {config['weights']}")
        
        return models, vectorizer, tokenizer, config
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
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
    # 基本预处理：转小写
    if isinstance(text, str):
        text = text.lower()
    else:
        text = ""
    
    return text


def predict_ensemble(texts, models, vectorizer, tokenizer, config):
    """
    使用ensemble模型进行预测
    
    Args:
        texts: 文本列表
        models: 模型字典
        vectorizer: TF-IDF向量化器
        tokenizer: Keras tokenizer
        config: Ensemble配置
    
    Returns:
        predictions: 预测结果数组
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # 准备TF-IDF特征（用于LR, XGBoost, SVM）
    X_tfidf = vectorizer.transform(texts)
    
    # 准备序列特征（用于BiLSTM）
    sequences = tokenizer.texts_to_sequences(texts)
    MAX_LEN = 100  # 与训练时相同
    X_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 获取各模型的预测概率
    lr_prob = models['lr'].predict_proba(X_tfidf)[:, 1]
    xgb_prob = models['xgboost'].predict_proba(X_tfidf)[:, 1]
    svm_prob = models['svm'].predict_proba(X_tfidf)[:, 1]
    bilstm_prob = models['glove_bilstm'].predict(X_seq, verbose=0).flatten()
    
    # 加权平均
    weights = config['weights']
    ensemble_prob = (
        weights['Logistic Regression'] * lr_prob +
        weights['XGBoost'] * xgb_prob +
        weights['SVM'] * svm_prob +
        weights['BiLSTM+GloVe'] * bilstm_prob
    )
    
    # 转换为二分类预测
    predictions = (ensemble_prob > 0.5).astype(int)
    
    return predictions


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
        print(f"\n📖 Successfully read {len(df)} samples")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found - {input_path}")
        sys.exit(1)
    
    # 检查是否包含'text'列
    if 'text' not in df.columns:
        print("❌ Error: Input file must contain 'text' column")
        sys.exit(1)
    
    # 加载模型
    models, vectorizer, tokenizer, config = load_ensemble_models()
    
    # 预处理文本
    print("\n⏳ Preprocessing texts...")
    texts = df['text'].apply(preprocess_text).tolist()
    
    # 使用ensemble进行预测
    print("⏳ Making predictions with ensemble model...")
    predictions = predict_ensemble(texts, models, vectorizer, tokenizer, config)
    
    # 保存结果
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    output_df.to_csv(output_path, index=False)
    
    # 统计结果
    pred_counts = output_df['prediction'].value_counts().to_dict()
    non_sarcastic = pred_counts.get(0, 0)
    sarcastic = pred_counts.get(1, 0)
    
    print(f"\n✅ Prediction completed!")
    print(f"📁 Results saved to: {output_path}")
    print(f"\n📊 Prediction Distribution:")
    print(f"   Non-Sarcastic (0): {non_sarcastic} ({non_sarcastic/len(df)*100:.1f}%)")
    print(f"   Sarcastic (1):     {sarcastic} ({sarcastic/len(df)*100:.1f}%)")
    print(f"   Total:             {len(df)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Sarcasm Detection - Ensemble Model Inference'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path (must contain "text" column)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path (will contain "text" and "prediction" columns)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CS461 SARCASM DETECTION - ENSEMBLE INFERENCE")
    print("="*80)
    
    # 执行预测
    predict(args.input, args.output)
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
