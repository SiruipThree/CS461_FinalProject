#!/usr/bin/env python3
"""
CS461 Final Project - Sarcasm Detection
Inference script for making predictions on new data

Usage:
    python predict_sarcasm.py --input test.csv --output predictions.csv
"""

import argparse
import pandas as pd
import pickle
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_stacking_models():
    """
    Load Stacking Ensemble models (CharCNN + BiLSTM + Meta-learner)
    
    Returns:
        charcnn_model: Character-level CNN
        char_tokenizer: Character tokenizer
        bilstm_model: BiLSTM with GloVe embeddings
        word_tokenizer: Word tokenizer
        meta_learner: Logistic Regression meta-learner
        config: Model configuration
    """
    model_dir = Path(__file__).parent / 'models'
    
    try:
        print("[LOADING] Loading Stacking Ensemble models...")
        
        # Load CharCNN model
        from tensorflow import keras
        charcnn_model = keras.models.load_model(model_dir / 'charcnn_model.h5', compile=False)
        
        with open(model_dir / 'char_tokenizer.pkl', 'rb') as f:
            char_tokenizer = pickle.load(f)
        
        # Load BiLSTM + GloVe model
        bilstm_model = keras.models.load_model(model_dir / 'final_glove_bilstm.h5', compile=False)
        
        with open(model_dir / 'tokenizer.pkl', 'rb') as f:
            word_tokenizer = pickle.load(f)
        
        # Load meta-learner (LR stacker)
        with open(model_dir / 'meta_learner.pkl', 'rb') as f:
            meta_learner = pickle.load(f)
        
        # Load configuration
        with open(model_dir / 'stacking_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        print("[OK] All models loaded successfully!")
        print(f"[INFO] Model type: {config['model_type']}")
        print(f"[INFO] Base models: {', '.join(config['base_models'])}")
        print(f"[INFO] Test accuracy: {config['test_accuracy']:.4f}")
        
        return charcnn_model, char_tokenizer, bilstm_model, word_tokenizer, meta_learner, config
        
    except FileNotFoundError as e:
        print(f"[X] Error: Model file not found - {e}")
        print("[TIP] Please ensure all model files are in the 'models/' directory:")
        print("  - charcnn_model.h5")
        print("  - char_tokenizer.pkl")
        print("  - final_glove_bilstm.h5")
        print("  - tokenizer.pkl")
        print("  - meta_learner.pkl")
        print("  - stacking_config.pkl")
        sys.exit(1)
    except Exception as e:
        print(f"[X] Error loading models: {e}")
        sys.exit(1)


def preprocess_text(texts, char_tokenizer, word_tokenizer, max_chars, max_words):
    """
    Preprocess text for both CharCNN and BiLSTM
    
    Args:
        texts: List of text strings
        char_tokenizer: Character tokenizer
        word_tokenizer: Word tokenizer
        max_chars: Maximum character sequence length
        max_words: Maximum word sequence length
    
    Returns:
        X_char: Character sequences for CharCNN
        X_word: Word sequences for BiLSTM
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Character-level preprocessing for CharCNN
    char_sequences = char_tokenizer.texts_to_sequences(texts)
    X_char = pad_sequences(char_sequences, maxlen=max_chars, padding='post', truncating='post')
    
    # Word-level preprocessing for BiLSTM
    word_sequences = word_tokenizer.texts_to_sequences(texts)
    X_word = pad_sequences(word_sequences, maxlen=max_words, padding='post', truncating='post')
    
    return X_char, X_word


def predict(input_path, output_path):
    """
    Make predictions on input data using Stacking Ensemble
    
    Args:
        input_path: Path to input CSV file (must have 'text' column)
        output_path: Path to output CSV file (will have 'text' and 'prediction' columns)
    """
    # Load data
    print(f"\n[LOADING] Reading input data: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[X] Error reading input file: {e}")
        sys.exit(1)
    
    # Validate input
    if 'text' not in df.columns:
        print("[X] Error: Input CSV must contain a 'text' column")
        sys.exit(1)
    
    print(f"[OK] Loaded {len(df)} samples")
    
    # Load models
    charcnn_model, char_tokenizer, bilstm_model, word_tokenizer, meta_learner, config = load_stacking_models()
    
    # Get configuration
    max_chars = config.get('max_chars', 300)
    max_words = config.get('max_words_len', 100)
    
    # Preprocess text
    print("\n[PROCESSING] Preprocessing text...")
    texts = df['text'].fillna('').astype(str).tolist()
    X_char, X_word = preprocess_text(texts, char_tokenizer, word_tokenizer, max_chars, max_words)
    
    print(f"[OK] Character sequences shape: {X_char.shape}")
    print(f"[OK] Word sequences shape: {X_word.shape}")
    
    # Get predictions from base models
    print("\n[PREDICT] Getting predictions from base models...")
    
    # CharCNN predictions
    print("[INFO] CharCNN predicting...")
    charcnn_probs = charcnn_model.predict(X_char, verbose=0).flatten()
    
    # BiLSTM predictions
    print("[INFO] BiLSTM predicting...")
    bilstm_probs = bilstm_model.predict(X_word, verbose=0).flatten()
    
    # Stack predictions
    X_meta = np.column_stack([charcnn_probs, bilstm_probs])
    
    # Meta-learner final prediction
    print("[INFO] Meta-learner combining predictions...")
    final_predictions = meta_learner.predict(X_meta)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': final_predictions
    })
    
    # Save predictions
    print(f"\n[SAVING] Writing predictions to: {output_path}")
    try:
        output_df.to_csv(output_path, index=False)
        print(f"[OK] Successfully saved {len(output_df)} predictions")
    except Exception as e:
        print(f"[X] Error saving output file: {e}")
        sys.exit(1)
    
    # Summary statistics
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(f"Total samples:        {len(output_df)}")
    print(f"Predicted sarcastic:  {sum(final_predictions)} ({sum(final_predictions)/len(final_predictions)*100:.1f}%)")
    print(f"Predicted normal:     {len(final_predictions) - sum(final_predictions)} ({(len(final_predictions) - sum(final_predictions))/len(final_predictions)*100:.1f}%)")
    print("="*70)
    print("\n[SUCCESS] Prediction completed successfully!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Sarcasm Detection - Predict sarcasm in text using Stacking Ensemble',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_sarcasm.py --input test.csv --output predictions.csv
  python predict_sarcasm.py -i data.csv -o results.csv

Input CSV format:
  Must contain a 'text' column with the text to classify

Output CSV format:
  Contains 'text' and 'prediction' columns (0=not sarcastic, 1=sarcastic)
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file (must have "text" column)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output CSV file (will contain "text" and "prediction" columns)'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict(args.input, args.output)


if __name__ == '__main__':
    main()
