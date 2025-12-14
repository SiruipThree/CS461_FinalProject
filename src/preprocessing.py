"""
文本预处理模块
包含文本清洗、规范化等功能
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# 下载必要的NLTK数据（首次运行时需要）
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


class TextPreprocessor:
    """文本预处理类"""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = False,
                 lemmatize: bool = False,
                 stem: bool = False):
        """
        初始化预处理器
        
        Args:
            lowercase: 是否转小写
            remove_punctuation: 是否去除标点
            remove_numbers: 是否去除数字
            remove_stopwords: 是否去除停用词
            lemmatize: 是否词形还原
            stem: 是否词干提取
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        if self.stem:
            self.stemmer = PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 原始文本
        
        Returns:
            cleaned_text: 清洗后的文本
        """
        # 转小写
        if self.lowercase:
            text = text.lower()
        
        # 去除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 去除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 去除数字
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # 去除标点
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 去除多余空白
        text = ' '.join(text.split())
        
        # 分词
        words = text.split()
        
        # 去除停用词
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words]
        
        # 词形还原
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        # 词干提取
        if self.stem:
            words = [self.stemmer.stem(w) for w in words]
        
        return ' '.join(words)
    
    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
        
        Returns:
            processed_texts: 处理后的文本列表
        """
        return [self.clean_text(text) for text in texts]


if __name__ == "__main__":
    # 测试预处理
    sample_texts = [
        "This is a GREAT example of sarcasm!!!",
        "Trump's latest policy is just amazing... NOT!",
        "Visit our website at https://example.com for more info."
    ]
    
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,
        remove_stopwords=False,
        lemmatize=False
    )
    
    for text in sample_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"原文: {text}")
        print(f"清洗后: {cleaned}")
        print()


