"""
資料預處理模組
處理 SMS 文本數據，包括清洗、分詞、向量化等功能
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from typing import Tuple, List
import joblib

# 下載 NLTK 資料（如果尚未下載）
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """文本預處理類別"""
    
    def __init__(self):
        """初始化預處理器"""
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗後的文本
        """
        # 轉換為小寫
        text = text.lower()
        
        # 移除特殊字符和數字，只保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 移除多餘的空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        分詞
        
        Args:
            text: 文本
            
        Returns:
            詞彙列表
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        移除停用詞
        
        Args:
            tokens: 詞彙列表
            
        Returns:
            移除停用詞後的詞彙列表
        """
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def preprocess(self, text: str) -> str:
        """
        完整的預處理流程
        
        Args:
            text: 原始文本
            
        Returns:
            預處理後的文本（以空格連接的詞彙）
        """
        # 清洗
        cleaned = self.clean_text(text)
        
        # 分詞
        tokens = self.tokenize(cleaned)
        
        # 移除停用詞
        filtered_tokens = self.remove_stopwords(tokens)
        
        # 重新組合為文本
        return ' '.join(filtered_tokens)


def load_data(file_path: str) -> pd.DataFrame:
    """
    載入資料集
    
    Args:
        file_path: CSV 文件路徑
        
    Returns:
        包含 label 和 text 列的 DataFrame
    """
    # 讀取 CSV 文件（無標題行）
    df = pd.read_csv(file_path, header=None, names=['label', 'text'])
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    預處理資料集
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        預處理後的 DataFrame
    """
    preprocessor = TextPreprocessor()
    
    # 應用預處理
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    
    return df


def create_vectorizer(vectorizer_type: str = 'tfidf', **kwargs) -> TfidfVectorizer | CountVectorizer:
    """
    建立向量化器
    
    Args:
        vectorizer_type: 向量化器類型 ('tfidf' 或 'count')
        **kwargs: 向量化器參數
        
    Returns:
        向量化器實例
    """
    default_params = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95
    }
    default_params.update(kwargs)
    
    if vectorizer_type == 'tfidf':
        return TfidfVectorizer(**default_params)
    elif vectorizer_type == 'count':
        return CountVectorizer(**default_params)
    else:
        raise ValueError(f"不支援的向量化器類型: {vectorizer_type}")


def prepare_features(df: pd.DataFrame, vectorizer: TfidfVectorizer | CountVectorizer, 
                     fit: bool = True) -> np.ndarray:
    """
    準備特徵向量
    
    Args:
        df: 包含 processed_text 列的 DataFrame
        vectorizer: 向量化器
        fit: 是否擬合向量化器
        
    Returns:
        特徵矩陣
    """
    if fit:
        features = vectorizer.fit_transform(df['processed_text'])
    else:
        features = vectorizer.transform(df['processed_text'])
    
    return features


if __name__ == '__main__':
    # 測試預處理功能
    print("載入資料...")
    df = load_data('datasets/sms_spam_no_header.csv')
    print(f"資料集大小: {df.shape}")
    print(f"標籤分布:\n{df['label'].value_counts()}")
    
    print("\n預處理資料...")
    df_processed = preprocess_data(df)
    print("預處理完成！")
    print(f"\n原始文本範例:\n{df['text'].iloc[0]}")
    print(f"\n預處理後文本範例:\n{df_processed['processed_text'].iloc[0]}")

