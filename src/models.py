"""
模型訓練模組
包含邏輯迴歸、樸素貝葉斯、支援向量機等分類器
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Tuple, Dict, Any
import joblib
import os

from src.data_preprocessing import load_data, preprocess_data, create_vectorizer, prepare_features


class SpamClassifier:
    """垃圾郵件分類器類別"""
    
    def __init__(self, model_type: str = 'logistic_regression', vectorizer_type: str = 'tfidf'):
        """
        初始化分類器
        
        Args:
            model_type: 模型類型 ('logistic_regression', 'naive_bayes', 'svm')
            vectorizer_type: 向量化器類型 ('tfidf' 或 'count')
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.vectorizer = create_vectorizer(vectorizer_type)
        self.model = self._create_model(model_type)
        self.is_trained = False
    
    def _create_model(self, model_type: str):
        """
        建立模型
        
        Args:
            model_type: 模型類型
            
        Returns:
            模型實例
        """
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            return MultinomialNB()
        elif model_type == 'svm':
            return SVC(kernel='linear', probability=True, random_state=42)
        else:
            raise ValueError(f"不支援的模型類型: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        訓練模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Args:
            X: 特徵
            
        Returns:
            預測結果
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        預測機率
        
        Args:
            X: 特徵
            
        Returns:
            預測機率
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        return self.model.predict_proba(X)
    
    def save(self, model_dir: str = 'models'):
        """
        保存模型
        
        Args:
            model_dir: 模型保存目錄
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(model_dir, f'{self.model_type}_model.joblib')
        joblib.dump(self.model, model_path)
        
        # 保存向量化器
        vectorizer_path = os.path.join(model_dir, f'{self.vectorizer_type}_vectorizer.joblib')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"模型已保存至: {model_path}")
        print(f"向量化器已保存至: {vectorizer_path}")
    
    def load(self, model_dir: str = 'models'):
        """
        載入模型
        
        Args:
            model_dir: 模型目錄
        """
        # 載入模型
        model_path = os.path.join(model_dir, f'{self.model_type}_model.joblib')
        self.model = joblib.load(model_path)
        
        # 載入向量化器
        vectorizer_path = os.path.join(model_dir, f'{self.vectorizer_type}_vectorizer.joblib')
        self.vectorizer = joblib.load(vectorizer_path)
        
        self.is_trained = True
        print(f"模型已載入: {model_path}")


def train_all_models(data_path: str = 'datasets/sms_spam_no_header.csv', 
                     test_size: float = 0.2, random_state: int = 42) -> Dict[str, SpamClassifier]:
    """
    訓練所有模型
    
    Args:
        data_path: 資料集路徑
        test_size: 測試集比例
        random_state: 隨機種子
        
    Returns:
        訓練好的模型字典
    """
    print("載入資料...")
    df = load_data(data_path)
    
    print("預處理資料...")
    df = preprocess_data(df)
    
    # 準備特徵和標籤
    print("準備特徵...")
    vectorizer = create_vectorizer('tfidf')
    X = prepare_features(df, vectorizer, fit=True)
    y = df['label'].values
    
    # 分割資料集
    print("分割資料集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")
    
    # 訓練所有模型
    models = {}
    model_types = ['logistic_regression', 'naive_bayes', 'svm']
    
    for model_type in model_types:
        print(f"\n訓練 {model_type} 模型...")
        classifier = SpamClassifier(model_type=model_type, vectorizer_type='tfidf')
        classifier.vectorizer = vectorizer  # 使用相同的向量化器
        classifier.train(X_train, y_train)
        classifier.save()
        models[model_type] = classifier
        print(f"{model_type} 模型訓練完成！")
    
    return models, X_test, y_test


if __name__ == '__main__':
    # 訓練所有模型
    models, X_test, y_test = train_all_models()
    print("\n所有模型訓練完成！")

