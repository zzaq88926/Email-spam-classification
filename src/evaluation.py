"""
模型評估模組
計算評估指標並生成視覺化圖表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, Tuple, List
import os

# 設定中文字體（用於 matplotlib）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    計算評估指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        
    Returns:
        評估指標字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='spam', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='spam', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label='spam', zero_division=0)
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: str = None):
    """
    繪製混淆矩陣
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        model_name: 模型名稱
        save_path: 保存路徑（可選）
    """
    cm = confusion_matrix(y_true, y_pred, labels=['ham', 'spam'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'{model_name} - 混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"混淆矩陣已保存至: {save_path}")
    
    return plt.gcf()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, 
                   model_name: str, save_path: str = None):
    """
    繪製 ROC 曲線
    
    Args:
        y_true: 真實標籤
        y_proba: 預測機率（spam 類別的機率）
        model_name: 模型名稱
        save_path: 保存路徑（可選）
        
    Returns:
        AUC 分數
    """
    # 將標籤轉換為二進制（spam=1, ham=0）
    y_binary = (y_true == 'spam').astype(int)
    
    fpr, tpr, thresholds = roc_curve(y_binary, y_proba)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='隨機猜測')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假陽性率 (False Positive Rate)')
    plt.ylabel('真陽性率 (True Positive Rate)')
    plt.title(f'{model_name} - ROC 曲線')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"ROC 曲線已保存至: {save_path}")
    
    return plt.gcf(), auc_score


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    生成分類報告
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        
    Returns:
        分類報告字符串
    """
    report = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])
    return report


def compare_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                  save_dir: str = 'results') -> pd.DataFrame:
    """
    比較多個模型的性能
    
    Args:
        models: 模型字典
        X_test: 測試特徵
        y_test: 測試標籤
        save_dir: 結果保存目錄
        
    Returns:
        比較結果 DataFrame
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    
    for model_name, classifier in models.items():
        # 預測
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]  # spam 類別的機率
        
        # 計算指標
        metrics = calculate_metrics(y_test, y_pred)
        metrics['model'] = model_name
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_test, y_pred, model_name, 
                            save_path=os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # 繪製 ROC 曲線
        _, auc_score = plot_roc_curve(y_test, y_proba, model_name,
                                     save_path=os.path.join(save_dir, f'{model_name}_roc_curve.png'))
        plt.close()
        
        metrics['auc'] = auc_score
        results.append(metrics)
        
        # 生成分類報告
        report = generate_classification_report(y_test, y_pred)
        report_path = os.path.join(save_dir, f'{model_name}_classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{model_name} 分類報告\n")
            f.write("=" * 50 + "\n")
            f.write(report)
        print(f"分類報告已保存至: {report_path}")
    
    # 建立比較 DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']]
    
    # 保存比較結果
    results_path = os.path.join(save_dir, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n模型比較結果已保存至: {results_path}")
    
    return results_df


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    繪製模型比較圖表
    
    Args:
        results_df: 結果 DataFrame
        save_path: 保存路徑（可選）
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    
    for i, metric in enumerate(metrics):
        axes[i].bar(results_df['model'], results_df[metric])
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('分數')
        axes[i].set_ylim([0, 1])
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"模型比較圖表已保存至: {save_path}")
    
    return plt.gcf()


if __name__ == '__main__':
    # 測試評估功能
    from src.models import train_all_models
    
    print("訓練模型...")
    models, X_test, y_test = train_all_models()
    
    print("\n評估模型...")
    results_df = compare_models(models, X_test, y_test)
    
    print("\n模型比較結果:")
    print(results_df)
    
    # 繪製比較圖表
    plot_model_comparison(results_df, save_path='results/model_comparison.png')
    plt.show()

