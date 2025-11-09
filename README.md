# 📧 垃圾郵件分類系統

一個使用機器學習模型進行 SMS 垃圾郵件分類的完整系統，包含資料預處理、模型訓練、評估和互動式 Streamlit UI。

## ✨ 功能特色

- **多種分類模型**: 支援邏輯迴歸、樸素貝葉斯、支援向量機三種分類器
- **完整的資料預處理**: 文本清洗、分詞、停用詞移除、向量化
- **詳細的模型評估**: 準確率、精確率、召回率、F1-score、混淆矩陣、ROC 曲線
- **互動式 UI**: Streamlit 介面，支援資料集概覽、模型訓練、評估和即時預測
- **模型比較**: 自動比較多種模型的性能並生成視覺化圖表

## 📋 專案結構

```
hw4/
├── app.py                          # Streamlit 主應用程式
├── requirements.txt                # Python 依賴套件
├── README.md                       # 專案說明文件
├── .gitignore                      # Git 忽略文件
├── datasets/                       # 資料集目錄
│   └── sms_spam_no_header.csv     # SMS 垃圾郵件數據集
├── src/                            # 原始碼目錄
│   ├── __init__.py
│   ├── data_preprocessing.py       # 資料預處理模組
│   ├── models.py                   # 模型訓練模組
│   └── evaluation.py              # 模型評估模組
├── models/                         # 訓練好的模型保存目錄
├── results/                        # 評估結果保存目錄
├── notebooks/                      # Jupyter Notebooks（可選）
└── openspec/                       # OpenSpec 工作流程文件
    ├── project.md                  # 專案上下文
    ├── specs/                      # 規格文件
    └── changes/                    # 變更提案
```

## 🚀 快速開始

### 1. 環境要求

- Python 3.8+
- pip 或 conda

### 2. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3. 下載 NLTK 資料

首次執行時，系統會自動下載所需的 NLTK 資料（stopwords 和 punkt tokenizer）。如果下載失敗，可以手動執行：

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 4. 準備資料集

確保資料集位於 `datasets/sms_spam_no_header.csv`，格式為：
- 第一列：標籤（"ham" 或 "spam"）
- 第二列：文本內容

### 5. 執行應用程式

```bash
streamlit run app.py
```

應用程式將在瀏覽器中自動開啟（通常是 `http://localhost:8501`）。

## 📖 使用說明

### 資料集概覽

- 查看資料集基本統計（總訊息數、Ham/Spam 分布）
- 查看標籤分布圖
- 瀏覽樣本訊息

### 模型訓練

1. 點擊「開始訓練模型」按鈕
2. 系統會自動訓練三種模型：
   - 邏輯迴歸（Logistic Regression）
   - 樸素貝葉斯（Naive Bayes）
   - 支援向量機（Support Vector Machine）
3. 訓練完成後，模型會自動保存到 `models/` 目錄

### 模型評估

1. 選擇要評估的模型
2. 點擊「評估模型」按鈕
3. 查看評估指標、混淆矩陣、ROC 曲線和分類報告

### 即時預測

1. 選擇要使用的模型
2. 輸入要分類的訊息
3. 點擊「預測」按鈕
4. 查看預測結果和機率

### 模型比較

1. 點擊「比較所有模型」按鈕
2. 查看所有模型的性能比較表格
3. 查看模型性能比較圖表
4. 系統會自動識別最佳模型

## 🔧 程式碼使用

### 資料預處理

```python
from src.data_preprocessing import load_data, preprocess_data, create_vectorizer, prepare_features

# 載入資料
df = load_data('datasets/sms_spam_no_header.csv')

# 預處理資料
df_processed = preprocess_data(df)

# 建立向量化器
vectorizer = create_vectorizer('tfidf')

# 準備特徵
X = prepare_features(df_processed, vectorizer, fit=True)
```

### 模型訓練

```python
from src.models import SpamClassifier

# 建立分類器
classifier = SpamClassifier(model_type='logistic_regression', vectorizer_type='tfidf')

# 訓練模型
classifier.train(X_train, y_train)

# 保存模型
classifier.save('models/')
```

### 模型評估

```python
from src.evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve

# 計算指標
metrics = calculate_metrics(y_test, y_pred)

# 繪製混淆矩陣
plot_confusion_matrix(y_test, y_pred, 'Logistic Regression')

# 繪製 ROC 曲線
plot_roc_curve(y_test, y_proba, 'Logistic Regression')
```

## 📊 評估指標

系統會計算以下評估指標：

- **準確率（Accuracy）**: 正確預測的比例
- **精確率（Precision）**: 預測為 spam 的訊息中，實際為 spam 的比例
- **召回率（Recall）**: 實際為 spam 的訊息中，被正確識別的比例
- **F1 分數（F1-Score）**: 精確率和召回率的調和平均數
- **AUC 分數**: ROC 曲線下的面積

## 🎯 模型性能

系統會自動比較三種模型的性能，並生成：

- 模型比較表格（CSV 格式）
- 模型性能比較圖表
- 各模型的混淆矩陣
- 各模型的 ROC 曲線
- 分類報告

## 📝 OpenSpec 工作流程

本專案使用 OpenSpec 進行規範驅動開發：

1. **專案上下文**: `openspec/project.md` 包含專案目的、技術堆疊和規範
2. **變更提案**: `openspec/changes/add-spam-classification-system/` 包含完整的變更提案
3. **規格文件**: `openspec/changes/add-spam-classification-system/specs/spam-classifier/spec.md` 包含功能規格

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License

## 🙏 致謝

- 資料集來源：UCI Machine Learning Repository
- 使用技術：scikit-learn, Streamlit, NLTK

## 📧 聯絡方式

如有問題或建議，請提交 Issue。

---

**注意**: 首次執行時，系統會自動下載 NLTK 資料。如果遇到下載問題，請檢查網路連接或手動下載。
