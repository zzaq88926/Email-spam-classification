# 📧 垃圾郵件分類系統

## 🌐 線上演示

**👉 [點擊這裡查看線上演示](https://email-spam-classification-5114056048.streamlit.app/)**

---

## 📋 專案簡介

這是一個完整的 SMS 垃圾郵件分類系統，使用機器學習模型（邏輯迴歸、樸素貝葉斯、支援向量機）進行文本分類。系統包含完整的資料預處理管線、模型訓練、評估指標和互動式 Streamlit UI，支援參數調整和即時預測功能。

## ✨ 主要功能

### 🎯 核心功能
- **多種分類模型**: 支援邏輯迴歸、樸素貝葉斯、支援向量機三種分類器
- **完整的資料預處理**: 文本清洗、分詞、停用詞移除、TF-IDF/Count 向量化
- **詳細的模型評估**: 準確率、精確率、召回率、F1-score、混淆矩陣、ROC 曲線、AUC 分數
- **互動式 UI**: Streamlit 介面，支援資料集概覽、模型訓練、評估和即時預測
- **模型比較**: 自動比較多種模型的性能並生成視覺化圖表
- **參數調整**: 可調整向量化器參數、模型超參數、訓練/測試集比例

### 🛠️ 進階功能
- **可調整參數**:
  - 資料分割參數（測試集比例、隨機種子）
  - 向量化器參數（類型、最大特徵數、min_df、max_df、N-gram 範圍）
  - 模型超參數（邏輯迴歸、樸素貝葉斯、支援向量機）
- **即時預測**:
  - 自定義 Spam 判定閾值
  - 機率分布視覺化
  - 範例訊息快速測試
  - 預處理文本查看

## 📊 系統架構

```
垃圾郵件分類系統
│
├── 資料預處理
│   ├── 文本清洗（移除特殊字符、數字、標點符號）
│   ├── 分詞（NLTK）
│   ├── 停用詞移除
│   └── 文本向量化（TF-IDF / Count Vectorizer）
│
├── 模型訓練
│   ├── 邏輯迴歸（Logistic Regression）
│   ├── 樸素貝葉斯（Naive Bayes）
│   └── 支援向量機（Support Vector Machine）
│
├── 模型評估
│   ├── 評估指標（準確率、精確率、召回率、F1-score）
│   ├── 混淆矩陣
│   ├── ROC 曲線和 AUC 分數
│   └── 分類報告
│
└── 互動式 UI
    ├── 資料集概覽
    ├── 模型訓練（參數調整）
    ├── 模型評估
    ├── 即時預測
    └── 模型比較
```

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

首次執行時，系統會自動下載所需的 NLTK 資料（stopwords 和 punkt_tab tokenizer）。如果下載失敗，可以手動執行：

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
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

### 📊 資料集概覽

- 查看資料集基本統計（總訊息數、Ham/Spam 分布、Spam 比例）
- 查看標籤分布圓餅圖
- 瀏覽樣本訊息（可調整顯示數量）

### 🚀 模型訓練

1. **調整參數**（可選）:
   - 展開「⚙️ 調整參數」區域
   - 調整資料分割參數（測試集比例、隨機種子）
   - 調整向量化器參數（類型、特徵數、N-gram 範圍）
   - 調整模型超參數（邏輯迴歸、樸素貝葉斯、支援向量機）

2. **訓練模型**:
   - 點擊「開始訓練模型」按鈕
   - 系統會自動訓練三種模型
   - 訓練完成後，模型會自動保存到 `models/` 目錄

3. **查看訓練結果**:
   - 查看訓練結果摘要
   - 查看使用的參數配置

### 📈 模型評估

1. 選擇要評估的模型
2. 點擊「評估模型」按鈕
3. 查看評估指標、混淆矩陣、ROC 曲線和分類報告

### 🔮 即時預測

1. **選擇模型**: 從下拉選單中選擇要使用的模型
2. **調整預測選項**（可選）:
   - 展開「⚙️ 預測選項」區域
   - 設定 Spam 判定閾值
   - 選擇顯示選項（預處理文本、機率條形圖）
3. **輸入訊息**: 
   - 手動輸入訊息，或
   - 點擊範例訊息按鈕快速測試
4. **查看結果**: 查看預測結果、機率和視覺化圖表

### ⚖️ 模型比較

1. 點擊「比較所有模型」按鈕
2. 查看所有模型的性能比較表格
3. 查看模型性能比較圖表
4. 系統會自動識別最佳模型（F1 分數）

## 🔧 可調整參數

### 資料分割參數
- **測試集比例**: 0.1 - 0.5（預設：0.2）
- **隨機種子**: 0 - 1000（預設：42）

### 向量化器參數
- **向量化器類型**: TF-IDF 或 Count Vectorizer
- **最大特徵數**: 1000 - 10000（預設：5000）
- **最小文檔頻率 (min_df)**: 1 - 10（預設：2）
- **最大文檔頻率 (max_df)**: 0.5 - 1.0（預設：0.95）
- **N-gram 範圍**: 最小 1-3，最大 1-3（預設：1-2）

### 模型超參數

#### 邏輯迴歸
- **正則化強度 (C)**: 0.01 - 100.0（預設：1.0）
- **正則化類型**: l2, l1, elasticnet（預設：l2）
- **求解器**: lbfgs, liblinear, sag, saga（預設：lbfgs）
- **最大迭代次數**: 100 - 5000（預設：1000）

#### 樸素貝葉斯
- **平滑參數 (alpha)**: 0.1 - 10.0（預設：1.0）

#### 支援向量機
- **正則化參數 (C)**: 0.01 - 100.0（預設：1.0）
- **核函數**: linear, rbf, poly, sigmoid（預設：linear）
- **Gamma**: scale, auto（當使用 rbf 核時）

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

## 💻 程式碼使用範例

### 資料預處理

```python
from src.data_preprocessing import load_data, preprocess_data, create_vectorizer, prepare_features

# 載入資料
df = load_data('datasets/sms_spam_no_header.csv')

# 預處理資料
df_processed = preprocess_data(df)

# 建立向量化器
vectorizer = create_vectorizer('tfidf', max_features=5000, ngram_range=(1, 2))

# 準備特徵
X = prepare_features(df_processed, vectorizer, fit=True)
```

### 模型訓練

```python
from src.models import SpamClassifier

# 建立分類器（可自定義參數）
classifier = SpamClassifier(
    model_type='logistic_regression',
    vectorizer_type='tfidf',
    vectorizer_params={'max_features': 5000, 'ngram_range': (1, 2)},
    model_params={'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
)

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

## 📝 OpenSpec 工作流程

本專案使用 OpenSpec 進行規範驅動開發：

1. **專案上下文**: `openspec/project.md` 包含專案目的、技術堆疊和規範
2. **變更提案**: `openspec/changes/add-spam-classification-system/` 包含完整的變更提案
3. **規格文件**: `openspec/changes/add-spam-classification-system/specs/spam-classifier/spec.md` 包含功能規格

詳細的 OpenSpec 教學文件請參考：`OpenSpec teaching.md`

## 🛠️ 技術堆疊

- **程式語言**: Python 3.8+
- **機器學習框架**: scikit-learn
- **資料處理**: pandas, numpy
- **自然語言處理**: nltk
- **視覺化**: matplotlib, seaborn
- **Web UI**: Streamlit
- **開發工作流程**: OpenSpec

## 📄 授權

MIT License

## 🙏 致謝

- 資料集來源：UCI Machine Learning Repository
- 使用技術：scikit-learn, Streamlit, NLTK
- 開發框架：OpenSpec

## 📧 聯絡方式

如有問題或建議，請提交 Issue 或 Pull Request。

---

**🌐 [點擊這裡查看線上演示](https://email-spam-classification-5114056048.streamlit.app/)**

**📝 注意**: 首次執行時，系統會自動下載 NLTK 資料。如果遇到下載問題，請檢查網路連接或手動下載。
