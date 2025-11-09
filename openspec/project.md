# Project Context

## Purpose
建立一個電子郵件（SMS）垃圾郵件分類系統，使用機器學習模型（邏輯迴歸、樸素貝葉斯、支援向量機）來識別和分類垃圾郵件。系統包含完整的資料預處理管線、模型訓練、評估指標和 Streamlit 互動式 UI。

## Tech Stack
- **程式語言**: Python 3.8+
- **機器學習框架**: scikit-learn
- **資料處理**: pandas, numpy
- **自然語言處理**: nltk, re
- **視覺化**: matplotlib, seaborn
- **Web UI**: Streamlit
- **開發工具**: jupyter notebook (可選)

## Project Conventions

### Code Style
- 遵循 PEP 8 Python 程式碼風格規範
- 使用有意義的變數和函數名稱（英文）
- 函數和類別需要 docstring 說明
- 使用類型提示（type hints）提高程式碼可讀性
- 程式碼註解使用中文，方便理解

### Architecture Patterns
- **模組化設計**: 將資料預處理、模型訓練、評估等功能分離為獨立模組
- **管道模式**: 使用 scikit-learn Pipeline 組織資料處理和模型訓練流程
- **配置驅動**: 將模型參數和路徑配置集中管理

### Testing Strategy
- 使用 train_test_split 進行資料分割（80/20 或 70/30）
- 使用交叉驗證評估模型穩定性
- 計算多種評估指標：準確率、精確率、召回率、F1-score、混淆矩陣

### Git Workflow
- 主分支：main
- 功能分支：feature/功能名稱
- 提交訊息使用中文，格式：`類型: 簡短描述`
  - 類型：新增、修改、修復、重構

## Domain Context
- **垃圾郵件分類**: 二分類問題（ham/spam）
- **資料集**: SMS 垃圾郵件數據集（CSV 格式，無標題行）
  - 第一列：標籤（"ham" 或 "spam"）
  - 第二列：文本內容
- **文本預處理**: 需要處理特殊字符、縮寫、數字等
- **特徵工程**: 使用 TF-IDF 或 Count Vectorizer 進行文本向量化
- **模型選擇**: 比較多種分類器性能，選擇最佳模型

## Important Constraints
- 資料集位於 `./datasets/sms_spam_no_header.csv`
- 需要處理不平衡數據集（ham 和 spam 的數量可能不相等）
- 模型需要可解釋性（至少提供特徵重要性）
- UI 需要支援即時預測功能

## External Dependencies
- **數據集**: `datasets/sms_spam_no_header.csv`（本地文件）
- **NLTK 資料**: 需要下載 stopwords 和 punkt tokenizer
  ```python
  nltk.download('stopwords')
  nltk.download('punkt')
  ```

