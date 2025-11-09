# Change: Add Spam Classification System

## Why
需要建立一個完整的垃圾郵件分類系統，包含資料預處理、模型訓練、評估和互動式 UI。這個系統將使用機器學習模型來識別和分類垃圾郵件，並提供 Streamlit 介面供使用者互動。

## What Changes
- 新增資料預處理模組（清洗、分詞、向量化）
- 新增模型訓練模組（邏輯迴歸、樸素貝葉斯、支援向量機）
- 新增評估模組（指標計算和視覺化）
- 新增 Streamlit UI 應用程式
- 新增 README 文件

## Impact
- Affected specs: 新增 `spam-classifier` capability
- Affected code: 
  - `src/data_preprocessing.py` - 資料預處理
  - `src/models.py` - 模型訓練
  - `src/evaluation.py` - 模型評估
  - `app.py` - Streamlit UI
  - `requirements.txt` - 依賴套件
  - `README.md` - 專案文件

