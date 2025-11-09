# OpenSpec 教學文件

## 📚 目錄

1. [什麼是 OpenSpec？](#什麼是-openspec)
2. [為什麼使用 OpenSpec？](#為什麼使用-openspec)
3. [OpenSpec 工作流程](#openspec-工作流程)
4. [建立專案步驟（以垃圾郵件分類專案為例）](#建立專案步驟以垃圾郵件分類專案為例)
5. [關鍵指令](#關鍵指令)
6. [專案結構說明](#專案結構說明)
7. [常見問題](#常見問題)

---

## 什麼是 OpenSpec？

OpenSpec 是一個**規範驅動開發（Spec-Driven Development）**框架，專門為 AI 編碼助手設計。它幫助開發者和 AI 在寫代碼之前就對功能規格達成一致。

### 核心概念

- **Specs（規格）**: 描述系統應該做什麼（`openspec/specs/`）
- **Changes（變更提案）**: 描述想要改變什麼（`openspec/changes/`）
- **Proposals（提案）**: 說明為什麼要改變、改變什麼、影響範圍
- **Tasks（任務）**: 具體的實施步驟清單

---

## 為什麼使用 OpenSpec？

### 優點

1. **明確的規格定義**: 在寫代碼前先定義清楚要做什麼
2. **可追蹤的變更**: 所有變更都有提案、任務和規格文件
3. **可審查的過程**: 團隊成員可以審查提案和規格
4. **AI 友好**: AI 助手可以根據規格文件生成代碼
5. **無需 API 金鑰**: 完全開源，不需要外部服務

### 適用場景

- ✅ 新功能開發
- ✅ 架構變更
- ✅ 重大功能迭代
- ✅ 需要文檔化的專案

---

## OpenSpec 工作流程

```
┌────────────────────┐
│ 1. 填充專案上下文   │  ← 定義專案目的、技術堆疊、規範
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 2. 建立變更提案      │  ← 描述要改變什麼、為什麼改變
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 3. 定義功能規格      │  ← 詳細描述功能需求和場景
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 4. 建立任務清單      │  ← 列出具體的實施步驟
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 5. 實施功能          │  ← 根據規格和任務實施代碼
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 6. 歸檔變更          │  ← 完成後歸檔，更新規格文件
└────────────────────┘
```

---

## 建立專案步驟（以垃圾郵件分類專案為例）

### 步驟 1: 初始化 OpenSpec 專案

**指令**:
```bash
openspec init
```

**作用**: 建立 OpenSpec 專案結構

**結果**: 建立以下目錄結構
```
openspec/
├── project.md              # 專案上下文（需要填充）
├── specs/                  # 當前規格（已建立的內容）
└── changes/                # 變更提案
    └── archive/            # 已歸檔的變更
```

**注意**: 如果無法執行 `openspec init`（如 PowerShell 執行政策限制），可以手動建立這些目錄。

---

### 步驟 2: 填充專案上下文（project.md）

**文件位置**: `openspec/project.md`

**作用**: 定義專案的基本資訊，讓 AI 助手了解專案背景

**需要填寫的內容**:

1. **Purpose（目的）**: 專案要做什麼
   ```markdown
   ## Purpose
   建立一個電子郵件（SMS）垃圾郵件分類系統，使用機器學習模型來識別和分類垃圾郵件。
   ```

2. **Tech Stack（技術堆疊）**: 使用哪些技術
   ```markdown
   ## Tech Stack
   - Python 3.8+
   - scikit-learn
   - pandas, numpy
   - NLTK
   - Streamlit
   ```

3. **Project Conventions（專案規範）**: 程式碼風格、架構模式等
   ```markdown
   ## Project Conventions
   ### Code Style
   - 遵循 PEP 8
   - 使用類型提示
   ```

4. **Domain Context（領域上下文）**: 專案相關的領域知識
   ```markdown
   ## Domain Context
   - 垃圾郵件分類是二分類問題（ham/spam）
   - 使用 TF-IDF 進行文本向量化
   ```

**提示給 AI**:
```
請閱讀 openspec/project.md 並幫助我填寫有關我的專案、技術堆疊和規範的詳細資訊。
```

---

### 步驟 3: 建立變更提案

**指令**:
```bash
# 使用 AI 助手（推薦）
"我想添加垃圾郵件分類系統。請為此功能建立一個 OpenSpec 變更提案。"

# 或使用 slash command（如果支援）
/openspec:proposal Add spam classification system
```

**手動建立結構**:
```bash
mkdir -p openspec/changes/add-spam-classification-system/specs/spam-classifier
```

**需要建立的文件**:

#### 3.1 提案文件（proposal.md）

**文件位置**: `openspec/changes/add-spam-classification-system/proposal.md`

**內容範例**:
```markdown
# Change: Add Spam Classification System

## Why
需要建立一個完整的垃圾郵件分類系統，包含資料預處理、模型訓練、評估和互動式 UI。

## What Changes
- 新增資料預處理模組
- 新增模型訓練模組
- 新增模型評估模組
- 新增 Streamlit UI 應用程式

## Impact
- Affected specs: 新增 `spam-classifier` capability
- Affected code: src/, app.py, requirements.txt
```

#### 3.2 任務清單（tasks.md）

**文件位置**: `openspec/changes/add-spam-classification-system/tasks.md`

**內容範例**:
```markdown
## 1. 專案設置
- [ ] 1.1 建立專案目錄結構
- [ ] 1.2 建立 requirements.txt
- [ ] 1.3 建立 .gitignore

## 2. 資料預處理
- [ ] 2.1 建立資料載入函數
- [ ] 2.2 實作文本清洗功能
- [ ] 2.3 實作分詞功能
...

## 3. 模型訓練
- [ ] 3.1 實作邏輯迴歸分類器
- [ ] 3.2 實作樸素貝葉斯分類器
...
```

#### 3.3 功能規格（spec.md）

**文件位置**: `openspec/changes/add-spam-classification-system/specs/spam-classifier/spec.md`

**內容範例**:
```markdown
# Delta for Spam Classifier

## ADDED Requirements

### Requirement: Data Preprocessing Pipeline
The system SHALL preprocess SMS text data for machine learning model training.

#### Scenario: Load and clean data
- **WHEN** the system loads the CSV dataset
- **THEN** it SHALL parse the label and text columns correctly
- **AND** it SHALL remove special characters, numbers, and punctuation

### Requirement: Model Training
The system SHALL train multiple classification models.

#### Scenario: Train logistic regression model
- **WHEN** preprocessed data is available
- **THEN** the system SHALL train a logistic regression classifier
- **AND** it SHALL save the trained model to disk
...
```

**規格格式說明**:

- **ADDED Requirements**: 新增的功能需求
- **MODIFIED Requirements**: 修改現有功能
- **REMOVED Requirements**: 移除的功能
- **Scenario**: 每個需求必須至少有一個場景（使用 `#### Scenario:` 格式）

---

### 步驟 4: 驗證變更提案

**指令**:
```bash
openspec validate add-spam-classification-system --strict
```

**作用**: 檢查變更提案的格式是否正確

**常見錯誤**:
- 缺少場景（Scenario）
- 場景格式錯誤（必須使用 `#### Scenario:`）
- 缺少需求標題（必須使用 `### Requirement:`）

---

### 步驟 5: 查看變更提案

**指令**:
```bash
openspec show add-spam-classification-system
```

**作用**: 顯示變更提案的詳細內容（提案、任務、規格）

---

### 步驟 6: 實施功能

**根據 tasks.md 中的任務清單，逐步實施功能**

**提示給 AI**:
```
請根據 openspec/changes/add-spam-classification-system/tasks.md 中的任務清單，
開始實施垃圾郵件分類系統的功能。
```

**實施順序**:
1. 完成任務 1（專案設置）
2. 完成任務 2（資料預處理）
3. 完成任務 3（模型訓練）
4. 完成任務 4（模型評估）
5. 完成任務 5（Streamlit UI）
6. 完成任務 6（文件）

**每完成一個任務，在 tasks.md 中標記為完成**:
```markdown
- [x] 1.1 建立專案目錄結構  ← 已完成
- [ ] 1.2 建立 requirements.txt  ← 待完成
```

---

### 步驟 7: 歸檔變更（完成後）

**指令**:
```bash
openspec archive add-spam-classification-system --yes
```

**作用**: 
- 將變更提案移動到 `openspec/changes/archive/` 目錄
- 更新 `openspec/specs/` 中的規格文件（將變更合併到主規格中）

**注意**: 只有在功能完全實施並測試通過後才歸檔。

---

## 關鍵指令

### 專案管理

```bash
# 初始化 OpenSpec 專案
openspec init

# 更新 AI 助手指令（當切換工具時）
openspec update
```

### 查看和列表

```bash
# 列出所有變更提案
openspec list

# 列出所有規格
openspec list --specs

# 查看變更提案詳情
openspec show <change-id>

# 查看規格詳情
openspec show <spec-id> --type spec
```

### 驗證

```bash
# 驗證變更提案（基本檢查）
openspec validate <change-id>

# 驗證變更提案（嚴格模式）
openspec validate <change-id> --strict

# 驗證所有變更和規格
openspec validate --strict
```

### 歸檔

```bash
# 歸檔變更提案（互動式）
openspec archive <change-id>

# 歸檔變更提案（非互動式）
openspec archive <change-id> --yes

# 歸檔但不更新規格（僅用於工具變更）
openspec archive <change-id> --skip-specs --yes
```

### 互動式儀表板

```bash
# 開啟互動式儀表板
openspec view
```

---

## 專案結構說明

### 完整專案結構

```
hw4/
├── openspec/                          # OpenSpec 工作流程目錄
│   ├── project.md                     # 專案上下文（目的、技術堆疊、規範）
│   ├── specs/                         # 當前規格（已建立的內容）
│   │   └── [capability]/             # 每個功能一個目錄
│   │       └── spec.md               # 功能規格文件
│   └── changes/                       # 變更提案
│       ├── [change-id]/              # 每個變更一個目錄
│       │   ├── proposal.md           # 變更提案（為什麼、改變什麼、影響）
│       │   ├── tasks.md              # 實施任務清單
│       │   ├── design.md             # 技術設計（可選）
│       │   └── specs/                # 變更的規格（delta）
│       │       └── [capability]/
│       │           └── spec.md      # 規格變更（ADDED/MODIFIED/REMOVED）
│       └── archive/                   # 已歸檔的變更
│           └── YYYY-MM-DD-[change-id]/
├── src/                               # 原始碼目錄
├── app.py                             # Streamlit 應用程式
├── requirements.txt                   # 依賴套件
└── README.md                          # 專案說明
```

### 文件說明

| 文件 | 位置 | 作用 |
|------|------|------|
| `project.md` | `openspec/project.md` | 定義專案上下文、技術堆疊、規範 |
| `proposal.md` | `openspec/changes/[id]/proposal.md` | 說明為什麼改變、改變什麼、影響範圍 |
| `tasks.md` | `openspec/changes/[id]/tasks.md` | 列出具體的實施步驟 |
| `spec.md` | `openspec/changes/[id]/specs/[capability]/spec.md` | 定義功能需求和場景 |
| `spec.md` | `openspec/specs/[capability]/spec.md` | 當前已建立的規格（主規格） |

---

## 常見問題

### Q1: 無法執行 `openspec init` 怎麼辦？

**A**: 可能是 PowerShell 執行政策限制。解決方法：

1. **手動建立目錄結構**:
   ```bash
   mkdir openspec\specs
   mkdir openspec\changes\archive
   ```

2. **建立 project.md**:
   ```bash
   # 手動建立 openspec/project.md 文件
   ```

3. **修改 PowerShell 執行政策**（需要管理員權限）:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Q2: 什麼時候需要建立變更提案？

**A**: 需要建立變更提案的情況：
- ✅ 新增功能
- ✅ 重大架構變更
- ✅ 破壞性變更（Breaking Changes）
- ✅ 性能優化（改變行為）

**不需要**建立變更提案的情況：
- ❌ Bug 修復（恢復預期行為）
- ❌ 拼寫錯誤、格式調整
- ❌ 非破壞性的依賴更新
- ❌ 配置變更

### Q3: 規格文件中的 Scenario 格式是什麼？

**A**: 必須使用以下格式：

```markdown
#### Scenario: 場景名稱
- **WHEN** 條件
- **THEN** 預期結果
- **AND** 額外條件或結果
```

**錯誤範例**:
```markdown
- **Scenario: 場景名稱**  ❌ 錯誤：使用列表格式
**Scenario**: 場景名稱     ❌ 錯誤：使用粗體
### Scenario: 場景名稱    ❌ 錯誤：使用三級標題
```

**正確範例**:
```markdown
#### Scenario: User login success
- **WHEN** valid credentials provided
- **THEN** return JWT token
```

### Q4: ADDED vs MODIFIED 的區別？

**A**: 
- **ADDED**: 新增一個獨立的功能需求
- **MODIFIED**: 修改現有功能（必須包含完整的更新後內容）

**範例**:

```markdown
## ADDED Requirements
### Requirement: Two-Factor Authentication
新增雙因素認證功能

## MODIFIED Requirements
### Requirement: User Authentication
修改現有的用戶認證功能（必須包含完整的新內容）
```

### Q5: 如何與 AI 助手合作使用 OpenSpec？

**A**: 使用以下提示：

1. **填充專案上下文**:
   ```
   請閱讀 openspec/project.md 並幫助我填寫有關我的專案、技術堆疊和規範的詳細資訊。
   ```

2. **建立變更提案**:
   ```
   我想添加[功能名稱]。請為此功能建立一個 OpenSpec 變更提案。
   ```

3. **實施功能**:
   ```
   請根據 openspec/changes/[change-id]/tasks.md 中的任務清單，開始實施功能。
   ```

4. **了解工作流程**:
   ```
   請解釋一下 OpenSpec 的工作流程（openspec/AGENTS.md），以及我應該如何與您合作完成這個專案。
   ```

---

## 實際案例：垃圾郵件分類專案

### 我們做了什麼？

1. **初始化 OpenSpec**: 建立 `openspec/` 目錄結構
2. **填充專案上下文**: 填寫 `openspec/project.md`（目的、技術堆疊、規範）
3. **建立變更提案**: 
   - `proposal.md`: 說明為什麼要建立垃圾郵件分類系統
   - `tasks.md`: 列出實施步驟（6 個主要任務）
   - `spec.md`: 定義功能規格（資料預處理、模型訓練、評估、UI）
4. **實施功能**: 根據 tasks.md 逐步實施
5. **完成**: 所有功能已實施，專案可以運行

### 變更提案結構

```
openspec/changes/add-spam-classification-system/
├── proposal.md                    # 為什麼、改變什麼、影響
├── tasks.md                       # 實施任務清單
└── specs/
    └── spam-classifier/
        └── spec.md               # 功能規格（ADDED Requirements）
```

### 實施的檔案

根據 tasks.md 實施的檔案：
- ✅ `src/data_preprocessing.py` - 資料預處理
- ✅ `src/models.py` - 模型訓練
- ✅ `src/evaluation.py` - 模型評估
- ✅ `app.py` - Streamlit UI
- ✅ `requirements.txt` - 依賴套件
- ✅ `README.md` - 專案文件

---

## 總結

OpenSpec 幫助我們：

1. **在寫代碼前先定義清楚要做什麼**（規格文件）
2. **追蹤所有變更**（變更提案）
3. **有條理地實施功能**（任務清單）
4. **與 AI 助手更好地合作**（明確的規格）

使用 OpenSpec 的關鍵是：
- 📝 **先寫規格，再寫代碼**
- 📋 **使用任務清單追蹤進度**
- 🔍 **定期驗證變更提案**
- 📦 **完成後歸檔變更**

---

## 參考資源

- [OpenSpec GitHub](https://github.com/Fission-AI/OpenSpec)
- [OpenSpec 官方文件](https://github.com/Fission-AI/OpenSpec#readme)
- [AGENTS.md 規範](https://agents.md/)

---

**提示**: 如果遇到問題，可以查看 `openspec/AGENTS.md` 文件，裡面有詳細的 AI 助手指令。

