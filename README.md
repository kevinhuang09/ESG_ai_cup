# VPESG 多任務分類系統 — 雙主力雙專家動態信心救場集成

> ESG 文本承諾與證據分析（Verifiable Promises in ESG, VPESG）
> 任務：四子任務多標籤分類（承諾狀態 / 驗證時程 / 證據狀態 / 證據品質）

本系統針對每筆 ESG 文本，同時預測四項子任務，採用「**雙主力 + 雙專家動態信心救場**」的多模型集成架構。本 README 旨在使第三方使用者能完整重現本隊之**環境配置、資料前處理、模型訓練與最終集成推論**結果。

---

## 一、系統總覽

```
原始資料 (train / val / test)
        │
        ▼
[前處理] convert_year.py
         └─ inject_features()：時間特徵 + 概念特徵注入（子任務四 Timeline 用）
        │
        ▼
[訓練] 四組模型分別訓練（10-Fold OOF；主力多設定平均）
   ├─ rbtl   hfl/chinese-roberta-wwm-ext-large          → 主力①（4 seeds）
   ├─ bge    BAAI/bge-large-zh-v1.5                       → 主力②（4 configs）
   ├─ els    IDEA-CCNL/Erlangshen-RoBERTa-330M-Similarity → 專家①
   └─ tower  BGE-M3 + Multilingual-E5-large + Text2Vec    → 專家②（三塔階層融合）
        │  （各自輸出 OOF 機率 + test 機率）
        ▼
[集成] final_ense.py ── 動態信心救場決策
   ① 雙主力加權融合：BGE×0.55 + RoBERTa×0.45
   ② 主力信心 < 0.6 → 觸發專家救場
   ③ 專家信心 > 0.8 才接手；els / tower 擇優（取信心高者）
   ④ 階層規則約束（promise=No → 後三欄 N/A；evidence=No → quality N/A）
        │
        ▼
submission.csv（id 12001–14000，共 2000 筆）
```

---

## 二、環境配置

### 2.1 硬體需求

| 項目 | 建議配置 |
|------|---------|
| GPU | NVIDIA GPU，顯存 ≥ 16GB（large 模型微調建議 24GB） |
| CUDA | 11.8 或以上 |
| 記憶體 | ≥ 32GB RAM |
| 磁碟 | ≥ 30GB（含預訓練模型快取與多 fold checkpoint） |

> 三塔（tower）專家模型之分類器階段（LogisticRegression / LinearSVC）可在 CPU 執行，僅句向量嵌入提取需 GPU。

### 2.2 軟體版本

| 套件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.0.1（CUDA 11.8） |
| transformers | 4.35.x |
| sentence-transformers | 2.2.2 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8（結果視覺化） |
| Pillow | 12.1.1（影像處理） |
| opencc-python-reimplemented | 1.1.x（選用，繁簡轉換，預設關閉） |

### 2.3 安裝步驟

```bash
# 1. 建立虛擬環境
conda create -n vpesg python=3.10 -y
conda activate vpesg

# 2. 安裝 PyTorch（依 CUDA 版本調整）
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝其餘相依套件
pip install -r requirements.txt
```

**requirements.txt 內容：**

```
transformers==4.35.2
sentence-transformers==2.2.2
scikit-learn==1.8.0
numpy==2.4.4
pandas==2.3.3
scipy==1.17.1
matplotlib==3.10.8
Pillow==12.1.1
tqdm
opencc-python-reimplemented==1.1.6   # 選用，預設關閉
```

### 2.4 預訓練模型下載

本系統使用以下 6 個 Hugging Face 預訓練模型，首次執行會自動下載至本機快取（`~/.cache/huggingface`），亦可預先下載：

| 用途 | Hugging Face 路徑 |
|------|------------------|
| 主力① rbtl | `hfl/chinese-roberta-wwm-ext-large` |
| 主力② bge | `BAAI/bge-large-zh-v1.5` |
| 專家① els | `IDEA-CCNL/Erlangshen-RoBERTa-330M-Similarity` |
| 三塔① tower | `BAAI/bge-m3` |
| 三塔② tower | `intfloat/multilingual-e5-large` |
| 三塔③ tower | `shibing624/text2vec-base-chinese` |

```bash
# 範例：預先下載
huggingface-cli download hfl/chinese-roberta-wwm-ext-large
huggingface-cli download BAAI/bge-large-zh-v1.5
# ...其餘四個同理

# 中國大陸 / 連線受限環境可改用鏡像：
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 三、目錄結構

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── vpesg_4k_train_1000.csv     # 訓練集
│   ├── vpesg4k_val_1000.csv        # 驗證集
│   └── vpesg4k_test_2000.csv       # 測試集（id 12001-14000）
├── preprocess/
│   ├── convert_year.py             # 時間特徵前處理進入點
│   └── year_parse.py               # inject_features() 特徵注入實作
├── train/
│   ├── train_rbtl.py               # 主力① RoBERTa（4 seeds）
│   ├── train_bge.py                # 主力② BGE（4 configs）
│   ├── train_els.py                # 專家① Erlangshen
│   └── train_tower.py              # 專家② 三塔嵌入 + 分類器（含軟偽標籤）
├── out_cv_rbtL_tf/  out_cv_rbtL_tf_s1/  _s2/  _s3/   # rbtl 4 seeds 輸出
├── out_bgeL_ep8_lr1e5/  ..._ep10_lr1e5/  ..._ep8_lr2e5/  ..._ep10_lr2e5/  # bge 4 configs
├── out_cv_els_tf/                  # els 輸出
├── out_cv10_soft_pseudo_labeling/  # tower 輸出
├── final_ense.py                   # 最終動態信心救場集成
└── submission.csv                  # 最終提交檔
```

---

## 四、資料格式

### 4.1 輸入欄位

| 欄位 | 說明 |
|------|------|
| `id` | 樣本編號（test 為 12001–14000） |
| `data` | ESG 承諾／證據文本（經特徵注入後使用） |
| `pdf_url` | 來源報告書連結（特徵注入時參考） |
| `promise_status` | Yes / No |
| `verification_timeline` | already / within_2_years / between_2_and_5_years / more_than_5_years / N/A |
| `evidence_status` | Yes / No / N/A |
| `evidence_quality` | Clear / Not Clear / Misleading / N/A |

### 4.2 子任務四（Timeline）標註定義

驗證時程以企業永續報告書公開之**西元 2024 年**起算：

| 類別 | 定義 |
|------|------|
| `already` | 承諾已實行，可在當期驗證 |
| `within_2_years` | 短期規劃，2 年內可驗證 |
| `between_2_and_5_years` | 中長期規劃，2–5 年內可驗證；**或承諾語句未明示完成年份時選用** |
| `more_than_5_years` | 長期規劃，5 年以上可驗證 |
| `N/A` | promise_status 為 No 時選用 |

### 4.3 輸出（submission.csv）

| 欄位 | 內容 |
|------|------|
| `id` | 12001–14000 |
| `promise_status` | Yes / No |
| `verification_timeline` | 四類之一，或 N/A |
| `evidence_status` | Yes / No / N/A |
| `evidence_quality` | 三類之一，或 N/A |

---

## 五、執行流程（重現結果）

### Step 1 — 資料前處理（時間特徵注入）

```bash
python preprocess/convert_year.py \
    --train data/vpesg_4k_train_1000.csv \
    --val   data/vpesg4k_val_1000.csv \
    --test  data/vpesg4k_test_2000.csv
```

**模組說明：** `convert_year.py` 呼叫 `year_parse.inject_features(data, pdf_url, timefeat=True, concfeat=True)`，以公開年 2024 為基準，將文本中承諾完成年份換算為相對時程，並注入時間與概念特徵至 `data` 欄，供 `verification_timeline` 任務使用。
**輸入：** 含 `data`、`pdf_url` 欄之 CSV。**輸出：** `data` 欄已增補特徵之 CSV。

### Step 2 — 訓練四組模型

每組皆採 **10 折分層交叉驗證（StratifiedKFold, NFOLD=10）**，產出折外（OOF）機率與測試集機率。

```bash
# 主力① RoBERTa：4 個隨機種子（cv_rbtL_tf 系列）
python train/train_rbtl.py --seeds 0 1 2 3
#   → out_cv_rbtL_tf / out_cv_rbtL_tf_s1 / _s2 / _s3

# 主力② BGE：4 組 epoch/lr 設定
python train/train_bge.py --configs "ep8_lr1e5 ep10_lr1e5 ep8_lr2e5 ep10_lr2e5"
#   → out_bgeL_ep8_lr1e5 / ... / out_bgeL_ep10_lr2e5

# 專家① Erlangshen
python train/train_els.py            # → out_cv_els_tf

# 專家② 三塔嵌入 + 分類器（含軟偽標籤自我蒸餾）
ESG_SEED=42 ESG_TAG=cv10_soft_pseudo_labeling ESG_OPENCC=0 \
python train/train_tower.py          # → out_cv10_soft_pseudo_labeling
```

各輸出目錄包含：
- `oof_prob.npy`：折外預測機率（供集成權重校調與本地評估）
- `test_prob.npy`：測試集預測機率（供最終集成）

### Step 3 — 最終集成推論

```bash
python final_ense.py     # 讀取四模型輸出目錄，產出 submission.csv
```

**輸出：** `submission.csv`（2000 筆最終預測）。

---

## 六、關鍵參數設定

### 6.1 任務評分權重

| 子任務 | 權重 |
|--------|------|
| promise_status | 0.20 |
| verification_timeline | 0.15 |
| evidence_status | 0.30 |
| evidence_quality | 0.35 |

### 6.2 訓練超參數

| 模型 | 設定 |
|------|------|
| rbtl | 4 seeds（cv_rbtL_tf / _s1 / _s2 / _s3），結果平均；多任務交叉熵損失加總；類別權重加權 |
| bge | 4 configs：epoch ∈ {8, 10} × lr ∈ {1e-5, 2e-5}，4 模型平均 |
| els | 預訓練編碼 + 多任務分類頭，端對端微調 |
| tower | 凍結句向量嵌入 + LogisticRegression / CalibratedClassifierCV(LinearSVC)，`class_weight='balanced'`，GridSearchCV 3-fold 搜尋 C |

### 6.3 三塔（tower）關鍵設定（train_tower.py）

| 參數 | 值 |
|------|----|
| `MODEL_NAME_1` | `BAAI/bge-m3` |
| `MODEL_NAME_2` | `intfloat/multilingual-e5-large`（輸入加註 `"passage:"` 前綴） |
| `MODEL_NAME_3` | `shibing624/text2vec-base-chinese` |
| `NFOLD` | 10 |
| `SOFT_PSEUDO_THRESH` | 0.85 |
| `USE_TIMEFEAT` / `USE_CONCFEAT` | True / True |
| `USE_OPENCC`（`ESG_OPENCC`） | 0（**預設關閉**） |
| 塔內 / 塔間融合權重 | 以驗證集 Macro-F1 網格搜尋；per-class 權重以 Nelder-Mead 微調 |

### 6.4 集成決策參數（final_ense.py）

| 參數 | 值 | 說明 |
|------|----|----|
| `W_BGE` | **0.55** | 雙主力中 BGE 權重 |
| `W_ROBERTA` | **0.45** | 雙主力中 RoBERTa 權重 |
| `MAIN_CONF_THRES` | **0.6** | 主力信心低於此值 → 觸發專家救場 |
| `EXP_CONF_THRES` | **0.8** | 專家信心須高於此值方可接手 |
| 專家擇優 | els / tower 皆達標時取信心較高者 | — |

### 6.5 階層規則約束

```
若 promise_status == No   →  verification_timeline / evidence_status / evidence_quality = N/A
若 evidence_status == No   →  evidence_quality = N/A
```

---

## 七、重要模組輸入／輸出對照

| 模組 | 輸入 | 輸出 | 功能 |
|------|------|------|------|
| `year_parse.inject_features` | `data`, `pdf_url` | 增補特徵之 `data` | 時間／概念特徵注入 |
| `convert_year.py` | 原始 CSV | 前處理 CSV | 前處理進入點 |
| `train.py` + `rbtl_ense.py` | 前處理 CSV | OOF / test 機率（.npy） | 主力① 微調（4 seeds） |
| `train.py` + `bge_ense.py` | 前處理 CSV | OOF / test 機率（.npy） | 主力② 微調（4 configs） |
| `train.py` + `els_ense.py`| 前處理 CSV | OOF / test 機率（.npy） | 專家① 微調 |
| `run_all_tower.py` | 前處理 CSV | OOF / test 機率（.npy） | 專家② 三塔嵌入 + 分類器（含軟偽標籤） |
| `final_ense.py` | 四模型 test 機率（.npy） | `submission.csv` | 動態信心救場集成 + 階層規則 |

---

## 八、本地驗證結果（OOF 分數）

| 模型 | 角色 | OOF 分數 |
|------|------|---------|
| BGE-large-zh-v1.5 | 主力 | 0.6161 |
| 中文 RoBERTa-wwm-ext-large | 主力 | 0.6132 |
| Erlangshen-RoBERTa-330M | 專家 | 0.5887 |
| 三塔嵌入結構（tower） | 專家 | 0.5773 |

集成後表現優於任一單一模型。詳細救場統計與分析請參閱報告「陸、分析與結論」。

---

## 九、第三方除錯指引（FAQ）

| 問題 | 排查方向 |
|------|---------|
| CUDA out of memory | 調降 batch size，或將 max_length 由 512 降至 256；large 模型建議 ≥ 24GB 顯存 |
| 模型下載失敗 | `export HF_ENDPOINT=https://hf-mirror.com` 使用鏡像，或預先離線下載 |
| OOF 分數與報告不符 | 確認 StratifiedKFold 之 `random_state` 與 `NFOLD=10` 一致；rbtl 4 seeds、bge 4 configs 須完整跑齊 |
| E5 模型效果異常 | 確認輸入文本已加註 `"passage:"` 前綴（E5 規範要求） |
| Timeline 任務分數偏低 | 確認前處理階段 `USE_TIMEFEAT=True`、`year_parse.inject_features` 有正確套用 |
| `submission.csv` 出現邏輯矛盾 | 確認 `final_ense.py` 階層規則約束有正確套用（promise=No / evidence=No） |
| 三塔分類器訓練過慢 | 縮小 GridSearchCV 之 C 搜尋空間，或將 `n_jobs=-1` 平行化 |

### 軟性偽標籤（soft pseudo-labeling）說明
`train_tower.py` 訓練時，以歷史最佳模型（`out_cv10_tripletower_features_on`）對測試集之預測，篩選四項子任務信心**皆 ≥ 0.85**（`SOFT_PSEUDO_THRESH`）之高可信樣本，依平均信心加權後併入訓練集再訓一輪。此偽標籤**僅來自競賽測試集本身**，未引入任何競賽外部資料。

### OpenCC（繁簡轉換）
程式碼保留 OpenCC 介面，但預設 `ESG_OPENCC=0`（**關閉**）。最終提交版本未啟用繁簡轉換。

---

## 十、授權與致謝

本系統使用之預訓練模型與開源套件，版權均歸原作者所有，詳見報告「捌、使用的外部資源與參考文獻」。本隊僅於競賽範圍內使用，**除大會提供之 VPESG 競賽資料集外，未引入任何競賽外部標註資料**。