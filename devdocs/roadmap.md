## 0. Project Structure

```python
hackathon-20205-evan-ston-energy/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example              # 放 LLM / API key 範例
│
├── data/
│   ├── players.csv
│   ├── player_valuations.csv
│   ├── appearances.csv
│   ├── games.csv
│   ├── clubs.csv
│   ├── competitions.csv
│   ├── transfers.csv
│   ├── game_events.csv
│   ├── game_lineups.csv
│   └── club_games.csv
│   │
│   ├── interim/              # 中間輸出（暫存 / debug 用）
│   │   └── ...
│   │
│   └── processed/            # 給模型 & 前端用的整理後資料
│       ├── player_snapshot.parquet          # 共用 feature table
│       ├── regression_outputs.parquet       # M1 輸出
│       ├── classification_outputs.parquet   # M2 輸出
│       ├── player_recommendations.parquet   # 給前端主用
│       └── mock_player_recommendations.csv  # 你現在可先用 mock
│
├── notebooks/
│   ├── 01_eda_overview.ipynb          # EDA & 初步圖表
│   ├── 02_feature_exploration.ipynb   # feature 分析
│   ├── 03_regression_prototype.ipynb  # 回歸模型 prototype
│   ├── 04_classification_prototype.ipynb
│   └── 05_dashboard_mockups.ipynb     # 畫圖/想 UI
│
├── config/
│   ├── paths.yaml          # 各種路徑設定（raw/processed/app 等）
│   ├── features.yaml       # 哪些 feature 要用、要 drop
│   └── model_params.yaml   # LightGBM hyperparams
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/               # 讀 CSV、清理、join、建 snapshot
│   │   ├── __init__.py
│   │   ├── load_raw.py                 # 讀 data/raw 裡的 CSV
│   │   ├── preprocess_utils.py         # 共用清理函數
│   │   └── build_player_snapshot.py    # 產生 player_snapshot
│   │
│   ├── features/           # feature engineering
│   │   ├── __init__.py
│   │   ├── performance_features.py     # 用 appearances/games 做 per-90、delta...
│   │   ├── club_league_features.py     # 用 clubs/competitions/club_games
│   │   ├── market_features.py          # 用 player_valuations/transfers
│   │   └── assemble_features.py        # 把上面全部 merge 成 snapshot
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   │
│   │   ├── regression/     # 成員 1 的地盤
│   │   │   ├── __init__.py
│   │   │   ├── train_regression.py     # 訓練 LightGBMRegressor
│   │   │   ├── evaluate_regression.py  # 計算 RMSE, MAE, R2
│   │   │   └── shap_regression.py      # 產出 regression SHAP & top features
│   │   │
│   │   ├── classification/ # 成員 2 的地盤
│   │   │   ├── __init__.py
│   │   │   ├── build_labels.py         # 定義 breakout label
│   │   │   ├── train_classification.py # 訓練 LightGBMClassifier
│   │   │   ├── evaluate_classification.py
│   │   │   └── shap_classification.py  # 產出 clf SHAP & top features
│   │   │
│   │   └── build_recommendations.py    # 合併 reg + clf → player_recommendations
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py          # LLM prompt 模板
│   │   ├── schema.py           # 定義要丟給 LLM 的 JSON 結構
│   │   └── llm_client.py       # 包一層 call OpenAI/其他 LLM API
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py    # 簡單 logging 設定
│       ├── io_utils.py         # 讀寫 parquet/csv 的小工具
│       └── time_utils.py       # 處理日期/rolling window 等
│
└── app/                     # 前端 + API / Streamlit
    ├── streamlit_app.py     # UI 主程式（或 web_app.py）
    ├── components/          # 前端畫面拆模組（如果你想拆）
    │   ├── __init__.py
    │   ├── player_table.py      # 推薦列表 table
    │   ├── player_detail.py     # 單一球員詳情區塊
    │   └── charts.py            # 各種圖表（MV 曲線、bar chart）
    │
    ├── backend/             # 若你想前後端分離，可以在這掛一個 API
    │   ├── __init__.py
    │   └── api_server.py    # （可選）FastAPI / Flask：/players, /llm_report
    │
    └── mock_data/           # 你現在可以放假資料測試 UI
        └── mock_player_recommendations.csv

```

## 1. 三人分工總覽

| 成員 | 角色 | 主要任務 |
| --- | --- | --- |
| Quentin | LightGBM Regression | 建立 & 訓練 **一年後身價成長率 regression 模型**，輸出 `y_growth_pred` / `mv_pred`，並產出 regression 相關的 features & SHAP。 |
| Felix | LightGBM Classification | 針對年輕球員建立 **breakout classification 模型**，輸出 `breakout_prob`，並產出 classification 相關的 features & SHAP。 |
| Eason | 前端 + LLM 報告 | 定義前端 data schema、畫面、API or 檔案讀取 + 設計 LLM prompt + 串接 & 顯示自然語言報告。 |

關鍵是：**大家共用同一個 `player_snapshot` base table**，然後 1 & 2 把自己的結果 merge 成一個 `player_recommendations` 表，給你用。

---

## 2. 共用中間產物：`player_snapshot`（由 Quentin & Felix 協作）

> 可以由任一個人負責寫，但 schema 要一起討論固定好。你可以先假設它長這樣來做前端。
> 

**檔名建議**：`data/player_snapshot.parquet`

**一列對應：某球員在某個時間點 t 的狀態**

主要欄位（示意）：

```
player_id                (int)
snapshot_date            (date)  # 對應 valuation 的日期 t
player_name              (string)
age                      (float)
position                 (string)
sub_position             (string)
club_id                  (int)
club_name                (string)
league_name              (string)
current_market_value     (float)   # MV_t
future_market_value      (float)   # MV_{t+365}, 計算 label 用
y_growth                 (float)   # log(MV_{t+365}) - log(MV_t)

# Performance features (過去 365 天)
minutes_per_90           (float)
goals_per_90             (float)
assists_per_90           (float)
shots_per_90             (float)
rating_mean              (float)
rating_std               (float)

# Performance delta vs 前一季
delta_goals_per_90       (float)
delta_minutes_per_90     (float)
delta_rating_mean        (float)

# Club / league context
club_total_value         (float)
club_win_rate            (float)
league_strength          (float)

# Market dynamics
mv_momentum_6m           (float)
mv_momentum_12m          (float)
has_recent_transfer      (int 0/1)

# Others...
...

```

> 你前端其實只要用到：player_id, player_name, age, position, club_name, current_market_value + 一些 performance features，就足夠顯示。
> 

---

## 3. Quentin：LightGBM Regression（預測身價成長）

### 🎯 目標

對每個 `player_snapshot`，預測 `y_growth_pred`，並推回未來一年預測身價 `mv_pred_1y`，以及每個球員的特徵重要度（for LLM & 解釋）。

### Step-by-step

1. **讀取原始資料 & 建立 snapshot**
    - Input：`players`, `player_valuations`, `appearances`, `games`, `clubs`, `competitions`
    - 建立上面說的 `player_snapshot` 表（pandas / DuckDB）。
2. **建立 Regression 訓練集**
    - `X_reg`：所有 feature 欄位（不含 target）
    - `y_reg = y_growth`
3. **切 train / valid / test**
    - 依年份分，例如：
        - train：2012–2021 snapshot_date
        - valid：2021–2023
        - test：2023–2025
4. **訓練 LightGBMRegressor**
    - 設定基本 hyperparameters（num_leaves, learning_rate, n_estimators…）
    - 評估：RMSE / MAE / R²
5. **產生預測與未來身價**
    - `y_growth_pred = model.predict(X_reg)`
    - `mv_pred_1y = current_market_value * exp(y_growth_pred)`
6. **SHAP / Feature Importance**
    - 使用 `shap.TreeExplainer(lgbm_model)`
    - 對 test set 的每個球員產生：
        - `shap_values[i]` → 每個 feature 對該球員 growth 預測的貢獻
    - 只保留 top K 特徵（例如 5 個）與其 SHAP 值，方便給前端 & LLM。

### Regression 輸出檔案（給成員 3）

**檔名**：`data/regression_outputs.parquet`

欄位示意：

```
player_id
snapshot_date
y_growth_pred            (float)
mv_pred_1y               (float)
reg_shap_top_features    (string, JSON-encoded)
                         # 例如：'[{"feature": "minutes_per_90", "shap_value": 0.12}, ...]'

```

---

## 4. Felix：LightGBM Classification（Breakout Prediction）

### 🎯 目標

對年輕球員預測「Breakout 機率」，並提供 classification 的 SHAP 重要特徵。

### Step-by-step

1. **定義 Breakout Label**
    - 在 `player_snapshot` 中篩出 `age < 23` 的球員。
    - 在每個 `(position, age_bucket)` 群組內：
        - 計算 `y_growth` 百分位數
        - 標記 `breakout_label = 1` 若 `y_growth` 在 top 20% 或 15%。
2. **建立 Classification 訓練集**
    - `X_clf`：和 regression 一樣或略有調整。
    - `y_clf = breakout_label`
    - 只用年輕球員（符合條件的 snapshot）訓練。
3. **切 train / valid / test**
    - 依年份分，例如：
        - train：2012–2021 snapshot_date
        - valid：2021–2023
        - test：2023–2025
4. **訓練 LightGBMClassifier**
    - 評估：AUC-ROC、Precision@K、Recall@K（重點是前面 ranking）。
5. **產生預測**
    - `breakout_prob = model.predict_proba(X_clf)[:, 1]`
6. **SHAP / Feature Importance**
    - 用 `shap.TreeExplainer` 對 classifier
    - 產出每個球員的 top K 特徵及 shap 值。

### Classification 輸出檔案（給Eason）

**檔名**：`data/classification_outputs.parquet`

欄位示意：

```
player_id
snapshot_date
breakout_prob               (float, 0~1)
clf_shap_top_features       (string, JSON-encoded)
                            # 例如：'[{"feature": "delta_goals_per_90", "shap_value": 0.20}, ...]'

```

---

## 5. Quentin & Felix 聯合：生成 `player_recommendations`

兩邊的 outputs + 原本的 snapshot merge 成 **前端主用表**。

**檔名**：`data/player_recommendations.parquet` / `player_recommendations.csv`

**Join key**：`(player_id, snapshot_date)`

欄位示例（這是你前端/LLM 主要依賴的 schema，超重要 ❗️）：

```
# 基本資訊
player_id, snapshot_date, player_name, age, position, sub_position
club_name, league_name

# 市場價值
current_market_value, mv_pred_1y, y_growth_pred

# 機器學習預測
breakout_prob, undervalued_score

# 表現數據
minutes_per_90, goals_per_90, assists_per_90
delta_goals_per_90, delta_minutes_per_90
rating_mean, mv_momentum_12m

# 發展曲線數據（Development）
expected_value_million, expected_ga_per_90, expected_minutes_per_90
valuation_above_curve, performance_above_curve, minutes_above_curve
aging_score, development_tier
peak_age, years_since_peak_value, valuation_slope_24m

# 可解釋性
reg_shap_top_features, clf_shap_top_features

# 其他
mv_history (JSON格式的歷史市值)
img_url (球員照片URL)

```

> 你現在就可以先假造一些這樣 schema 的 CSV 來開發前端 & LLM prompt。
> 

---

## 6. Eason：前端 + LLM 分析報告

### 🎯 目標

- 做出一個可以：
    - 篩選條件（聯賽、年齡、位置）
    - 顯示推薦球員列表（underpriced / breakout）
    - 點選球員 → 顯示圖表（MV 曲線、表現） + 丟資料給 LLM → 顯示「分析報告」

### Step-by-step

### Step 3.1：先用假資料開發前端 UI

1. 建立一個 `mock_player_recommendations.csv`，欄位按上面 schema 來。
2. 選擇技術：
    - 最快：**Streamlit**（不用前後端分離）
    - 或 React / Next.js + backend API（如果你想秀前端實力）
3. 初版畫面設計：

**頁面 1：球員列表**

- 篩選器：
    - League / Competition
    - Age range
    - Position
    - Min breakout_prob / Min undervalued_score
- 表格欄位：
    - Player, Age, Position, Club, Current MV, Pred MV, Growth %, Breakout Prob, Undervalued Score

**頁面 2：球員詳情頁（點擊 row 後）**

- 顯示：
    - Player name, age, position, club, league
    - Current MV vs Pred MV
    - 基本 stats: minutes/90, goals/90, assists/90, rating_mean, mv_momentum_12m
    - SHAP top features（用簡單 tag 顯示）
- 下方：一個「Generate LLM Report」按鈕

### Step 3.2：設計 LLM Prompt & Input Schema

你要從 `player_recommendations` 的一列資料，抽一個 JSON 給 LLM，例如：

```json
{
  "player_name": "John Doe",
  "age": 21,
  "position": "Forward",
  "club_name": "Midtable FC",
  "league_name": "Italian Serie A",
  "current_market_value": 8000000,
  "mv_pred_1y": 15000000,
  "y_growth_pred": 0.65,
  "breakout_prob": 0.78,
  "undervalued_score": 7000000,
  "key_stats": {
    "minutes_per_90": 78.3,
    "goals_per_90": 0.55,
    "assists_per_90": 0.18,
    "delta_goals_per_90": 0.30,
    "delta_minutes_per_90": 15.0,
    "rating_mean": 7.25,
    "mv_momentum_12m": 0.40
  },
  "reg_shap_top_features": [
    {"feature": "goals_per_90", "shap_value": 0.22},
    {"feature": "minutes_per_90", "shap_value": 0.15},
    {"feature": "mv_momentum_12m", "shap_value": 0.10}
  ],
  "clf_shap_top_features": [
    {"feature": "delta_goals_per_90", "shap_value": 0.25},
    {"feature": "age", "shap_value": 0.18}
  ]
}

```

**Prompt 範例（英文）**：

> You are a football scouting analyst.
> 
> 
> Given the following player data and model outputs, explain in a concise scouting report:
> 
> - Why this player might be undervalued in the market.
> - Why this player has a high or low probability of breaking out in the next season.
> - Mention specific stats (minutes, goals, assists, rating, growth) and context (club, league, age).
> - Explain the most important model features in natural language.
> - End with a one-line recommendation (e.g., "Recommended as a high-upside signing for mid-tier clubs").
> 
> Player data:
> 
> ```json
> {player_json_here}
> 
> ```
> 

你也可以請 LLM 回傳結構化回答，例如：

```json
{
  "summary": "...",
  "undervaluation_reason": "...",
  "breakout_reason": "...",
  "risk_factors": "...",
  "one_line_recommendation": "..."
}

```

然後前端渲染成卡片。

### Step 3.3：之後串真實模型輸出

等成員 1 & 2 產出真正的 `player_recommendations.parquet`/`csv`：

1. 換成讀真實檔案（schema 不變）。
2. 若用 API，可加一層 FastAPI / Flask，給前端：
    - `GET /players` → list / filter players
    - `GET /players/{player_id}` → details
    - `POST /players/{player_id}/llm_report` → 傳 JSON 給 LLM，回文字