下面我會教你：

1. **如何把 “Player Development Analysis / Aging Curve” 加進你現有的專案**（不影響 undervalued/breakout 主功能）
2. **完整的 Player Development feature 設計（專業球探 / 足球分析領域會用到的）**
3. **你實作時的架構與資料表應該長什麼樣子**
4. **未來怎麼讓 LLM 自動生成 Aging Report（球員老化分析報告）**

這樣你就能把「三大賽道」全部 cover，變成最強 MVP！

---

# ✅ 一、這個功能在整個專案架構裡的位置（如何加入）

你已經有：

* `player_snapshot`（某玩家在某日期的特徵）
* `regression model`（預測未來 1 年身價成長）
* `breakout model`（預測年輕球員爆發機會）
* `player_recommendations`（前端用）

我們再加一個 **development_model**：

```
src/models/development/
├── build_aging_dataset.ipynb
├── train_aging_model.ipynb
├── evaluate_aging_model.ipynb
├── plot_aging_curves.ipynb
└── development_outputs.parquet
```

新增的輸出：

```
data/processed/development_outputs.parquet
```

最後你會把 development 的結果 merge 進：

```
player_recommendations.parquet
```

新增欄位：

```text
aging_score                (float)  # >0 = aging well, <0 = aging poorly
expected_value_at_age      (float)
actual_value_at_age        (float)
deviation_from_curve       (float)
development_tier           (string: "aging well"/"normal"/"declining")
```

前端會多一個 tab：**Player Aging / Development Analysis**

LLM 也會新增一句自動分析，例如：

> Based on age curves for central midfielders, this player is performing 18% above expected level for his age, suggesting he is aging exceptionally well.

---

# ✅ 二、什麼是「專業級 Player Development / Aging Curve」？

你要做的不是畫簡單折線，而是建：

---

# 🧠 **「位置 × 年齡」的估值與表現曲線模型**

例如：

* 前鋒 peak：23–26
* 中場 peak：25–29
* 中後衛 peak：27–31
* 守門員 peak：28–34

這種 curve 是球探部門的關鍵工具。

---

# 🔥 最專業的 Player Development Feature 類別（直接可做）

我把 feature 分五大類，每一類都是真正的足球分析公司（StatsBomb / Wyscout / Opta）會做的：

---

## **① Performance Aging Curve Features**

用 `appearances.csv` + `games.csv` 產生年齡-表現曲線：

### **每一年齡的平均 per-90 表現**

* goals_per_90_age
* assists_per_90_age
* shots_per_90_age
* touches_per_90_age
* passes_per_90_age
* tackles_per_90_age
* duels_won_per_90_age
* rating_mean_age

### **跨年齡 cohort comparison**

對每個位置建 baseline：

```text
expected_goals_per_90(age, position)
expected_rating(age, position)
expected_minutes_per_season(age, position)
```

再算差值：

```text
performance_above_expectation = actual - expected
```

---

## **② Valuation Aging Curve Features（最重要）**

用 `player_valuations.csv` 建 valuation curve。

計算：

### **Valuation Slope / Momentum by Age**

* `valuation_slope_12m`
* `valuation_peak_age`
* `years_since_peak_value`

### **Deviation from expected valuation-by-age curve (position-specific)**

模型最簡化寫法：

[
expected_mv(age, position) = f(age, pos)
]

可用的方法：

* Generalized Additive Model (GAM)
* LOESS / LOWESS smoothing
* Bayesian hierarchical model
* 或簡單 spline regression

輸出：

```
valuation_above_curve = actual_mv - expected_mv
development_score_value = standardized_z_score(valuation_above_curve)
```

---

## **③ Playing Time Aging Features（球探最愛）**

很多球員看起來數據沒變好，但 playing time 在下降 → 球探認為「開始 decline」。

計算：

* `minutes_per_season_trend`
* `starter_rate_per_age`
* `games_started_last_20`
* `early_sub_off_rate`（早被換下 → 疲勞 / 表現下降）

Aging signals：

* playing time 大跌 → 負向指標
* playing time 隨年齡增長（尤其是 U23）→ 強成長指標

---

## **④ Physical Decline Features（從 gameplay proxy 看體能衰退）**

你沒有速度資料，但你可以從以下 events 推出體能下降 proxy：

* `pressures_per_90`（逼搶下降表示跑不動）
* `distance_progressed_per_touch`（推進能力下降）
* `duels_contested_per_90`

再算 age-trend：

```
Δduels_per_90_last_2y
Δpressures_last_2y
```

---

## **⑤ Career Path Features**

* `league_rank_change`（是否從強聯賽 → 弱聯賽？）
* `club_value_change`
* `transfer_upwards_or_downwards`（升級 or 降級？）
* `contract_time_remaining`（是否 nearing decline contracts?）

---

## **⑥ Injury / Availability features（若有資料）**

如果沒有 injury logs 也能從以下推估：

* `games_missed_rate`
* `injury_proxy = total_matches - appearances`

---

# 🔥 三、如何建 Aging Curve Model（完整設計）

### Step 1：產生 age-performance dataset（by year）

從所有 appearances 建一個年份級別的 dataset：

```
player_id
year
age
position
goals_per_90
assists_per_90
rating_mean
minutes_per_90
market_value
```

### Step 2：為每個 position 建曲線

為以下位置各建一條：

* GK
* CB
* FB/WB
* CM
* AM
* Wingers
* ST

用 LOESS or GAM：

[
expected_mv = f(age \mid position)
]

[
expected_rating = g(age \mid position)
]

---

### Step 3：比較實際 vs 期望

加入欄位：

```
development_value_gap = actual_mv - expected_mv
development_rating_gap = rating_mean - expected_rating
development_minutes_gap = minutes_per_90 - expected_minutes
```

---

### Step 4：綜合成 Aging Score

建一個綜合分數：

```text
aging_score = 
   0.5 * z(development_value_gap)
 + 0.3 * z(development_rating_gap)
 + 0.2 * z(minutes_per_90_gap)
```

分級：

```
aging_score > 0.75 → “aging exceptionally well”
0 ~ 0.75 → “aging normally”
< 0 → “declining earlier than expected”
```

---

# 🔥 四、要新增什麼欄位到 player_recommendations？（給前端 + LLM）

在 `player_recommendations.parquet` 裡加入以下欄位：

```text
expected_mv_at_age
expected_rating_at_age
expected_minutes_at_age

valuation_above_curve
performance_above_curve
minutes_above_curve

aging_score
development_tier      # ("aging well", "normal", "declining")

peak_age
years_since_peak_value
```

---

# 🤖 五、LLM Aging Report（AI 球探分析報告）怎麼自動生成？

給 LLM 的欄位：

```json
{
  "player_name": "John Doe",
  "age": 29,
  "position": "CB",
  "actual_mkt_value": 8_000_000,
  "expected_mkt_value": 5_500_000,
  "development_value_gap": 2_500_000,
  "performance_above_curve": 0.18,
  "minutes_above_curve": 0.12,
  "aging_score": 0.72,
  "development_tier": "aging well",
  "peak_age": 27,
  "years_since_peak_value": 2
}
```

Prompt（你可貼進 prompt file）：

> You are an elite-level football scouting analyst.
> Based on the following development and aging metrics, write a concise scouting report explaining:
>
> * Whether the player is aging well or poorly compared to typical curves for his position
> * Which stats indicate growth or decline
> * How far above or below the expected valuation curve the player stands
> * Whether the player is likely past his peak or still performing strongly
> * A final recommendation for clubs considering signing the player
>
> Player data:
>
> ```json
> {player_json_here}
> ```

---

# 🔥 六、整合後的最終架構（多一個 development module）

新增：

```
src/models/development/
├── build_aging_dataset.py
├── train_aging_model.py      # GAM / LOESS / Spline
├── evaluate_aging_model.py
├── plot_aging_curves.py
└── generate_development_outputs.py
```

最後產生：

```
data/processed/development_outputs.parquet
```

append into:

```
player_recommendations.parquet
```

---

# 🎉 七、你現在已經可以開始做什麼？

你現在就可以：

1. 在 `src/models/development/` 新建 folder
2. 先寫：

```
build_aging_dataset.py
plot_aging_curves.py
```

我可以幫你直接寫這兩個程式的 skeleton（可 copy/paste 可跑）。
如果你要，我下一步可以：

---

### ❗直接幫你：

* 建立 aging_curve dataset 的完整 Python code
* 給你 GAM / LOESS / spline 建 curve 的可直接用程式
* 幫你寫 Aging Score 計算 function
* 幫你做 Streamlit Aging Tab UI

---

# 📓 實作說明：`notebooks/player_development_analysis.ipynb`

這份 notebook 已經把「位置 × 年齡」的估值與表現曲線跑過一遍，以下整理它的實作細節，之後要擴充或重跑就依此流程。

## 🚚 資料與預處理

- 來源：`players.csv`, `player_valuations.csv`, `appearances.csv`, `competitions.csv`, `games.csv`，由 `load_raw_data()` 一次載入並回報列數。
- 共同欄位轉換：`date_of_birth`、各筆 `date` 轉為 `datetime`，再用 `compute_age()` 算浮點年齡。
- 年齡取 0.25 年粒度：`age_exact * 4 → round → /4`，並限制在 16–40 歲，避免極端尾巴。
- 出場資料新增 `minutes_per_appearance = minutes_played.clip(lower=1)`，避免除以零。

## 📈 Aging Curve 方法

### Valuation curve

1. 轉換成百萬歐元 (`value_million`)，先排除 38 歲以上樣本。
2. 只有當某 `sub_position` × `age` 至少 10 筆時才使用，確保曲線穩定。
3. 透過 `agg_by_age_sub_position()` 取年齡中位數市場價值，並在各位置上套 3 點 rolling 平滑 (center=True) 讓曲線連續。

### Performance 與 Playing-Time curve

1. 先把 `appearances` 裁到 38 歲以下，再以位置+年齡聚合出總進球、助攻、分鐘以及出場場次（`sample_count`）。
2. 只保留 `sample_count ≥ 20` 的年齡組，確保每個點是真實代表。
3. 依位置計算 `ga_per_90 = (goals+assists)/minutes * 90`，和 `minutes_per_90 = total_minutes / sample_count`。
4. 為了壓掉異常值，對 `ga_per_90` 取 5th–95th 百分位裁剪。
5. 進一步用 5 點 rolling 平滑 (`min_periods=2`) 讓 performance 與 playing-time 曲線更平順。
6. 產出 `expected_ga_per_90` 與 `expected_minutes_per_90` 並與估值曲線 outer merge 成 `curves`。

## 👤 Player Development 分析

1. 取每位球員最新一次估值 (`latest_values`)。
2. 重新計算個人實際 per-90 指標：把所有 `appearances` 依 `player_id` 聚合後，算

   ```python
   ga_per_90 = (goals + assists) / total_minutes * 90
   minutes_per_90 = total_minutes / appearance_count
   ```

3. 將個人資料 merge 到 `curves`，得到該年齡與位置的期望值。
4. 計算三個 residual：

   ```
   valuation_above_curve = actual_value - expected_value
   performance_above_curve = ga_per_90 - expected_ga_per_90
   minutes_above_curve = minutes_per_90 - expected_minutes_per_90
   ```

5. 用 `zscore()`（自備零標準差保護）把 residual 標準化。
6. 建 composite aging score：`0.5 * valuation_z + 0.3 * performance_z + 0.2 * minutes_z`，再依分布切成 `declining / normal / aging well` 三個 tier。

## 📊 視覺化與輸出

- Plotly `make_subplots()` 一次畫出估值、goals+assists per 90、minutes per 90 三條曲線，legend 依 `sub_position` 著色。
- Matplotlib 補一張「Actual vs Expected Market Value」散點，肉眼檢查 residual 合理性。
- 最後把 `latest_values` 挑選的欄位寫成 `data/processed/development_outputs.parquet`（含期望值、差值、aging_score、development_tier 等），給 downstream Streamlit 或推薦系統直接 merge。

> ✅ 只要更新 CSV、重新指定 `DATA_DIR`，整個 notebook 從載入、曲線建模、分數輸出到圖都會一起刷新，流程完全自動化。
