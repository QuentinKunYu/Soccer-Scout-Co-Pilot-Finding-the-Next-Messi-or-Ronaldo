超關鍵的一步來了 🧠：
你現在要做的是「把所有時序 &關聯資料壓成一張 **tabular player_snapshot 表**，讓 regression / classification 都能吃」。

我幫你分成三個層次講：

1. **先講整體設計原則（避免踩雷：時間泄漏 / noisy feature）**
2. **再給一份「可以直接實作」的 feature 設計清單（按類別分）**
3. **說明：哪些是 regression、哪些對 classification（breakout）特別重要**

---

## 一、設計 tabular features 的大原則

### 1. 一個 row 表示什麼？

先固定概念：

> 一筆資料 = 「某球員在某個時間點 `snapshot_date` 的狀態」

這個 `snapshot_date` 可以直接用 `player_valuations.date`，例如每一次有 market value 更新都是一個 snapshot。

之後所有 feature 計算，**只能用 `snapshot_date` 之前的資料**。

---

### 2. 要處理的時間邏輯

對每個 `(player_id, snapshot_date)`：

* 取過去某個 window 的資料（例如 365 天 / 2 季）
* 做 aggregation：

  * sum / mean / per-90 / 成長率 / variance…

常見 window：

* `last_season`（用 `season` 欄位）
* `last_365_days`
* 甚至 `last_90_days` 看近期狀態

重點：**所有特徵都必須是「在那個時間點可見的資訊」**。

---

### 3. 盡量 per-90 / normalized

足球 data 很容易有：

* 有人踢 3000 分鐘，有人只踢 300 分鐘
* 直接用 raw goals、assists 會 bias 給出場多的人

建議：

* 用 `goals_per_90`, `assists_per_90`, `shots_per_90` 等
* 再加上 `minutes_total` / `games_played` 當作「機會」指標

---

### 4. Regression & Classification feature 可以共用

基本上你可以：

* 建一張 **通用 feature 表 `player_snapshot`**
* regression / classification 只是在這張表上：

  * 換 label (`y_growth` vs `breakout_label`)
  * 可能 drop 或加幾個特定 feature

好處：pipeline 單純，team 分工也清楚。

---

## 二、實際可用的 Feature 設計（逐類型）

下面這些就是你可以放進 `player_snapshot` 的欄位們。
我也會標註：

* `R` = regression (market value growth) 很重要
* `C` = classification (breakout) 很重要

---

### A. Player Profile（靜態＋緩慢變化）

**來源**：`players.csv` + `player_valuations.date`

| Feature                                | 說明                                             | 用法                    |
| -------------------------------------- | ---------------------------------------------- | --------------------- |
| `age` (R, C)                           | 用 `snapshot_date - date_of_birth` 算            | breakouts 通常集中在 18–23 |
| `position`, `sub_position` (R, C)      | one-hot encode                                 | 不同位置價值曲線不同            |
| `height_in_cm` (R)                     | 數值，或分桶                                         | 對某些位置（CB/CF）有價值       |
| `preferred_foot` (L/R/Both) (R)        | one-hot                                        | 可選                    |
| `years_to_contract_end` (R)            | `contract_expiration_date - snapshot_date` (年) | 合約剩多久影響市場價值           |
| `country_of_birth` / `nationality` (R) | 可能簡化為「EU / non-EU」                             | 某些聯賽 quota 有關         |

---

### B. Current Market & Valuation History

**來源**：`player_valuations.csv`

| Feature                                    | 說明                                        | 用法                   |
| ------------------------------------------ | ----------------------------------------- | -------------------- |
| `current_market_value` (R, C)              | `market_value_in_eur` 在 snapshot_date     | regression 的 base 水平 |
| `highest_market_value` (R)                 | 過去所有紀錄中的 max                              | 看是不是已經「過巔峰」          |
| `mv_ratio_to_peak` (R, C)                  | `current_mv / highest_mv`                 | 小於 1 太多 = 可能受傷 /低估   |
| `mv_momentum_6m`, `mv_momentum_12m` (R, C) | 用最近 6 / 12 個月的 MV time series 做線性回歸 slope | 市場已經認為他在漲/跌          |
| `num_valuations_last_year` (R)             | Transfermarkt 更新頻率                        | 也某種程度代表關注度           |

---

### C. Performance Features（單季 / 滾動視窗）

**來源**：`appearances.csv` + `games.csv`（用 `player_id`, `game_id` join）

以「過去 365 天」為例（你也可以加 `last_season` 版本）：

#### 1. 上場與使用率（Playtime & Usage）

| Feature                       | 說明                                                 | 用法            |
| ----------------------------- | -------------------------------------------------- | ------------- |
| `minutes_total_365` (R, C)    | 過去 365 天總上場分鐘                                      | 機會多本來就 valued |
| `games_played_365` (R, C)     | 多少場有出場                                             |               |
| `minutes_per_game_365` (R, C) | `minutes_total_365 / games_played_365`             | 穩定首發 vs 替補    |
| `starter_rate_365` (R, C)     | 先發場次 / 總出場場次（可用 `game_lineups.type == "starting"`） | 機會指標          |

再對比前一季：

| Feature                   | 說明                        |
| ------------------------- | ------------------------- |
| `delta_minutes_total` (C) | 本季 vs 前一季的 minutes change |
| `delta_starter_rate` (C)  | 先發機會成長多少                  |

> breakout 通常會看到「minutes 大幅成長」這種 pattern。

#### 2. 攻擊數據（對前鋒 / 中場特別重要）

| Feature                           | 說明                     | 用法                |
| --------------------------------- | ---------------------- | ----------------- |
| `goals_per_90_365` (R, C)         | `goals / (minutes/90)` |                   |
| `assists_per_90_365` (R, C)       |                        |                   |
| `shots_per_90_365` (R)            |                        |                   |
| `shots_on_target_per_90_365` (R)  |                        |                   |
| `goal_involvements_per_90` (R, C) | `(goals+assists)/90`   | 對 regression 非常重要 |

對比前一季：

| Feature                    | 說明                   | 用法            |
| -------------------------- | -------------------- | ------------- |
| `delta_goals_per_90` (C)   | 今年 vs 去年的 goals/90 差 | growth signal |
| `delta_assists_per_90` (C) | 同上                   |               |

#### 3. 防守 / 中場數據（如果有）

看 `fouls`, `tackles`, `interceptions`…（如果 dataset 沒有這麼細，這塊可略）

#### 4. Rating 與穩定度

| Feature                  | 說明                   | 用法        |
| ------------------------ | -------------------- | --------- |
| `rating_mean_365` (R, C) | 過去 365 天平均 rating    | 品質        |
| `rating_std_365` (R)     | 表現穩定度                | 太高可能代表不穩  |
| `delta_rating_mean` (C)  | 今年 vs 去年的 rating 平均差 | 成長 signal |

---

### D. Injury / Availability Proxy（不直接有傷病，但可以用 minutes 推）

你沒有 injury 欄，但可以用以下 proxy：

| Feature                        | 說明               |
| ------------------------------ | ---------------- |
| `games_missed_ratio`           | 該季球隊比賽中，球員沒上場的比例 |
| `consecutive_games_missed_max` | 連續缺席場數最大值（可能是傷病） |

這些對 regression 再 fine-tune 時可以幫忙讓 model 不過度樂觀某些「剛回來的球員」。

---

### E. Club & League Context Features

**來源**：`clubs.csv`, `competitions.csv`, `club_games.csv`

#### 1. Club strength

| Feature                         | 說明               | 用法              |
| ------------------------------- | ---------------- | --------------- |
| `club_total_market_value` (R)   | 球隊總市值            | 大球會球員通常 premium |
| `club_value_rank_in_league` (R) | 在該 league 的 rank |                 |
| `club_win_rate_365` (R)         | 過去一年勝率           | 好球隊帶動球員身價       |
| `club_goal_diff_per_game` (R)   |                  |                 |

#### 2. League & competition

| Feature                 | 說明          | 用法                    |
| ----------------------- | ----------- | --------------------- |
| `league_name` (R, C)    |             | 報表用、one-hot           |
| `league_strength` (R)   | 可用該聯賽球隊平均市值 | 「英超 premium」可以表現出來    |
| `is_top5_league` (R, C) | 0/1         | 轉到 top5 league 也會帶動身價 |

---

### F. Transfer & Career Movement Features

**來源**：`transfers.csv`

| Feature                       | 說明                                        | 用法               |
| ----------------------------- | ----------------------------------------- | ---------------- |
| `has_recent_transfer` (R, C)  | 過去 12 個月是否有轉會                             |                  |
| `moved_to_bigger_club` (R, C) | 轉會時，目標球隊總市值 > 原球隊？                        | 升格視窗             |
| `transfer_fee_vs_mv` (R)      | `transfer_fee - market_value_at_transfer` | 高於市價買的球，可能後面更被期待 |

對 breakout：

* 轉到大球隊但尚未 fully break out 的年輕人，是典型候選。

---

### G. Label-related Derived Features（只用在 ranking 不會當 feature）

這些不會當 model input，但會放入 `player_recommendations` 表：

| Feature                    | 說明                                |
| -------------------------- | --------------------------------- |
| `y_growth_pred`            | regression model 預測出來             |
| `mv_pred_1y`               | `current_mv * exp(y_growth_pred)` |
| `undervalued_score`        | `mv_pred_1y - current_mv`         |
| `breakout_prob`            | classifier 預測機率                   |
| `rank_undervalued_overall` | 用於 dashboard 排名                   |
| `rank_breakout_young`      | 同上                                |

---

## 三、Regression vs Classification：重點 feature 有點不同

### Regression（預測 market value growth / future MV）

偏重：

* **現在的水準 + 表現 + club/league context**

  * `current_mv`, `goals_per_90`, `assists_per_90`, `rating_mean`, `club_total_value`, `league_strength`
* **中期趨勢**

  * `mv_momentum_12m`, `delta_rating_mean`, `delta_minutes_per_90`

你可以讓 regression 更看整體「價值合理不合理？」。

---

### Classification（breakout probability）

偏重：

* **成長率與機會變化**（比絕對值更重要）

  * `delta_goals_per_90`, `delta_assists_per_90`, `delta_minutes_total`, `delta_starter_rate`
* **年齡與階段**

  * `age`, `mv_ratio_to_peak`（年輕但離 peak 還很遠）
* **環境提升**

  * `moved_to_bigger_club`, `is_top5_league`, `club_win_rate_365`

你可以直覺把 classification 看成：

> 這個人是不是：「最近打得明顯更多 + 數據成長 + 處在有利環境」
> 如果是 → high breakout probability。

---

## 四、實作建議：最後 tabular 長什麼樣？

你可以讓 `player_snapshot` 看起來像這樣（簡化版）：

```text
player_id
snapshot_date
player_name
age
position
club_name
league_name

# Market & contract
current_market_value
highest_market_value
mv_ratio_to_peak
mv_momentum_12m
years_to_contract_end

# Performance - level
minutes_total_365
minutes_per_game_365
goals_per_90_365
assists_per_90_365
shots_per_90_365
rating_mean_365
rating_std_365

# Performance - growth
delta_minutes_total
delta_goals_per_90
delta_assists_per_90
delta_rating_mean

# Club & league
club_total_market_value
club_win_rate_365
league_strength
is_top5_league

# Transfer
has_recent_transfer
moved_to_bigger_club

# Labels (train 用)
y_growth
breakout_label
```

Regression / Classification 都可以：

* 用同一張表 `player_snapshot`
* 用 config（例如 `config/features.yaml`）指定各自要用哪些欄位

---

如果你願意，下一步我可以幫你：

* 把「用 pandas 實際從 `appearances` 做出 `goals_per_90_365` / `delta_goals_per_90`」寫成具體程式碼範例
* 或是幫你設計 `features.yaml`，明確列出 `numerical_features`, `categorical_features` 給 model 用。
