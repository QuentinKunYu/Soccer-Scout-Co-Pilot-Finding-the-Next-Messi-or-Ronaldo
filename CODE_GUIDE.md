# 程式解讀指南（Eason 前端 + LLM 原型）

## 1. 整體流程
1. `app/streamlit_app.py` 是 Streamlit 入口：
   - 透過 `src/utils/io_utils.load_player_recommendations` 讀入 `app/mock_data/mock_player_recommendations.csv`。
   - 呼叫 `app.components.filters.render_filters` 在 sidebar 產生篩選器並回傳過濾後的 DataFrame 與狀態。
   - 呼叫 `app.components.player_table.render_player_table` 顯示表格並取得使用者選到的球員。
   - 若有選擇球員，交給 `app.components.player_detail.render_player_detail` 顯示詳細資訊與圖表。
   - 按下「Generate LLM Report」會用 `src.llm.schema.PlayerLLMInput` 轉換資料列，並透過 `src.llm.llm_client.LLMClient` 送進 LLM（或 fallback stub），將輸出交給 `src.llm.prompts.render_report_to_markdown` 轉成 Markdown 呈現。

## 2. Mock 資料
- 檔案：`app/mock_data/mock_player_recommendations.csv`。
- 欄位遵循 roadmap 中 `player_recommendations` schema，額外加上 `mv_history` 以便畫市場價值折線圖。
- JSON 欄位（SHAP / MV history）會在 `load_player_recommendations` 中自動轉成 Python list/dict，前端直接使用即可。

## 3. LLM 區塊
- `src/llm/schema.py`：定義 `KeyStats`、`FeatureImportance` 與 `PlayerLLMInput`，集中處理欄位轉換與驗證，確保送給 LLM 的 JSON 統一。
- `src/llm/prompts.py`：`build_player_prompt` 將 payload 轉成指令文本；`render_report_to_markdown` 將 LLM 回傳的 JSON 轉成人類可讀段落。
- `src/llm/llm_client.py`：
  - 若系統有 `OPENAI_API_KEY` 以及 `openai` SDK，會直接呼叫 API 並要求 JSON schema 回覆。
  - 否則使用 `_build_stub_response` 根據統計數字產生 deterministic 回覆，方便本地測試。

## 4. UI Components
- `app/components/filters.py`：建立 sidebar 篩選器，回傳 `FilterState` 與篩好的資料。
- `app/components/player_table.py`：負責格式化欄位（轉成百萬歐元、百分比）並顯示於資料表，同時用 selectbox 讓使用者選擇球員。
- `app/components/player_detail.py`：顯示 KPI、統計數字、`Altair` MV 曲線，以及 SHAP 文字列表。

## 5. 如何延伸
1. 把 `app/mock_data/mock_player_recommendations.csv` 換成 `data/processed/player_recommendations.parquet`（可先在 `io_utils` 中新增 reader）。
2. 在 `LLMClient` 裡改成實際部署模型或 FastAPI endpoint，Streamlit 邏輯不用變。
3. 若要切換到 React / API 架構，可以沿用相同 schema 與 LLM prompt，後端只需提供 `/players` 與 `/players/{id}/llm_report` 兩個 route。
