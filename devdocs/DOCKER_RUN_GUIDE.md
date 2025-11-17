# Docker 運行說明

## 1. 先決條件
- 你已安裝 Docker 與 docker-compose（Docker Desktop 也可）。
- `OPENAI_API_KEY` 等敏感資訊放在 `.env`（可複製 `.env.example` 後填入）。

## 2. 建立映像檔
```bash
docker compose build
```
這會依照 `Dockerfile` 先裝好 requirements，再把整個專案放進映像檔中。

## 3. 啟動服務
```bash
docker compose up
```
- 預設會把 Streamlit 跑在 `http://localhost:8501`。
- `docker-compose.yml` 有把專案目錄掛進容器，所以你在本機改 code，重新整理頁面就會看到更新。

## 4. 常見操作
- 只想重啟服務可用 `docker compose up --build`（會順便更新映像）。
- 想清掉背景容器可按 `Ctrl+C`，或在另一個終端下 `docker compose down`。

## 5. 問題排除
- 如果看到權限錯誤，確認 `.env` 有存在且權限正確。
- 若要使用真實 LLM，記得在 `.env` 設定 `OPENAI_API_KEY`，容器會自動讀取。
