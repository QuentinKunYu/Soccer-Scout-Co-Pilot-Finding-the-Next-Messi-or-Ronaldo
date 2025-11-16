import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump


# ----------------- 路徑設定 -----------------

PLAYER_SNAPSHOT_PATH = "data/processed/player_snapshot.parquet"
REG_OUTPUT_PATH = "data/processed/regression_outputs.parquet"
REG_METRICS_PATH = "data/processed/regression_metrics.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_regressor.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "xgb_features.json")


# ----------------- 資料讀取 & feature 構建 -----------------

def load_player_snapshot(path: str = PLAYER_SNAPSHOT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not np.issubdtype(df["snapshot_date"].dtype, np.datetime64):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    從 player_snapshot 建 feature matrix & target。
      - target: y_growth
      - 返回 X_df, y, data (與 X_df / y 對齊的原表)
    """
    target_col = "y_growth"

    # 數值特徵
    num_cols = [
        "age",
        "height_in_cm",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "mv_ratio_to_peak",
        "years_to_contract_end",
        "minutes_total",
        "games_played",
        "minutes_per_game",
        "goals_per_90",
        "assists_per_90",
        "delta_minutes_total",
        "delta_goals_per_90",
        "delta_assists_per_90",
        "club_total_market_value",
        "club_win_rate",
        "club_goal_diff_per_game",
        "league_strength",
    ]

    # 二元 flags
    bin_cols = [
        "league_is_major",
        "is_top5_league",
        "has_recent_transfer",
        "moved_to_bigger_club_flag",
    ]

    # 類別特徵（one-hot）
    cat_cols = [
        "position",
        "sub_position",
        "foot",
        "country_of_citizenship",
        "current_club_name",
        "league_name",
        "league_country",
    ]

    # 只保留有 label 的 rows
    data = df.dropna(subset=[target_col]).copy()

    if not np.issubdtype(data["snapshot_date"].dtype, np.datetime64):
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")

    # numeric / binary
    X_num = data[[c for c in num_cols if c in data.columns]].copy()
    X_bin = data[[c for c in bin_cols if c in data.columns]].copy()

    # categorical -> one-hot
    cat_present = [c for c in cat_cols if c in data.columns]
    if cat_present:
        X_cat = pd.get_dummies(
            data[cat_present],
            dummy_na=True,
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=data.index)

    # 合併
    X_df = pd.concat([X_num, X_bin, X_cat], axis=1)
    y = data[target_col].values

    print(f"[build_feature_matrix] X shape = {X_df.shape}, y length = {len(y)}")
    return X_df, y, data


# ----------------- 時間切分 & Rolling Validation -----------------

def split_by_time(
    data: pd.DataFrame,
    X_df: pd.DataFrame,
    y: np.ndarray,
    train_end_year: int = 2020,
    valid_start_year: int = 2021,
    valid_end_year: int = 2022,
    test_start_year: int = 2023,
):
    """
    最終模型用的 train/valid/test 切法：
      - train: <= train_end_year
      - valid: valid_start_year ~ valid_end_year
      - test : >= test_start_year
    """
    years = data["snapshot_date"].dt.year

    train_mask = years <= train_end_year
    valid_mask = (years >= valid_start_year) & (years <= valid_end_year)
    test_mask = years >= test_start_year

    X_train = X_df[train_mask]
    y_train = y[train_mask]

    X_valid = X_df[valid_mask]
    y_valid = y[valid_mask]

    X_test = X_df[test_mask]
    y_test = y[test_mask]

    print(
        "[split_by_time] "
        f"train={X_train.shape}, valid={X_valid.shape}, test={X_test.shape}"
    )

    masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks


def rolling_validation_scores(X_df, y, data, model_params, val_years):
    """
    滾動式驗證：
      Fold k: Train < val_year_k, Validate == val_year_k
    """
    years = data["snapshot_date"].dt.year
    results = []

    for val_year in val_years:
        train_mask = years < val_year
        valid_mask = years == val_year

        X_train, y_train = X_df[train_mask], y[train_mask]
        X_valid, y_valid = X_df[valid_mask], y[valid_mask]

        if len(X_valid) == 0 or len(X_train) == 0:
            continue

        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_valid, y_pred))
        r2 = float(r2_score(y_valid, y_pred))

        print(f"[Fold {val_year}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
        results.append({"year": val_year, "rmse": rmse, "mae": mae, "r2": r2})

    if results:
        rv_df = pd.DataFrame(results)
        print("\n[Rolling Validation] summary:")
        print(rv_df.describe())
    else:
        print("[Rolling Validation] No valid folds (check year ranges).")

    return results


# ----------------- 評估函數 -----------------

def evaluate_regression(model, X, y, split_name: str):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    print(f"[{split_name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return rmse, mae, r2, y_pred


# ----------------- 主流程 -----------------

def main():
    # 1) 讀取 snapshot
    df = load_player_snapshot()

    # 2) feature matrix + target
    X_df, y, data = build_feature_matrix(df)
    years = data["snapshot_date"].dt.year
    unique_years = sorted(years.unique())
    print(f"[main] years in data: {unique_years[:5]} ... {unique_years[-5:]}")

    # 3) XGBoost 參數（之後可以微調）
    model_params = dict(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )

    # 4) Rolling validation：例如 2019–2023
    #    只用資料裡真的存在的年份
    candidate_val_years = [2019, 2020, 2021, 2022, 2023]
    val_years = [y for y in candidate_val_years if y in unique_years]

    if len(val_years) >= 1:
        print(f"[main] Rolling validation years: {val_years}")
        rolling_validation_scores(X_df, y, data, model_params, val_years)
    else:
        print("[main] No overlapping years for rolling validation, skip.")

    # 5) 最終 train / valid / test 切分（跟之前一樣）
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks = split_by_time(
        data,
        X_df,
        y,
        train_end_year=2020,
        valid_start_year=2021,
        valid_end_year=2022,
        test_start_year=2023,
    )

    # 6) 用整個 train set 訓練最終模型
    model = XGBRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True,
    )

    # 7) train / valid / test 評估
    _, _, _, _ = evaluate_regression(model, X_train, y_train, "train")
    _, _, _, _ = evaluate_regression(model, X_valid, y_valid, "valid")
    rmse_test, mae_test, r2_test, _ = evaluate_regression(
        model, X_test, y_test, "test"
    )

    # 8) 對所有 row 做預測，算 mv_pred_1y
    y_pred_all = model.predict(X_df)
    data_out = data.copy()
    data_out["y_growth_pred"] = y_pred_all
    data_out["mv_pred_1y"] = data_out["market_value_in_eur"] * np.exp(
        data_out["y_growth_pred"]
    )

    # 9) 存 regression_outputs（先不含 SHAP，之後 shap_regression 會補 reg_shap_top_features）
    os.makedirs(os.path.dirname(REG_OUTPUT_PATH), exist_ok=True)

    cols_to_save = [
        "player_id",
        "snapshot_date",
        "y_growth",
        "y_growth_pred",
        "market_value_in_eur",
        "mv_pred_1y",
    ]
    cols_to_save = [c for c in cols_to_save if c in data_out.columns]

    data_out[cols_to_save].to_parquet(REG_OUTPUT_PATH, index=False)
    print(f"[train_regression] Saved regression_outputs -> {REG_OUTPUT_PATH}")

    # 10) 存 metrics
    metrics = {
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
    }
    os.makedirs(os.path.dirname(REG_METRICS_PATH), exist_ok=True)
    with open(REG_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_regression] Saved regression_metrics -> {REG_METRICS_PATH}")

    # 11) 存 model + feature 名（給 SHAP 用）
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"[train_regression] Saved XGBoost model -> {MODEL_PATH}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X_df.columns), f)
    print(f"[train_regression] Saved feature names -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
