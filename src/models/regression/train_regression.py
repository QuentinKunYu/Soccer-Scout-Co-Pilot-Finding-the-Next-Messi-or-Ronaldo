import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump


# ---------- Helpers ----------

def load_player_snapshot(path: str = "data/processed/player_snapshot.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    # 確保 snapshot_date 是 datetime
    if not np.issubdtype(df["snapshot_date"].dtype, np.datetime64):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    從 player_snapshot 裡選出 feature + target。
    回傳:
        X_df: features (DataFrame, 保留欄位名給 SHAP 用)
        y:    target (np.array)
        data: 與 X_df, y 對齊的原始資料（有 player_id / snapshot_date / y_growth ...）
    """
    target_col = "y_growth"

    # 數值特徵
    num_cols = [
        "age",
        "height_in_cm",
        "market_value_in_eur",
        "club_total_value",
        "squad_size",
        "average_age",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_game",
        "delta_goals_per_90",
        "delta_assists_per_90",
        "delta_minutes_per_game",
        "mv_change_rate",
    ]

    # 類別特徵，用 one-hot
    cat_cols = [
        "position",
        "sub_position",
        "league_name",
        "league_country",
        "club_name",
        "country_of_citizenship",
    ]

    # 二元 flag
    bin_cols = [
        "league_is_major",
        "is_left_footed",
        "is_right_footed",
    ]

    # 只保留有 target 的 row
    data = df.dropna(subset=[target_col]).copy()

    # 再次保險 snapshot_date 是 datetime
    if not np.issubdtype(data["snapshot_date"].dtype, np.datetime64):
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")

    # 數值 + 二元
    X_num = data[[c for c in num_cols if c in data.columns]].copy()
    X_bin = data[[c for c in bin_cols if c in data.columns]].copy()

    # 類別 one-hot
    cat_cols_present = [c for c in cat_cols if c in data.columns]
    if cat_cols_present:
        X_cat = pd.get_dummies(
            data[cat_cols_present],
            dummy_na=True,
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=data.index)

    # 合併
    X_df = pd.concat([X_num, X_bin, X_cat], axis=1)

    # target
    y = data[target_col].values

    print(f"Feature matrix shape: {X_df.shape}, target length: {len(y)}")

    return X_df, y, data


def split_by_time(data: pd.DataFrame, X_df: pd.DataFrame, y: np.ndarray):
    """
    用 snapshot_date 的年份做 time-based split:
        train: year <= 2020
        valid: 2021–2022
        test:  year > 2022
    """
    years = data["snapshot_date"].dt.year

    train_mask = years <= 2020
    valid_mask = (years > 2020) & (years <= 2022)
    test_mask = years > 2022

    X_train = X_df[train_mask]
    y_train = y[train_mask]

    X_valid = X_df[valid_mask]
    y_valid = y[valid_mask]

    X_test = X_df[test_mask]
    y_test = y[test_mask]

    print("Train size:", X_train.shape, "Valid size:", X_valid.shape, "Test size:", X_test.shape)

    # 把 mask 也回傳給 SHAP / 其他用途
    masks = {
        "train": train_mask,
        "valid": valid_mask,
        "test": test_mask,
    }

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks


def evaluate_regression(model, X, y, split_name: str):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    print(f"[{split_name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return rmse, mae, r2, y_pred


# ---------- Main pipeline ----------

def main():
    # 1) 讀 player_snapshot
    df = load_player_snapshot()

    # 2) 建 feature matrix
    X_df, y, data = build_feature_matrix(df)

    # 3) time-based split
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks = split_by_time(
        data, X_df, y
    )

    # 4) XGBoost 模型
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        tree_method="hist",   # CPU 快速 histogram
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True,
    )

    # 5) 評估
    _, _, _, _ = evaluate_regression(model, X_train, y_train, "train")
    _, _, _, _ = evaluate_regression(model, X_valid, y_valid, "valid")
    rmse_test, mae_test, r2_test, y_pred_test = evaluate_regression(
        model, X_test, y_test, "test"
    )

    # 6) 對所有 row 做預測（方便後續 merge）
    y_pred_all = model.predict(X_df)
    data_out = data.copy()
    data_out["y_growth_pred"] = y_pred_all
    data_out["mv_pred_1y"] = data_out["market_value_in_eur"] * np.exp(
        data_out["y_growth_pred"]
    )

    # 7) 存 regression_outputs（先不含 SHAP，之後 shap_regression 會補）
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/regression_outputs.parquet"

    cols_to_save = [
        "player_id",
        "snapshot_date",
        "y_growth",
        "y_growth_pred",
        "market_value_in_eur",
        "mv_pred_1y",
    ]
    cols_to_save = [c for c in cols_to_save if c in data_out.columns]

    data_out[cols_to_save].to_parquet(out_path, index=False)
    print(f"Saved regression_outputs -> {out_path}")

    # 8) 存 metrics
    metrics = {
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test,
    }
    with open("data/processed/regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved regression_metrics.json")

    # 9) 存 model + feature columns，給 SHAP 用
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_regressor.joblib"
    dump(model, model_path)
    print(f"Saved XGBoost model -> {model_path}")

    feat_path = "models/xgb_features.json"
    with open(feat_path, "w") as f:
        json.dump(list(X_df.columns), f)
    print(f"Saved feature names -> {feat_path}")


if __name__ == "__main__":
    main()
