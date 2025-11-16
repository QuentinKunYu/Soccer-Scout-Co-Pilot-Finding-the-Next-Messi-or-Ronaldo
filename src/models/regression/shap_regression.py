import os
import json
import numpy as np
import pandas as pd

import shap
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --- 跟 train_regression 裡的 helper 保持一致 ---

def load_player_snapshot(path: str = "data/processed/player_snapshot.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not np.issubdtype(df["snapshot_date"].dtype, np.datetime64):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame):
    target_col = "y_growth"

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

    cat_cols = [
        "position",
        "sub_position",
        "league_name",
        "league_country",
        "club_name",
        "country_of_citizenship",
    ]

    bin_cols = [
        "league_is_major",
        "is_left_footed",
        "is_right_footed",
    ]

    data = df.dropna(subset=[target_col]).copy()

    if not np.issubdtype(data["snapshot_date"].dtype, np.datetime64):
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")

    X_num = data[[c for c in num_cols if c in data.columns]].copy()
    X_bin = data[[c for c in bin_cols if c in data.columns]].copy()

    cat_cols_present = [c for c in cat_cols if c in data.columns]
    if cat_cols_present:
        X_cat = pd.get_dummies(
            data[cat_cols_present],
            dummy_na=True,
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=data.index)

    X_df = pd.concat([X_num, X_bin, X_cat], axis=1)
    y = data[target_col].values

    print(f"[SHAP] Feature matrix shape: {X_df.shape}, target length: {len(y)}")

    return X_df, y, data


def split_by_time(data: pd.DataFrame, X_df: pd.DataFrame, y: np.ndarray):
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

    print("[SHAP] Test size:", X_test.shape)

    masks = {
        "train": train_mask,
        "valid": valid_mask,
        "test": test_mask,
    }
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks


def main(top_k: int = 5):
    # 1) 讀 snapshot & feature matrix
    df = load_player_snapshot()
    X_df, y, data = build_feature_matrix(df)
    (_, _), (_, _), (X_test, y_test), masks = split_by_time(data, X_df, y)

    test_mask = masks["test"]
    data_test = data[test_mask].copy()

    # 2) 載入 model & feature names
    model = load("models/xgb_regressor.joblib")
    with open("models/xgb_features.json", "r") as f:
        feat_names_trained = json.load(f)

    # 保證 X_df 欄位順序跟訓練時一樣（有缺的補 0）
    for col in feat_names_trained:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[feat_names_trained]

    # 只取 test set
    X_test = X_df[test_mask]

    # 3) 計算 SHAP
    print("[SHAP] Computing SHAP values for test set ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # shape: (n_test, n_features)

    shap_values = np.array(shap_values)
    feature_names = X_test.columns.tolist()

    # 4) 對每個球員取 top K feature，轉成 JSON string
    top_features_json = []

    for i in range(X_test.shape[0]):
        row_shap = shap_values[i]
        # 依絕對值排序
        idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
        row_list = []
        for j in idx_sorted:
            row_list.append(
                {
                    "feature": feature_names[j],
                    "shap_value": float(row_shap[j]),
                }
            )
        top_features_json.append(json.dumps(row_list))

    data_test = data_test.reset_index(drop=True)
    data_test["reg_shap_top_features"] = top_features_json

    # 5) merge 回 regression_outputs.parquet
    reg = pd.read_parquet("data/processed/regression_outputs.parquet")

    # 確保日期型態一致
    if not np.issubdtype(reg["snapshot_date"].dtype, np.datetime64):
        reg["snapshot_date"] = pd.to_datetime(reg["snapshot_date"], errors="coerce")

    merged = reg.merge(
        data_test[["player_id", "snapshot_date", "reg_shap_top_features"]],
        on=["player_id", "snapshot_date"],
        how="left",
    )

    out_path = "data/processed/regression_outputs.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"[SHAP] Updated regression_outputs with reg_shap_top_features -> {out_path}")


if __name__ == "__main__":
    main()
