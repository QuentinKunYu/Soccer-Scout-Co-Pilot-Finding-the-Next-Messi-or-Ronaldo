import os
import json
import numpy as np
import pandas as pd

import shap
from joblib import load


PLAYER_SNAPSHOT_PATH = "data/processed/player_snapshot.parquet"
REG_OUTPUT_PATH = "data/processed/regression_outputs.parquet"
MODEL_PATH = "models/xgb_regressor.joblib"
FEATURES_PATH = "models/xgb_features.json"


def load_player_snapshot(path: str = PLAYER_SNAPSHOT_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not np.issubdtype(df["snapshot_date"].dtype, np.datetime64):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame):
    # 跟 train_regression 的 feature 定義保持一致
    target_col = "y_growth"

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

    bin_cols = [
        "league_is_major",
        "is_top5_league",
        "has_recent_transfer",
        "moved_to_bigger_club_flag",
    ]

    cat_cols = [
        "position",
        "sub_position",
        "foot",
        "country_of_citizenship",
        "current_club_name",
        "league_name",
        "league_country",
    ]

    data = df.dropna(subset=[target_col]).copy()

    if not np.issubdtype(data["snapshot_date"].dtype, np.datetime64):
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")

    X_num = data[[c for c in num_cols if c in data.columns]].copy()
    X_bin = data[[c for c in bin_cols if c in data.columns]].copy()

    cat_present = [c for c in cat_cols if c in data.columns]
    if cat_present:
        X_cat = pd.get_dummies(
            data[cat_present],
            dummy_na=True,
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=data.index)

    X_df = pd.concat([X_num, X_bin, X_cat], axis=1)
    y = data[target_col].values

    print(f"[shap_regression] X shape = {X_df.shape}, y length = {len(y)}")
    return X_df, y, data


def split_by_time(data: pd.DataFrame, X_df: pd.DataFrame, y: np.ndarray):
    years = data["snapshot_date"].dt.year

    train_mask = years <= 2020
    valid_mask = (years > 2020) & (years <= 2022)
    test_mask = years >= 2023

    X_train = X_df[train_mask]
    y_train = y[train_mask]
    X_valid = X_df[valid_mask]
    y_valid = y[valid_mask]
    X_test = X_df[test_mask]
    y_test = y[test_mask]

    print(f"[shap_regression] test size = {X_test.shape}")

    masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks


def main(top_k: int = 5):
    # 1) 讀 snapshot & feature matrix
    df = load_player_snapshot()
    X_df, y, data = build_feature_matrix(df)
    (_, _), (_, _), (X_test, y_test), masks = split_by_time(data, X_df, y)

    test_mask = masks["test"]
    data_test = data[test_mask].copy()

    # 2) 載入 model & feature 列表（確保欄位順序匹配）
    model = load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feat_names_trained = json.load(f)

    # 保證 X_df 跟訓練時欄位對齊（缺的補 0）
    for col in feat_names_trained:
        if col not in X_df.columns:
            X_df[col] = 0.0
    X_df = X_df[feat_names_trained]

    X_test = X_df[test_mask]

    # 3) 算 SHAP values（只對 test set）
    print("[shap_regression] Computing SHAP values on test set ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # (n_test, n_features)

    shap_values = np.array(shap_values)
    feature_names = X_test.columns.tolist()

    # 4) 對每個球員取 top K features，轉成 JSON string
    top_features_json = []

    for i in range(X_test.shape[0]):
        row_shap = shap_values[i]
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
    reg = pd.read_parquet(REG_OUTPUT_PATH)
    if not np.issubdtype(reg["snapshot_date"].dtype, np.datetime64):
        reg["snapshot_date"] = pd.to_datetime(reg["snapshot_date"], errors="coerce")

    merged = reg.merge(
        data_test[["player_id", "snapshot_date", "reg_shap_top_features"]],
        on=["player_id", "snapshot_date"],
        how="left",
    )

    merged.to_parquet(REG_OUTPUT_PATH, index=False)
    print(
        "[shap_regression] Updated regression_outputs with reg_shap_top_features "
        f"-> {REG_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
