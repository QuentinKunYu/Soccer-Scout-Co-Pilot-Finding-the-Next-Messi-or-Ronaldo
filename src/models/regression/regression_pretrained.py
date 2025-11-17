"""
Use pretrained XGBoost regression model to generate player market value growth
predictions (no retraining).

Usage (for non-technical users):

    python run_regression_pretrained.py

This will:
- Load the latest player_snapshot.parquet
- Load pretrained model + feature list
- Recreate feature matrix
- Predict y_growth and 1y-ahead market value
- Compute per-player SHAP top features
- Save outputs to:
    data/processed/regression_outputs.parquet
    data/processed/player_predictions.csv
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load
import shap
import argparse

# Import core utilities and constants from your training script
from .train_regression import (
    load_player_snapshot,
    build_feature_matrix,
    TARGET_YEAR_FOR_PRED,
    PLAYER_SNAPSHOT_PATH,
    REG_OUTPUT_PATH,
    REG_PREDICTIONS_CSV,
    MODEL_PATH,
    FEATURES_PATH,
)



def run_pretrained(
    snapshot_path: str = PLAYER_SNAPSHOT_PATH,
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    output_parquet: str = REG_OUTPUT_PATH,
    output_csv: str = REG_PREDICTIONS_CSV,
):
    """
    Run inference using a pretrained model only (no retraining).

    Assumes:
        - snapshot_path points to a player_snapshot.parquet file
        - model_path points to a trained XGBRegressor saved with joblib
        - features_path points to a list of feature names used during training
    """
    print("=== Using pretrained XGBoost regression model ===")

    # ----------------- Load data -----------------
    print(f"[inference] Loading snapshot from: {snapshot_path}")
    df = load_player_snapshot(snapshot_path)

    # Rebuild features (we don't *use* y here; it's just returned by the function)
    X_df, y_dummy, data = build_feature_matrix(df)
    years = data["snapshot_date"].dt.year
    print(f"[inference] Feature matrix shape: {X_df.shape}")

    # ----------------- Load pretrained model & feature names -----------------
    print(f"[inference] Loading pretrained model from: {model_path}")
    model = load(model_path)

    print(f"[inference] Loading feature list from: {features_path}")
    with open(features_path, "r") as f:
        trained_features = json.load(f)

    # Make sure columns match the training-time features
    #   - Missing cols -> fill with 0
    #   - Extra cols -> dropped
    X_aligned = X_df.reindex(columns=trained_features, fill_value=0.0)
    print(f"[inference] Aligned feature matrix shape: {X_aligned.shape}")

    # ----------------- Predict for all rows -----------------
    print("[inference] Predicting y_growth for all snapshots...")
    y_pred_all = model.predict(X_aligned)

    data_out = data.copy()
    data_out["y_growth_pred"] = y_pred_all
    data_out["mv_pred_1y"] = data_out["market_value_in_eur"] * np.exp(
        data_out["y_growth_pred"]
    )

    # ----------------- Filter to target prediction year -----------------
    print(f"[inference] Selecting snapshots in year {TARGET_YEAR_FOR_PRED}...")
    target_mask = years == TARGET_YEAR_FOR_PRED
    data_target = data_out[target_mask].copy()
    print(f"[inference] Target-year rows: {len(data_target)}")

    X_target = X_aligned.loc[data_target.index].copy()

    # ----------------- SHAP for target year -----------------
    print("[inference] Computing SHAP values for target year snapshots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    shap_values = np.array(shap_values)

    feature_names = X_target.columns.tolist()
    top_k = 5
    top_features_json = []

    for i in range(X_target.shape[0]):
        row_shap = shap_values[i]
        idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
        row_list = [
            {"feature": feature_names[j], "shap_value": float(row_shap[j])}
            for j in idx_sorted
        ]
        top_features_json.append(json.dumps(row_list))

    data_target = data_target.reset_index(drop=True)
    data_target["reg_shap_top_features"] = top_features_json

    # ----------------- Build output table -----------------
    output_cols = [
        "player_id",
        "snapshot_date",
        "y_growth_pred",
        "mv_pred_1y",
        "reg_shap_top_features",
    ]
    regression_outputs = data_target[output_cols].copy()

    # Ensure datetime and sorting
    if not pd.api.types.is_datetime64_any_dtype(regression_outputs["snapshot_date"]):
        regression_outputs["snapshot_date"] = pd.to_datetime(
            regression_outputs["snapshot_date"]
        )
    regression_outputs = regression_outputs.sort_values("player_id")

    # ----------------- Save outputs -----------------
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    regression_outputs.to_parquet(output_parquet, index=False)
    print(f"[inference] Saved regression_outputs -> {output_parquet}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    regression_outputs.to_csv(output_csv, index=False)
    print(f"[inference] Saved player predictions CSV -> {output_csv}")
    print(f"[inference] Output shape: {regression_outputs.shape}")
    print(f"[inference] Columns: {list(regression_outputs.columns)}")

    print("=== Done. Pretrained predictions ready. ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretrained XGBoost regression model to generate market value growth predictions."
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default=PLAYER_SNAPSHOT_PATH,
        help=f"Path to player_snapshot parquet (default: {PLAYER_SNAPSHOT_PATH})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to pretrained model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=FEATURES_PATH,
        help=f"Path to JSON with feature names (default: {FEATURES_PATH})",
    )
    parser.add_argument(
        "--out-parquet",
        type=str,
        default=REG_OUTPUT_PATH,
        help=f"Output parquet path (default: {REG_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=REG_PREDICTIONS_CSV,
        help=f"Output CSV path (default: {REG_PREDICTIONS_CSV})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretrained(
        snapshot_path=args.snapshot_path,
        model_path=args.model_path,
        features_path=args.features_path,
        output_parquet=args.out_parquet,
        output_csv=args.out_csv,
    )
