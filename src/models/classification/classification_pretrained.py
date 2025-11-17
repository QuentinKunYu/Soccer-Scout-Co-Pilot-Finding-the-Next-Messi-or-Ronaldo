"""
Pretrained LightGBM Classification Model for Player Breakout Prediction

This module provides inference functionality for a pretrained LightGBM classification model
that predicts the probability of a player experiencing a "breakout" season.

A breakout is defined as a significant improvement in a player's performance metrics and
market value trajectory relative to their position and age group. The model uses position-specific
weighting schemes and age factors to account for different development patterns across positions.

The model predicts:
    - breakout_prob: Probability (0-1) that a player will experience a breakout
    - clf_shap_top_features: Top 5 most important features for each prediction (SHAP values)

The model uses the same feature engineering pipeline as the training script, including:
    - Performance metrics (goals, assists, minutes per 90)
    - Market value momentum and percentiles
    - Transfer history (recent moves to bigger clubs)
    - Discipline features (yellow/red cards)
    - Position-specific breakout scoring

Input:
    - data/processed/player_snapshot.parquet: Player snapshot dataset
    - models/lgb_classifier.joblib: Pretrained LightGBM model
    - models/lgb_features.json: List of feature names used during training

Output:
    - data/processed/classification_outputs.parquet: Predictions with SHAP features
    - data/processed/breakout_predictions.csv: CSV version of predictions

Usage:
    python -m src.models.classification.classification_pretrained
    
    Or import and use programmatically:
    from src.models.classification.classification_pretrained import run_pretrained
    run_pretrained(
        snapshot_path="data/processed/player_snapshot.parquet",
        model_path="models/lgb_classifier.joblib",
        features_path="models/lgb_features.json"
    )
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load
import shap
import argparse

# Import core utilities and constants from training script
from .train_classification import (
    load_data,
    preprocess_data,
    create_recent_move_up_flag,
    create_discipline_features,
    create_features,
    create_breakout_label,
    prepare_model_data,
    TARGET_YEAR_FOR_PRED,
    PLAYER_SNAPSHOT_PATH,
    CLF_OUTPUT_PATH,
    CLF_PREDICTIONS_CSV,
    MODEL_PATH,
    FEATURES_PATH,
)


def run_pretrained(
    snapshot_path: str = PLAYER_SNAPSHOT_PATH,
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    output_parquet: str = CLF_OUTPUT_PATH,
    output_csv: str = CLF_PREDICTIONS_CSV,
):
    """
    Run inference using a pretrained model only (no retraining).

    Assumes:
        - snapshot_path points to a player_snapshot.parquet file
        - model_path points to a trained LightGBM classifier saved with joblib
        - features_path points to a list of feature names used during training
    """
    print("=== Using pretrained LightGBM classification model ===")

    # ----------------- Load data -----------------
    print(f"[inference] Loading snapshot from: {snapshot_path}")
    super_df, players, valuations, appearances, competitions, clubs, transfers = load_data()

    # ----------------- Feature engineering pipeline -----------------
    print("[inference] Running feature engineering pipeline...")
    super_df = preprocess_data(super_df, clubs)
    super_df = create_recent_move_up_flag(super_df, transfers, competitions, clubs)
    super_df = create_discipline_features(super_df, appearances)
    super_df = create_features(super_df)
    super_df = create_breakout_label(super_df)
    
    # Prepare model data
    super_df, feature_cols_built = prepare_model_data(super_df)
    
    print(f"[inference] Feature matrix shape: {super_df.shape}")

    # ----------------- Load pretrained model & feature names -----------------
    print(f"[inference] Loading pretrained model from: {model_path}")
    model = load(model_path)

    print(f"[inference] Loading feature list from: {features_path}")
    with open(features_path, "r") as f:
        trained_features = json.load(f)

    # Make sure columns match the training-time features
    X_all = super_df[feature_cols_built].copy()
    X_aligned = X_all.reindex(columns=trained_features, fill_value=0.0)
    print(f"[inference] Aligned feature matrix shape: {X_aligned.shape}")

    # ----------------- Filter to target prediction year -----------------
    print(f"[inference] Selecting snapshots in year {TARGET_YEAR_FOR_PRED}...")
    target_mask = super_df["snapshot_year"] == TARGET_YEAR_FOR_PRED
    data_target = super_df[target_mask].copy()
    print(f"[inference] Target-year rows: {len(data_target)}")

    # Get latest snapshot per player
    data_target = (
        data_target.sort_values(["player_id", "snapshot_date"])
        .groupby("player_id")
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"[inference] Target-year unique players: {len(data_target)}")

    X_target = X_aligned.loc[data_target.index].copy()

    # ----------------- Predict for target year -----------------
    print("[inference] Predicting breakout probability for target year...")
    y_pred_prob = model.predict(X_target)
    
    data_target["breakout_prob"] = y_pred_prob

    # ----------------- SHAP for target year -----------------
    print("[inference] Computing SHAP values for target year snapshots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    
    # For binary classification, take class 1 (breakout=1)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])
    else:
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
    data_target["clf_shap_top_features"] = top_features_json

    # ----------------- Build output table -----------------
    output_cols = [
        "player_id",
        "snapshot_date",
        "breakout_prob",
        "clf_shap_top_features",
    ]
    classification_outputs = data_target[output_cols].copy()

    # Ensure datetime and sorting
    if not pd.api.types.is_datetime64_any_dtype(classification_outputs["snapshot_date"]):
        classification_outputs["snapshot_date"] = pd.to_datetime(
            classification_outputs["snapshot_date"]
        )
    classification_outputs = classification_outputs.sort_values("player_id")

    # ----------------- Save outputs -----------------
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    classification_outputs.to_parquet(output_parquet, index=False)
    print(f"[inference] Saved classification_outputs → {output_parquet}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    classification_outputs.to_csv(output_csv, index=False)
    print(f"[inference] Saved breakout predictions CSV → {output_csv}")
    print(f"[inference] Output shape: {classification_outputs.shape}")
    print(f"[inference] Columns: {list(classification_outputs.columns)}")

    print("=== Done. Pretrained predictions ready. ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretrained LightGBM classification model to generate breakout predictions."
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
        default=CLF_OUTPUT_PATH,
        help=f"Output parquet path (default: {CLF_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=CLF_PREDICTIONS_CSV,
        help=f"Output CSV path (default: {CLF_PREDICTIONS_CSV})",
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