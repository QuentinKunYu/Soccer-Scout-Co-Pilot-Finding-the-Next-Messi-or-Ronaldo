"""
XGBoost Regression Model for Player Market Value Growth Prediction

This module trains an XGBoost regression model to predict player market value growth
(y_growth) based on player snapshots. It includes:
- Feature engineering from player snapshot data
- Time-based train/validation/test splitting
- Rolling validation for model evaluation
- SHAP value computation for model interpretability
- Prediction generation for target year snapshots

The model predicts 1-year market value growth and saves predictions along with
SHAP feature importance values for downstream analysis.
"""

import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump
import shap


# ----------------- Path Configuration -----------------

PLAYER_SNAPSHOT_PATH = "data/processed/player_snapshot.parquet"
REG_OUTPUT_PATH = "data/processed/regression_outputs.parquet"
REG_PREDICTIONS_CSV = "data/processed/player_predictions.csv"  # CSV output for predictions
REG_METRICS_PATH = "data/processed/regression_metrics.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_regressor.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "xgb_features.json")

# ----------------- Model Configuration -----------------
FINAL_TRAIN_END_YEAR = 2024      # Final year to include in training data
TARGET_YEAR_FOR_PRED = 2024      # Year of snapshots to use for prediction (predicts → 2025)



# ----------------- Data Loading & Feature Engineering -----------------

def load_player_snapshot(path: str = PLAYER_SNAPSHOT_PATH) -> pd.DataFrame:
    """
    Load player snapshot data from parquet file.
    
    Args:
        path: Path to the player snapshot parquet file
        
    Returns:
        DataFrame with player snapshot data, with snapshot_date converted to datetime
    """
    df = pd.read_parquet(path)
    if not np.issubdtype(df["snapshot_date"].dtype, np.datetime64):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Build feature matrix and target vector from player snapshot data.
    
    Constructs features from:
    - Numerical features: age, height, market value, performance metrics, etc.
    - Binary flags: league indicators, transfer flags, etc.
    - Categorical features: position, foot, country, club, league (one-hot encoded)
    
    Args:
        df: DataFrame containing player snapshot data
        
    Returns:
        Tuple of (X_df, y, data) where:
        - X_df: Feature matrix (DataFrame) with all engineered features
        - y: Target vector (numpy array) containing y_growth values
        - data: Original DataFrame aligned with X_df and y (rows with missing y_growth removed)
    """
    target_col = "y_growth"

    # Numerical features
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

    # Binary flags
    bin_cols = [
        "league_is_major",
        "is_top5_league",
        "has_recent_transfer",
        "moved_to_bigger_club_flag",
    ]

    # Categorical features (will be one-hot encoded)
    cat_cols = [
        "position",
        "sub_position",
        "foot",
        "country_of_citizenship",
        "current_club_name",
        "league_name",
        "league_country",
    ]

    # Keep only rows with valid target labels
    data = df.dropna(subset=[target_col]).copy()

    if not np.issubdtype(data["snapshot_date"].dtype, np.datetime64):
        data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")

    # Extract numerical and binary features
    X_num = data[[c for c in num_cols if c in data.columns]].copy()
    X_bin = data[[c for c in bin_cols if c in data.columns]].copy()

    # One-hot encode categorical features
    cat_present = [c for c in cat_cols if c in data.columns]
    if cat_present:
        X_cat = pd.get_dummies(
            data[cat_present],
            dummy_na=True,
            drop_first=True,
        )
    else:
        X_cat = pd.DataFrame(index=data.index)

    # Concatenate all feature types
    X_df = pd.concat([X_num, X_bin, X_cat], axis=1)
    y = data[target_col].values

    print(f"[build_feature_matrix] X shape = {X_df.shape}, y length = {len(y)}")
    return X_df, y, data


# ----------------- Time-based Splitting & Rolling Validation -----------------

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
    Split data into train/validation/test sets based on snapshot year.
    
    This function implements time-based splitting for the final model evaluation:
    - Training set: all snapshots with year <= train_end_year
    - Validation set: snapshots with year between valid_start_year and valid_end_year (inclusive)
    - Test set: all snapshots with year >= test_start_year
    
    Args:
        data: DataFrame with snapshot_date column
        X_df: Feature matrix aligned with data
        y: Target vector aligned with data
        train_end_year: Last year to include in training set
        valid_start_year: First year to include in validation set
        valid_end_year: Last year to include in validation set
        test_start_year: First year to include in test set
        
    Returns:
        Tuple of ((X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks) where:
        - Each (X, y) pair is the feature matrix and target for that split
        - masks is a dict with boolean masks for each split
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
    Perform rolling validation across multiple years.
    
    For each validation year, trains a model on all data before that year
    and evaluates on data from that year. This provides a time-series aware
    cross-validation approach.
    
    Args:
        X_df: Feature matrix
        y: Target vector
        data: DataFrame with snapshot_date column
        model_params: Dictionary of XGBoost hyperparameters
        val_years: List of years to use as validation folds
        
    Returns:
        List of dictionaries, each containing metrics (rmse, mae) for one fold
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

        print(f"[Fold {val_year}] RMSE={rmse:.4f} MAE={mae:.4f}")
        results.append({"year": val_year, "rmse": rmse, "mae": mae})

    if results:
        rv_df = pd.DataFrame(results)
        print("\n[Rolling Validation] summary:")
        print(rv_df.describe())
    else:
        print("[Rolling Validation] No valid folds (check year ranges).")

    return results


# ----------------- Evaluation Functions -----------------

def evaluate_regression(model, X, y, split_name: str):
    """
    Evaluate regression model performance on a dataset.
    
    Computes and prints RMSE and MAE for the given split.
    
    Args:
        model: Trained regression model with predict() method
        X: Feature matrix
        y: True target values
        split_name: Name of the data split (e.g., "train", "valid", "test")
        
    Returns:
        Tuple of (rmse, mae, y_pred) where:
        - rmse: Root Mean Squared Error
        - mae: Mean Absolute Error
        - y_pred: Predicted values
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y, y_pred))
    print(f"[{split_name}] RMSE={rmse:.4f}  MAE={mae:.4f}")
    return rmse, mae, y_pred


# ----------------- Main Training Pipeline -----------------

def main():
    """
    Main training pipeline for XGBoost regression model.
    
    This function:
    1. Loads player snapshot data
    2. Builds feature matrix and target vector
    3. Performs rolling validation for model evaluation
    4. Trains evaluation model on train/valid/test splits
    5. Trains final model on all data up to FINAL_TRAIN_END_YEAR
    6. Generates predictions for TARGET_YEAR_FOR_PRED snapshots
    7. Computes SHAP values for model interpretability
    8. Saves predictions, metrics, and model artifacts
    """
    #  Load player snapshot data
    df = load_player_snapshot()

    # Build feature matrix and target vector
    X_df, y, data = build_feature_matrix(df)
    years = data["snapshot_date"].dt.year
    unique_years = sorted(years.unique())
    print(f"[main] years in data: {unique_years[:5]} ... {unique_years[-5:]}")

    # XGBoost hyperparameters (can be tuned further)
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

    # Perform rolling validation (e.g., 2019-2023)
    candidate_val_years = [2019, 2020, 2021, 2022, 2023]
    val_years = [yy for yy in candidate_val_years if yy in unique_years]

    if len(val_years) >= 1:
        print(f"[main] Rolling validation years: {val_years}")
        rolling_validation_scores(X_df, y, data, model_params, val_years)
    else:
        print("[main] No overlapping years for rolling validation, skip.")

    # Split data into train/valid/test sets (for computing evaluation metrics)
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), masks = split_by_time(
        data,
        X_df,
        y,
        train_end_year=2020,
        valid_start_year=2021,
        valid_end_year=2022,
        test_start_year=2023,
    )

    # Train evaluation model (trained only up to 2020 for fair evaluation)
    model_eval = XGBRegressor(**model_params)
    model_eval.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True,
    )

    # Evaluate on train/valid/test splits (save metrics to JSON)
    _, _, _ = evaluate_regression(model_eval, X_train, y_train, "train")
    _, _, _ = evaluate_regression(model_eval, X_valid, y_valid, "valid")
    rmse_test, mae_test, _ = evaluate_regression(
        model_eval, X_test, y_test, "test"
    )

    # Train final model: use all data <= 2024 to predict 2025
    final_train_mask = years <= FINAL_TRAIN_END_YEAR
    X_train_final = X_df[final_train_mask]
    y_train_final = y[final_train_mask]

    print(f"[main] Final training set (<= {FINAL_TRAIN_END_YEAR}): {X_train_final.shape}")

    model = XGBRegressor(**model_params)
    model.fit(X_train_final, y_train_final)

    # Generate predictions for all rows (useful for debugging and overall results)
    y_pred_all = model.predict(X_df)
    data_out = data.copy()
    data_out["y_growth_pred"] = y_pred_all
    data_out["mv_pred_1y"] = data_out["market_value_in_eur"] * np.exp(
        data_out["y_growth_pred"]
    )

    # Filter to TARGET_YEAR_FOR_PRED (2024) snapshots → predict 2024→2025 growth
    print(f"\n[train_regression] Filtering to snapshots in {TARGET_YEAR_FOR_PRED}...")
    target_mask = years == TARGET_YEAR_FOR_PRED
    data_target = data_out[target_mask].copy()
    print(f"[train_regression] Filtered to {len(data_target)} rows in {TARGET_YEAR_FOR_PRED}")

    # Use corresponding X with same index
    X_target = X_df.loc[data_target.index].copy()
    X_target = X_target[X_df.columns]  # Ensure column order consistency

    # Compute SHAP values (only for target year snapshots)
    print("[train_regression] Computing SHAP values for target year snapshots...")
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

    # Build output table: one row = one player's 2024 snapshot 1-year prediction
    output_cols = [
        "player_id",
        "snapshot_date",
        "y_growth_pred",
        "mv_pred_1y",
        "reg_shap_top_features",
    ]
    regression_outputs = data_target[output_cols].copy()

    # Ensure snapshot_date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(regression_outputs["snapshot_date"]):
        regression_outputs["snapshot_date"] = pd.to_datetime(regression_outputs["snapshot_date"])

    # Sort by player_id
    regression_outputs = regression_outputs.sort_values("player_id")

    # Save parquet and CSV files (for team use)
    os.makedirs(os.path.dirname(REG_OUTPUT_PATH), exist_ok=True)
    regression_outputs.to_parquet(REG_OUTPUT_PATH, index=False)
    print(f"[train_regression] Saved regression_outputs -> {REG_OUTPUT_PATH}")

    os.makedirs(os.path.dirname(REG_PREDICTIONS_CSV), exist_ok=True)
    regression_outputs.to_csv(REG_PREDICTIONS_CSV, index=False)
    print(f"[train_regression] Saved player predictions CSV -> {REG_PREDICTIONS_CSV}")
    print(f"[train_regression] Output shape: {regression_outputs.shape}")
    print(f"[train_regression] Columns: {list(regression_outputs.columns)}")

    # Save evaluation metrics (from eval model)
    metrics = {
        "rmse_test": rmse_test,
        "mae_test": mae_test,
    }
    os.makedirs(os.path.dirname(REG_METRICS_PATH), exist_ok=True)
    with open(REG_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_regression] Saved regression_metrics -> {REG_METRICS_PATH}")

    # Save final model and feature names (for SHAP and future use)
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"[train_regression] Saved XGBoost model -> {MODEL_PATH}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X_df.columns), f)
    print(f"[train_regression] Saved feature names -> {FEATURES_PATH}")



if __name__ == "__main__":
    main()
