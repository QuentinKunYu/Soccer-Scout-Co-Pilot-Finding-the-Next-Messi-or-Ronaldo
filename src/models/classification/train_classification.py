"""
LightGBM Classification Model for Player Breakout Prediction

This module trains a LightGBM classification model to predict player breakout
probability based on player snapshots. It includes:
- Feature engineering from player snapshot data
- Time-based train/validation/test splitting
- SHAP value computation for model interpretability
- Prediction generation for target year snapshots

The model predicts breakout probability and saves predictions along with
SHAP feature importance values for downstream analysis.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import lightgbm as lgb
from joblib import dump
import shap


# ----------------- Path Configuration -----------------

PLAYER_SNAPSHOT_PATH = "data/processed/player_snapshot.parquet"
CLF_OUTPUT_PATH = "data/processed/classification_outputs.parquet"
CLF_PREDICTIONS_CSV = "data/processed/breakout_predictions.csv"
CLF_METRICS_PATH = "data/processed/classification_metrics.json"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgb_classifier.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "lgb_features.json")

# ----------------- Model Configuration -----------------
FINAL_TRAIN_END_YEAR = 2021      # Final year to include in training data
VALID_END_YEAR = 2023            # End year for validation
TARGET_YEAR_FOR_PRED = 2024      # Year of snapshots to use for prediction (predicts → 2025)


# ----------------- Data Loading Functions -----------------

def load_data():
    """Load all required data files"""
    from pathlib import Path
    
    data_path = Path("data")
    
    print("[train_classification] Loading data files...")
    
    super_df = pd.read_parquet(PLAYER_SNAPSHOT_PATH)
    players = pd.read_csv(data_path / "players.csv")
    valuations = pd.read_csv(data_path / "player_valuations.csv")
    appearances = pd.read_csv(data_path / "appearances.csv")
    competitions = pd.read_csv(data_path / "competitions.csv")
    clubs = pd.read_csv(data_path / "clubs.csv")
    transfers = pd.read_csv(data_path / "transfers.csv")
    
    print(f"[train_classification] Loaded super_df: {super_df.shape}")
    return super_df, players, valuations, appearances, competitions, clubs, transfers


# ----------------- Preprocessing Functions -----------------

def preprocess_data(super_df, clubs):
    """Initial data preprocessing"""
    print("[train_classification] Preprocessing data...")
    
    # Drop records without sub_position
    super_df = super_df.dropna(subset=["sub_position"])
    
    # Convert dates
    super_df['snapshot_date'] = pd.to_datetime(super_df['snapshot_date'])
    super_df['snapshot_year'] = super_df['snapshot_date'].dt.year
    
    # Merge club information
    super_df = super_df.merge(
        clubs[[
            "club_id", "domestic_competition_id", "total_market_value", 
            "squad_size", "average_age", "foreigners_number", 
            "national_team_players", "net_transfer_record"
        ]],
        left_on="current_club_id",
        right_on="club_id",
        how="left"
    )
    
    return super_df


# ----------------- Feature Engineering Functions -----------------

def create_recent_move_up_flag(super_df, transfers, competitions, clubs):
    """Create feature for recent transfers to bigger clubs"""
    print("[train_classification] Creating recent_move_up_flag...")
    
    # Ensure dates are datetime
    super_df["snapshot_date"] = pd.to_datetime(super_df["snapshot_date"], errors="coerce")
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    
    # Build club strength
    comp_small = competitions[["competition_id", "sub_type", "is_major_national_league"]].copy()
    
    clubs_league = clubs.merge(
        comp_small,
        left_on="domestic_competition_id",
        right_on="competition_id",
        how="left"
    )
    
    sub_type_score_map = {
        "first_tier": 2.0,
        "second_tier": 1.5,
        "domestic_cup": 1.2,
        "domestic_super_cup": 1.1,
    }
    clubs_league["sub_type_score"] = clubs_league["sub_type"].map(sub_type_score_map).fillna(1.0)
    
    clubs_league["league_base_score"] = np.where(
        clubs_league["is_major_national_league"] == True,
        3.0,
        clubs_league["sub_type_score"]
    )
    
    # Normalize extras
    def safe_norm(s):
        s = s.astype(float)
        return (s - s.min()) / (s.max() - s.min() + 1e-6)
    
    clubs_league["ntp_norm"] = safe_norm(clubs_league.get("national_team_players", 0))
    clubs_league["seats_norm"] = safe_norm(clubs_league.get("stadium_seats", 0))
    clubs_league["foreigners_norm"] = safe_norm(clubs_league.get("foreigners_percentage", 0))
    
    clubs_league["club_strength"] = (
        clubs_league["league_base_score"]
        + 0.5 * clubs_league["ntp_norm"]
        + 0.3 * clubs_league["seats_norm"]
        + 0.2 * clubs_league["foreigners_norm"]
    )
    
    club_strength = clubs_league[["club_id", "club_strength"]].copy()
    
    # Mark transfers as move up
    t = transfers.merge(
        club_strength.rename(columns={"club_id": "from_club_id", "club_strength": "from_strength"}),
        on="from_club_id", how="left"
    ).merge(
        club_strength.rename(columns={"club_id": "to_club_id", "club_strength": "to_strength"}),
        on="to_club_id", how="left"
    )
    
    t["from_strength"] = t["from_strength"].fillna(0)
    t["to_strength"] = t["to_strength"].fillna(0)
    t["move_up"] = (t["to_strength"] > t["from_strength"]).astype(int)
    
    # Get recent move up before snapshot
    tmp = super_df.merge(
        t[["player_id", "transfer_date", "move_up"]],
        on="player_id", how="left"
    )
    
    tmp = tmp[tmp["transfer_date"] < tmp["snapshot_date"]]
    tmp = tmp.sort_values(["player_id", "snapshot_date", "transfer_date"])
    
    recent_up = (
        tmp.groupby(["player_id", "snapshot_date"])
        .agg(last_move_up=("move_up", "last"))
        .reset_index()
    )
    
    recent_up["recent_move_up_flag"] = recent_up["last_move_up"].fillna(0).astype(int)
    
    # Merge back
    if "recent_move_up_flag" in super_df.columns:
        super_df = super_df.drop(columns=["recent_move_up_flag"])
    
    super_df = super_df.merge(
        recent_up[["player_id", "snapshot_date", "recent_move_up_flag"]],
        on=["player_id", "snapshot_date"],
        how="left"
    )
    
    super_df["recent_move_up_flag"] = super_df["recent_move_up_flag"].fillna(0).astype(int)
    
    return super_df


def create_discipline_features(super_df, appearances):
    """Create yellow/red card features"""
    print("[train_classification] Creating discipline features...")
    
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["snapshot_year"] = appearances["date"].dt.year
    
    discipline_stats = appearances.groupby(
        ["player_id", "snapshot_year"]
    ).agg(
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
        total_minutes=("minutes_played", "sum")
    ).reset_index()
    
    discipline_stats["total_minutes"] = discipline_stats["total_minutes"].replace(0, np.nan)
    
    discipline_stats["yellow_cards_per_90"] = (
        discipline_stats["total_yellow_cards"] / discipline_stats["total_minutes"] * 90
    )
    
    discipline_stats["red_cards_per_90"] = (
        discipline_stats["total_red_cards"] / discipline_stats["total_minutes"] * 90
    )
    
    discipline_stats = discipline_stats.fillna(0)
    
    super_df = super_df.merge(
        discipline_stats[["player_id", "snapshot_year", "yellow_cards_per_90", "red_cards_per_90"]],
        on=["player_id", "snapshot_year"],
        how="left"
    ).fillna(0)
    
    return super_df


def create_features(df):
    """Create all model features"""
    print("[train_classification] Creating model features...")
    
    def pct_rank(x):
        return x.rank(pct=True).fillna(0)
    
    # Time features
    df["minutes_per_game"] = df["minutes_total"] / df["games_played"]
    df["minutes_per_game"] = df["minutes_per_game"].fillna(0)
    
    df["time_pct"] = df.groupby(["sub_position", "season_year"])["minutes_per_game"].transform(pct_rank)
    
    # Performance features
    df["gi_per_90"] = df["goals_per_90"] + df["assists_per_90"]
    
    df["discipline_penalty"] = (
        df["yellow_cards_per_90"] * 0.3 +
        df["red_cards_per_90"] * 1.5
    )
    
    df["perf_base"] = df["gi_per_90"] - df["discipline_penalty"]
    df["perf_pct"] = df.groupby(["sub_position", "snapshot_year"])["perf_base"].transform(pct_rank)
    
    # Performance momentum
    df["perf_base_shift"] = df.groupby("player_id")["perf_base"].shift(1)
    df["perf_momentum"] = (df["perf_base"] - df["perf_base_shift"]).fillna(0)
    df["perf_momentum_pct"] = df.groupby(["sub_position", "snapshot_year"])["perf_momentum"].transform(pct_rank)
    
    # Market value features
    df["mv_pct"] = df.groupby(["sub_position", "snapshot_year"])["market_value_in_eur"].transform(pct_rank)
    
    df["market_value_shift"] = df.groupby("player_id")["market_value_in_eur"].shift(1)
    df["mv_momentum"] = (df["market_value_in_eur"] - df["market_value_shift"]).fillna(0)
    df["mv_momentum_pct"] = df.groupby(["sub_position", "snapshot_year"])["mv_momentum"].transform(pct_rank)
    
    # League features
    df["league_pct"] = (
        df.groupby("snapshot_year")["league_level"]
        .transform(lambda x: (-x).rank(pct=True))
    )
    
    return df


def create_breakout_label(df):
    """Create breakout label using position-specific weights"""
    print("[train_classification] Creating breakout labels...")
    
    # Position mapping
    position_map = {
        "Centre-Forward": "FW",
        "Second Striker": "FW",
        "Left Winger": "WING",
        "Right Winger": "WING",
        "Attacking Midfield": "AM",
        "Central Midfield": "CM",
        "Defensive Midfield": "DM",
        "Left Midfield": "LM",
        "Right Midfield": "RM",
        "Left-Back": "LB",
        "Right-Back": "RB",
        "Centre-Back": "CB",
        "Goalkeeper": "GK"
    }
    
    df["position_group"] = df["sub_position"].map(position_map)
    
    # Position-specific weights
    weights_FW = {
        "mv_pct": 0.30, "perf_pct": 0.35, "time_pct": 0.12,
        "league_pct": 0.08, "mv_momentum_pct": 0.07,
        "perf_momentum_pct": 0.03, "recent_move_up_flag": 0.05,
    }
    
    weights_WING = {
        "mv_pct": 0.26, "perf_pct": 0.30, "time_pct": 0.14,
        "league_pct": 0.08, "mv_momentum_pct": 0.10,
        "perf_momentum_pct": 0.07, "recent_move_up_flag": 0.05,
    }
    
    weights_AM = {
        "mv_pct": 0.25, "perf_pct": 0.33, "time_pct": 0.18,
        "league_pct": 0.07, "mv_momentum_pct": 0.10,
        "perf_momentum_pct": 0.02, "recent_move_up_flag": 0.05,
    }
    
    weights_CM_DM = {
        "mv_pct": 0.18, "perf_pct": 0.22, "time_pct": 0.32,
        "league_pct": 0.15, "mv_momentum_pct": 0.05,
        "perf_momentum_pct": 0.03, "recent_move_up_flag": 0.05,
    }
    
    weights_FB = {
        "mv_pct": 0.18, "perf_pct": 0.20, "time_pct": 0.32,
        "league_pct": 0.18, "mv_momentum_pct": 0.07,
        "perf_momentum_pct": 0.05, "recent_move_up_flag": 0.05,
    }
    
    weights_CB = {
        "mv_pct": 0.20, "perf_pct": 0.10, "time_pct": 0.30,
        "league_pct": 0.25, "mv_momentum_pct": 0.08,
        "perf_momentum_pct": 0.02, "recent_move_up_flag": 0.05,
    }
    
    weights_GK = {
        "mv_pct": 0.12, "perf_pct": 0.00, "time_pct": 0.45,
        "league_pct": 0.20, "mv_momentum_pct": 0.08,
        "perf_momentum_pct": 0.05, "recent_move_up_flag": 0.10,
    }
    
    weight_dict = {
        "FW": weights_FW, "WING": weights_WING, "AM": weights_AM,
        "CM": weights_CM_DM, "DM": weights_CM_DM,
        "LM": weights_CM_DM, "RM": weights_CM_DM,
        "LB": weights_FB, "RB": weights_FB,
        "CB": weights_CB, "GK": weights_GK
    }
    
    def compute_position_score(row):
        weights = weight_dict.get(row["position_group"], weights_CM_DM)
        return sum(row[f] * w for f, w in weights.items())
    
    df["breakout_score_pos"] = df.apply(compute_position_score, axis=1)
    
    # Age factors
    def position_age_factor(age, pos):
        if pos == "GK":
            if age <= 22: return 0.70
            if age <= 25: return 0.85
            if age <= 29: return 1.00
            if age <= 33: return 1.05
            return 0.80
        
        if pos == "CB":
            if age <= 21: return 0.75
            if age <= 24: return 0.90
            if age <= 28: return 1.05
            if age <= 31: return 1.00
            return 0.85
        
        if pos in ["LB", "RB"]:
            if age <= 20: return 0.85
            if age <= 23: return 1.00
            if age <= 26: return 1.05
            if age <= 29: return 1.00
            return 0.90
        
        if pos in ["CM", "DM"]:
            if age <= 20: return 1.05
            if age <= 22: return 1.00
            if age <= 26: return 1.05
            if age <= 29: return 0.95
            return 0.85
        
        if pos == "AM":
            if age <= 20: return 1.15
            if age <= 22: return 1.05
            if age <= 25: return 1.00
            if age <= 28: return 0.90
            return 0.80
        
        if pos == "WING":
            if age <= 19: return 1.30
            if age <= 21: return 1.20
            if age <= 23: return 1.10
            if age <= 26: return 0.95
            return 0.80
        
        if pos == "FW":
            if age <= 19: return 1.25
            if age <= 21: return 1.15
            if age <= 23: return 1.05
            if age <= 26: return 0.95
            return 0.80
        
        return 1.0
    
    df["age_factor"] = df.apply(
        lambda row: position_age_factor(row["age"], row["position_group"]),
        axis=1
    )
    
    df["breakout_score"] = df["breakout_score_pos"] * df["age_factor"]
    
    # Calculate delta
    df = df.sort_values(["player_id", "snapshot_date"])
    df["breakout_score_now"] = df["breakout_score"]
    df["breakout_score_prev"] = df.groupby("player_id")["breakout_score_now"].shift(1)
    df["delta_breakout"] = df["breakout_score_now"] - df["breakout_score_prev"]
    
    # Create label by position/year quantile
    df["breakout_label"] = 0
    quantile_cut = 0.80
    
    groups = df.groupby(["position_group", "snapshot_year"])
    
    for (pos, yr), sub_df in groups:
        if len(sub_df) < 30:
            continue
        
        threshold = sub_df["delta_breakout"].quantile(quantile_cut)
        idx = sub_df.index[sub_df["delta_breakout"] >= threshold]
        df.loc[idx, "breakout_label"] = 1
    
    print(f"[train_classification] Breakout label distribution:\n{df['breakout_label'].value_counts()}")
    
    return df


# ----------------- Model Preparation & Training -----------------

def prepare_model_data(df):
    """Prepare data for modeling"""
    print("[train_classification] Preparing model data...")
    
    # Define features
    drop_cols = [
        "player_id", "breakout_label", "breakout_score_pos", "age_factor",
        "snapshot_date", "breakout_score", "breakout_score_now",
        "breakout_score_prev", "delta_breakout"
    ]
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"[train_classification] Features used: {feature_cols}")
    
    # Encode categorical variables
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    print(f"[train_classification] Categorical columns: {cat_cols}")
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df, feature_cols


def split_data(df, feature_cols):
    """Split data by time periods"""
    print("[train_classification] Splitting data...")
    
    target = "breakout_label"
    
    train_df = df[df["snapshot_date"].dt.year <= FINAL_TRAIN_END_YEAR]
    valid_df = df[(df["snapshot_date"].dt.year > FINAL_TRAIN_END_YEAR) & 
                  (df["snapshot_date"].dt.year <= VALID_END_YEAR)]
    test_df = df[df["snapshot_date"].dt.year == TARGET_YEAR_FOR_PRED]
    
    X_train, y_train = train_df[feature_cols], train_df[target]
    X_valid, y_valid = valid_df[feature_cols], valid_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]
    
    print(f"[train_classification] Train shape: {X_train.shape}")
    print(f"[train_classification] Valid shape: {X_valid.shape}")
    print(f"[train_classification] Test shape: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_model(X_train, y_train, X_valid, y_valid):
    """Train LightGBM model"""
    print("[train_classification] Training model...")
    
    # LightGBM parameters
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "max_depth": -1,
        "verbosity": -1,
        "seed": 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    
    print(f"[train_classification] Best iteration: {model.best_iteration}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n[train_classification] Evaluating model...")
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    metrics = {
        "auc": float(roc_auc_score(y_test, y_pred_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    print("\n===== Test Metrics =====")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics


# ----------------- Prediction & SHAP Functions -----------------

def predict_target_year(model, df, feature_cols):
    """Predict breakout for target year"""
    print(f"\n[train_classification] Predicting for year {TARGET_YEAR_FOR_PRED}...")
    
    df_target = df[df["snapshot_year"] == TARGET_YEAR_FOR_PRED].copy()
    
    df_target_latest = (
        df_target.sort_values(["player_id", "snapshot_date"])
        .groupby("player_id")
        .tail(1)
        .reset_index(drop=True)
    )
    
    print(f"[train_classification] Target year unique players: {df_target_latest['player_id'].nunique()}")
    
    X_target = df_target_latest[feature_cols]
    df_target_latest["breakout_prob"] = model.predict(X_target)
    
    return df_target_latest, X_target


def compute_shap_values(model, X_target, top_k=5):
    """Compute SHAP values for interpretability"""
    print("[train_classification] Computing SHAP values...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_target)
    
    # For binary classification, take class 1 (breakout=1)
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[1])
    else:
        shap_values = np.array(shap_values)
    
    feature_names = X_target.columns.tolist()
    
    # Get top-k SHAP features for each player
    top_features_json = []
    
    for i in range(X_target.shape[0]):
        row_shap = shap_values[i]
        idx_sorted = np.argsort(-np.abs(row_shap))[:top_k]
        
        row_list = [
            {
                "feature": feature_names[j],
                "shap_value": float(row_shap[j])
            }
            for j in idx_sorted
        ]
        
        top_features_json.append(json.dumps(row_list))
    
    return top_features_json


def save_predictions(pred_df, shap_features):
    """Save final predictions with SHAP values"""
    print("\n[train_classification] Saving predictions...")
    
    pred_df = pred_df.reset_index(drop=True)
    pred_df["clf_shap_top_features"] = shap_features
    
    # Prepare output
    output_cols = [
        "player_id",
        "snapshot_date",
        "breakout_prob",
        "clf_shap_top_features"
    ]
    
    classification_outputs = pred_df[output_cols].copy()
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(classification_outputs["snapshot_date"]):
        classification_outputs["snapshot_date"] = pd.to_datetime(classification_outputs["snapshot_date"])
    
    classification_outputs = classification_outputs.sort_values("player_id")
    
    # Save to files
    os.makedirs(os.path.dirname(CLF_OUTPUT_PATH), exist_ok=True)
    classification_outputs.to_parquet(CLF_OUTPUT_PATH, index=False)
    print(f"[train_classification] Saved → {CLF_OUTPUT_PATH}")
    
    os.makedirs(os.path.dirname(CLF_PREDICTIONS_CSV), exist_ok=True)
    classification_outputs.to_csv(CLF_PREDICTIONS_CSV, index=False)
    print(f"[train_classification] Saved → {CLF_PREDICTIONS_CSV}")
    print(f"[train_classification] Output shape: {classification_outputs.shape}")
    
    return classification_outputs


# ----------------- Main Training Pipeline -----------------

def main():
    """Main training pipeline for LightGBM classification model"""
    print("="*60)
    print("Breakout Player Classification Model Training")
    print("="*60)
    
    # Load data
    super_df, players, valuations, appearances, competitions, clubs, transfers = load_data()
    
    # Preprocess
    super_df = preprocess_data(super_df, clubs)
    
    # Feature engineering
    super_df = create_recent_move_up_flag(super_df, transfers, competitions, clubs)
    super_df = create_discipline_features(super_df, appearances)
    super_df = create_features(super_df)
    super_df = create_breakout_label(super_df)
    
    # Prepare for modeling
    super_df, feature_cols = prepare_model_data(super_df)
    
    # Split data
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(super_df, feature_cols)
    
    # Train model
    model = train_model(X_train, y_train, X_valid, y_valid)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Predict target year
    pred_df, X_target = predict_target_year(model, super_df, feature_cols)
    
    # Compute SHAP values
    shap_features = compute_shap_values(model, X_target)
    
    # Save predictions
    predictions = save_predictions(pred_df, shap_features)
    
    # Save metrics
    os.makedirs(os.path.dirname(CLF_METRICS_PATH), exist_ok=True)
    with open(CLF_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train_classification] Saved metrics → {CLF_METRICS_PATH}")
    
    # Save model and feature names
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"[train_classification] Saved model → {MODEL_PATH}")
    
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)
    print(f"[train_classification] Saved features → {FEATURES_PATH}")
    
    print("\n" + "="*60)
    print("Classification pipeline completed successfully!")
    print("="*60)
    
    return model, predictions, metrics


if __name__ == "__main__":
    main()