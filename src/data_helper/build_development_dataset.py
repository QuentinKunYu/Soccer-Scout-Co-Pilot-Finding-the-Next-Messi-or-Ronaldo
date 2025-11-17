"""
Generate Development Outputs Script
====================================

This script implements the player development analysis pipeline, used to:
1. Build aging curves - analyze market value and performance by age and position
2. Calculate deviations between actual and expected player performance
3. Generate aging_score and development_tier indicators
4. Output the development_outputs.parquet file

Usage:
    python generate_development_outputs.py

Input files:
    - data/players.csv: Basic player information (including date of birth, position, etc.)
    - data/player_valuations.csv: Player market value history
    - data/appearances.csv: Player appearance records (including goals, assists, minutes played, etc.)

Output files:
    - data/processed/development_outputs.parquet: Dataset containing player development metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Constants
# ============================================================================
# Get the project root directory (this script is under src/data_helper/, so going up three levels gives the root)
# __file__: path to the current script file
# Path(__file__).resolve(): get absolute path of the script
# .parent.parent.parent: move up three levels to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# DATA_DIR: Directory containing raw data
DATA_DIR = PROJECT_ROOT / "data"
# PROCESSED_DIR: Output directory for processed data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# MILLION: Divisor to convert euros to million euros
MILLION = 1_000_000


# ============================================================================
# Helper Functions
# ============================================================================

def load_raw_data(data_dir: Path) -> dict:
    """
    Load all CSVs required for the development analysis.
    
    Parameters:
        data_dir (Path): The directory path containing raw CSV files
    
    Returns:
        dict: Dictionary with dataset names as keys and DataFrames as values
    
    Why call it load_raw_data:
        - "load" = load data
        - "raw" = raw, unprocessed
        - "data" = data
        This function loads the raw CSV files from disk.
    
    Example usage:
        dfs = load_raw_data(Path("../data"))
        players_df = dfs["players"]  # access the players DataFrame
    """
    # files: defines the mapping of datasets to file names
    files = {
        "players": "players.csv",
        "player_valuations": "player_valuations.csv",
        "appearances": "appearances.csv",
        "competitions": "competitions.csv",
        "games": "games.csv",
    }
    
    dfs = {}
    # Iterate through each file, load it, and store it in the dictionary
    for key, fname in files.items():
        path = data_dir / fname
        dfs[key] = pd.read_csv(path)
        print(f"✅ Loaded {key} with {dfs[key].shape[0]:,} rows")
    
    return dfs


def compute_age(birth_date: pd.Series, reference_date: pd.Series) -> pd.Series:
    """
    Return age in years (float) for each reference date.
    
    Parameters:
        birth_date (pd.Series): Series of birth dates (datetime format)
        reference_date (pd.Series): Series of reference dates (datetime format)
    
    Returns:
        pd.Series: Series of ages in years (float)
    
    Why call it compute_age:
        - "compute" = calculate
        - "age" = age
        This function computes the age difference between two dates.
    
    Calculation logic:
        1. reference_date - birth_date results in timedelta
        2. .dt.days to get number of days
        3. / 365.25 to convert to years (taking leap years into account)
    
    Example usage:
        age = compute_age(df["date_of_birth"], df["valuation_date"])
    """
    return (reference_date - birth_date).dt.days / 365.25


def agg_by_age_sub_position(
    df: pd.DataFrame, 
    value_col: str, 
    smooth: bool = False
) -> pd.DataFrame:
    """
    Aggregate a column by age & position; optionally apply rolling smoothing.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing player data
        value_col (str): Name of the column to aggregate (e.g. "value_million")
        smooth (bool): Whether to apply rolling window smoothing (default False)
    
    Returns:
        pd.DataFrame: Aggregated DataFrame with sub_position, age, and expected_{value_col}
    
    Why call it agg_by_age_sub_position:
        - "agg" = abbreviation for aggregate
        - "by_age_sub_position" = grouped by age and sub_position
        This function groups data by age and position and computes statistics.
    
    Calculation logic:
        1. groupby sub_position and age
        2. median() for robust statistics
        3. If smooth=True, apply rolling window of 3 ages to smooth the curve
        4. Rename column to expected_{value_col}
    
    Example usage:
        valuation_curve = agg_by_age_sub_position(
            df=valuations, 
            value_col="value_million", 
            smooth=True
        )
    """
    grouped = (
        df.groupby(["sub_position", "age"])[value_col]
        .median()
        .reset_index()
        .sort_values(["sub_position", "age"])
    )
    
    if smooth:
        grouped[value_col] = grouped.groupby("sub_position")[value_col].transform(
            lambda s: s.rolling(3, center=True, min_periods=1).mean()
        )
    
    grouped.rename(columns={value_col: f"expected_{value_col}"}, inplace=True)
    return grouped


def zscore(series: pd.Series) -> pd.Series:
    """
    Return the z-score with guard against zero std.
    
    Parameters:
        series (pd.Series): Numeric series to standardize
    
    Returns:
        pd.Series: Standardized series (mean 0, std 1)
    
    Why call it zscore:
        - "z" = standard score in statistics
        - "score" = score
        z-score is a standard statistical term.
    
    Calculation:
        z = (x - mean) / std
        
    Use:
        Standardize indicators to a comparable scale
        e.g., combine market value z-score and performance z-score directly
    
    Example usage:
        df["valuation_z"] = zscore(df["valuation_above_curve"])
    """
    std = series.std(ddof=0)  # ddof=0: population std (divide by N)
    
    if std == 0:
        return pd.Series(0, index=series.index)
    
    return (series - series.mean()) / std


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def build_aging_curves(dfs: dict) -> pd.DataFrame:
    """
    Build aging curves for valuation and performance by age and position.
    
    Parameters:
        dfs (dict): Dictionary of all raw data (from load_raw_data)
    
    Returns:
        pd.DataFrame: Aging curve data, including:
                     - sub_position: player position
                     - age: age
                     - expected_value_million: expected market value (in million euros)
                     - expected_ga_per_90: expected goals+assists per 90
                     - expected_minutes_per_90: expected minutes per 90
    
    Why call it build_aging_curves:
        - "build" = build, construct
        - "aging_curves" = curves describing player performance by age
    
    Steps:
        1. Prepare data: merge player properties and compute ages
        2. Filter data: remove groups with too few samples & outlier ages
        3. Build market value curve
        4. Build performance curves (goals+assists, minutes played)
        5. Merge all curves
    
    Example usage:
        curves = build_aging_curves(dfs)
    """
    print("\n" + "="*70)
    print("Building Aging Curves...")
    print("="*70)
    
    # ------------------------------------------------------------------------
    # 1. Data Preparation
    # ------------------------------------------------------------------------
    players = dfs["players"].copy()
    valuations = dfs["player_valuations"].copy()
    appearances = dfs["appearances"].copy()
    
    # Convert columns to datetime
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    
    # ------------------------------------------------------------------------
    # 2. Process Market Value Data (Valuations)
    # ------------------------------------------------------------------------
    # Merge player info into valuations
    valuations = valuations.merge(
        players[["player_id", "sub_position", "date_of_birth"]], 
        on="player_id", 
        how="inner"
    )
    
    valuations.dropna(subset=["date", "date_of_birth"], inplace=True)
    
    # Compute precise age
    valuations["age_exact"] = compute_age(valuations["date_of_birth"], valuations["date"])
    
    # Round age to nearest 0.25 years (3 months)
    valuations["age"] = np.round(valuations["age_exact"] * 4) / 4
    
    # Keep players aged 16-40 only
    valuations = valuations[(valuations["age"] >= 16) & (valuations["age"] <= 40)]
    
    valuations["value_million"] = valuations["market_value_in_eur"] / MILLION
    
    # ------------------------------------------------------------------------
    # 3. Filter Market Value Data (improve quality)
    # ------------------------------------------------------------------------
    valuations_age_filtered = valuations[valuations["age"] < 38]
    
    # Count number of samples per (position, age) group
    valuation_counts = (
        valuations_age_filtered
        .groupby(["sub_position", "age"])
        .size()
        .reset_index(name="count")
    )
    
    # Keep only groups with >= 10 samples
    valid_age_groups = valuation_counts[valuation_counts["count"] >= 10][
        ["sub_position", "age"]
    ]
    
    valuations_filtered = valuations_age_filtered.merge(
        valid_age_groups, 
        on=["sub_position", "age"], 
        how="inner"
    )
    
    # Build aging curve for market value (with smoothing)
    valuation_curve = agg_by_age_sub_position(
        valuations_filtered, 
        "value_million", 
        smooth=True
    )
    
    # ------------------------------------------------------------------------
    # 4. Process Appearances Data
    # ------------------------------------------------------------------------
    appearances = appearances.merge(
        players[["player_id", "sub_position", "date_of_birth"]], 
        on="player_id", 
        how="inner"
    )
    appearances.dropna(subset=["date", "date_of_birth"], inplace=True)
    appearances["age_exact"] = compute_age(appearances["date_of_birth"], appearances["date"])
    appearances["age"] = np.round(appearances["age_exact"] * 4) / 4
    appearances = appearances[(appearances["age"] >= 16) & (appearances["age"] <= 40)]
    
    # Clip minutes_played at minimum 1 to avoid divide by zero
    appearances["minutes_per_appearance"] = appearances["minutes_played"].clip(lower=1)
    
    # Exclude players age 38+
    appearances_age_filtered = appearances[appearances["age"] < 38]
    
    # ------------------------------------------------------------------------
    # 5. Compute Performance Aging Curves
    # ------------------------------------------------------------------------
    # Aggregate by (position, age): total goals, assists, and minutes
    perf_agg = appearances_age_filtered.groupby(["sub_position", "age"]).agg({
        "goals": "sum",
        "assists": "sum",
        "minutes_per_appearance": "sum",
        "player_id": "count"
    }).reset_index()
    
    perf_agg.rename(columns={"player_id": "sample_count"}, inplace=True)
    
    perf_agg_filtered = perf_agg[perf_agg["sample_count"] >= 20].copy()
    
    # Calculate goals+assists per 90 minutes
    perf_agg_filtered["ga_per_90"] = (
        (perf_agg_filtered["goals"] + perf_agg_filtered["assists"]) / 
        perf_agg_filtered["minutes_per_appearance"]
    ) * 90
    
    # Calculate average minutes per appearance
    perf_agg_filtered["minutes_per_90"] = (
        perf_agg_filtered["minutes_per_appearance"] / 
        perf_agg_filtered["sample_count"]
    )
    
    # ------------------------------------------------------------------------
    # 6. Clip Extreme Values and Smooth
    # ------------------------------------------------------------------------
    # Use 5th and 95th percentiles to remove outliers
    ga_5th = perf_agg_filtered["ga_per_90"].quantile(0.05)
    ga_95th = perf_agg_filtered["ga_per_90"].quantile(0.95)
    
    perf_agg_filtered["ga_per_90"] = perf_agg_filtered["ga_per_90"].clip(
        lower=ga_5th, 
        upper=ga_95th
    )
    
    perf_agg_filtered = perf_agg_filtered.sort_values(["sub_position", "age"])
    
    perf_agg_filtered["ga_per_90"] = perf_agg_filtered.groupby("sub_position")[
        "ga_per_90"
    ].transform(
        lambda s: s.rolling(5, center=True, min_periods=2).mean()
    )
    
    perf_agg_filtered["minutes_per_90"] = perf_agg_filtered.groupby("sub_position")[
        "minutes_per_90"
    ].transform(
        lambda s: s.rolling(5, center=True, min_periods=2).mean()
    )
    
    # ------------------------------------------------------------------------
    # 7. Prepare Final Curve Data
    # ------------------------------------------------------------------------
    perf_ga_curve = perf_agg_filtered[["sub_position", "age", "ga_per_90"]].rename(
        columns={"ga_per_90": "expected_ga_per_90"}
    )
    
    perf_min_curve = perf_agg_filtered[["sub_position", "age", "minutes_per_90"]].rename(
        columns={"minutes_per_90": "expected_minutes_per_90"}
    )
    
    # Merge all curves
    curves = valuation_curve.merge(
        perf_ga_curve, 
        on=["sub_position", "age"], 
        how="outer"
    )
    curves = curves.merge(
        perf_min_curve, 
        on=["sub_position", "age"], 
        how="outer"
    )
    curves.sort_values(["sub_position", "age"], inplace=True)
    
    print(f"✅ Valuation curve: {len(valuation_curve)} age-position combinations")
    print(f"✅ Performance curve: {len(perf_ga_curve)} age-position combinations")
    print(f"✅ GA per 90 range: {perf_ga_curve['expected_ga_per_90'].min():.3f} - "
          f"{perf_ga_curve['expected_ga_per_90'].max():.3f}")
    print(f"✅ Market value range: {valuation_curve['expected_value_million'].min():.3f} - "
          f"{valuation_curve['expected_value_million'].max():.3f}M€")
    
    return curves, valuations, appearances


def compute_player_deviations(
    curves: pd.DataFrame, 
    valuations: pd.DataFrame, 
    appearances: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute per-player deviations from aging curves.
    
    Parameters:
        curves (pd.DataFrame): Aging curve data
        valuations (pd.DataFrame): Player market value data
        appearances (pd.DataFrame): Player appearance data
    
    Returns:
        pd.DataFrame: DataFrame with player deviation metrics, including:
                     - aging_score: composite aging score
                     - development_tier: development tier (declining/normal/aging well)
    
    Why call it compute_player_deviations:
        - "compute" = calculate
        - "player_deviations" = player deviations
        This function calculates how much each player deviates from the expected curves.
    
    Calculation logic:
        1. Get each player's latest market value
        2. Calculate each player's actual performance indicators
        3. Merge expected values (from aging curves)
        4. Deviation = actual value - expected value
        5. Standardize deviations as z-score
        6. Calculate composite score and tier
    
    Example usage:
        development_df = compute_player_deviations(curves, valuations, appearances)
    """
    print("\n" + "="*70)
    print("Computing Player-Level Deviations...")
    print("="*70)
    
    # ------------------------------------------------------------------------
    # 1. Get latest market value for each player
    # ------------------------------------------------------------------------
    latest_values = valuations.sort_values("date").drop_duplicates(
        "player_id", 
        keep="last"
    )
    
    # ------------------------------------------------------------------------
    # 2. Calculate actual player performance metrics
    # ------------------------------------------------------------------------
    player_performance = appearances.groupby("player_id").agg({
        "goals": "sum",
        "assists": "sum",
        "minutes_per_appearance": "sum",
        "game_id": "count"
    }).reset_index()
    
    player_performance.rename(columns={"game_id": "appearance_count"}, inplace=True)
    
    player_performance["ga_per_90"] = (
        (player_performance["goals"] + player_performance["assists"]) / 
        player_performance["minutes_per_appearance"]
    ) * 90
    
    player_performance["minutes_per_90"] = (
        player_performance["minutes_per_appearance"] / 
        player_performance["appearance_count"]
    )
    
    player_performance = player_performance[[
        "player_id", 
        "ga_per_90", 
        "minutes_per_90"
    ]]
    
    # ------------------------------------------------------------------------
    # 3. Merge performance and expected values
    # ------------------------------------------------------------------------
    latest_values = latest_values.merge(
        player_performance, 
        on="player_id", 
        how="left"
    )
    
    latest_values = latest_values.merge(
        curves, 
        on=["sub_position", "age"], 
        how="left", 
        suffixes=("", "_expected")
    )
    
    # ------------------------------------------------------------------------
    # 4. Compute deviation values (actual - expected)
    # ------------------------------------------------------------------------
    latest_values["valuation_above_curve"] = (
        latest_values["value_million"] - 
        latest_values["expected_value_million"]
    )
    
    latest_values["performance_above_curve"] = (
        latest_values["ga_per_90"] - 
        latest_values["expected_ga_per_90"]
    )
    
    latest_values["minutes_above_curve"] = (
        latest_values["minutes_per_90"] - 
        latest_values["expected_minutes_per_90"]
    )
    
    # ------------------------------------------------------------------------
    # 5. Compute z-scores (standardized deviations)
    # ------------------------------------------------------------------------
    latest_values["valuation_z"] = zscore(
        latest_values["valuation_above_curve"].fillna(0)
    )
    latest_values["performance_z"] = zscore(
        latest_values["performance_above_curve"].fillna(0)
    )
    latest_values["minutes_z"] = zscore(
        latest_values["minutes_above_curve"].fillna(0)
    )
    
    # ------------------------------------------------------------------------
    # 6. Compute Composite Aging Score
    # ------------------------------------------------------------------------
    # aging_score: weighted sum of the three z-scores
    # - 50% market value z-score (most important indicator)
    # - 30% performance z-score (goals+assists)
    # - 20% minutes played z-score
    # 
    # Higher score = performing better than expected = aging well
    # Lower score = underperforming = possibly declining
    latest_values["aging_score"] = (
        0.5 * latest_values["valuation_z"] +
        0.3 * latest_values["performance_z"] +
        0.2 * latest_values["minutes_z"]
    )
    
    # ------------------------------------------------------------------------
    # 7. Assign Development Tiers
    # ------------------------------------------------------------------------
    # pd.cut: cut continuous variable into intervals
    # bins:
    #   - (-inf, 0]: declining
    #   - (0, 0.75]: normal
    #   - (0.75, inf): aging well
    # labels: tier labels
    latest_values["development_tier"] = pd.cut(
        latest_values["aging_score"],
        bins=[-np.inf, 0, 0.75, np.inf],
        labels=["declining", "normal", "aging well"],
    )
    
    print(f"✅ Computed deviations for {len(latest_values)} players")
    print(f"✅ Development tier distribution:")
    print(latest_values["development_tier"].value_counts())
    
    return latest_values


def save_development_outputs(development_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save development analysis results to parquet file.
    
    Parameters:
        development_df (pd.DataFrame): DataFrame with development metrics
        output_dir (Path): Output directory path
    
    Returns:
        Path: Full path to the output file
    
    Why call it save_development_outputs:
        - "save" = save
        - "development_outputs" = outputs of the development analysis
    
    Output columns:
        - player_id: Player ID
        - age: Age
        - sub_position: Position
        - value_million: Actual market value
        - expected_value_million: Expected market value
        - valuation_above_curve: Market value deviation
        - ga_per_90: Actual goals+assists per 90
        - expected_ga_per_90: Expected goals+assists per 90
        - performance_above_curve: Performance deviation
        - minutes_per_90: Actual minutes per 90
        - expected_minutes_per_90: Expected minutes per 90
        - minutes_above_curve: Minutes deviation
        - aging_score: Composite aging score
        - development_tier: Development tier
    
    Example usage:
        output_path = save_development_outputs(development_df, PROCESSED_DIR)
    """
    print("\n" + "="*70)
    print("Saving Development Outputs...")
    print("="*70)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    output_cols = [
        "player_id",
        "age",
        "sub_position",
        "value_million",
        "expected_value_million",
        "valuation_above_curve",
        "ga_per_90",
        "expected_ga_per_90",
        "performance_above_curve",
        "minutes_per_90",
        "expected_minutes_per_90",
        "minutes_above_curve",
        "aging_score",
        "development_tier",
    ]
    
    development_outputs = development_df[output_cols].copy()
    
    output_path = output_dir / "development_outputs.parquet"
    
    development_outputs.to_parquet(output_path, index=False)
    
    print(f"✅ Saved {len(development_outputs)} records to {output_path}")
    print(f"✅ File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return output_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Main function to execute the complete development analysis pipeline.
    
    Steps:
        1. Load raw data
        2. Build aging curves
        3. Compute player deviations
        4. Save output results
    
    Why call it main:
        - "main" = main
        This is the main entry point of the program and runs the whole pipeline
    
    How to use:
        Run this script directly: python generate_development_outputs.py
    """
    print("\n" + "="*70)
    print("PLAYER DEVELOPMENT ANALYSIS PIPELINE")
    print("="*70)
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Output directory: {PROCESSED_DIR.absolute()}")
    print("="*70)
    
    try:
        # Step 1: Load raw data
        print("\n[Step 1/4] Loading raw data...")
        dfs = load_raw_data(DATA_DIR)
        
        # Step 2: Build aging curves
        print("\n[Step 2/4] Building aging curves...")
        curves, valuations, appearances = build_aging_curves(dfs)
        
        # Step 3: Compute player deviations
        print("\n[Step 3/4] Computing player deviations...")
        development_df = compute_player_deviations(curves, valuations, appearances)
        
        # Step 4: Save outputs
        print("\n[Step 4/4] Saving outputs...")
        output_path = save_development_outputs(development_df, PROCESSED_DIR)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Output file: {output_path}")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ PIPELINE FAILED!")
        print("="*70)
        print(f"Error: {str(e)}")
        print("="*70)
        raise


# Standard Python idiom: only execute main() if this script is run directly
# If this file is imported, main() will not be run automatically
if __name__ == "__main__":
    main()

