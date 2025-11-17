"""
Generate final player recommendations by integrating all model outputs.

This module combines outputs from multiple models and data sources to create
a comprehensive player recommendation dataset for the frontend application.

The function merges:
- Player snapshot data (basic player information and performance metrics)
- Regression model outputs (market value growth predictions)
- Classification model outputs (breakout probability predictions)
- Development curve analysis (aging curves and development tiers)
- Player images and market value history

Output:
    - data/processed/player_recommendations.parquet
    - data/processed/player_recommendations.csv
    - app/mock_data/player_recommendations.csv (for frontend use)

Usage:
    from src.data_helper.generate_recommendations import generate_player_recommendations
    recommendations = generate_player_recommendations()
"""

import pandas as pd
import json
import os
from pathlib import Path

def generate_player_recommendations():
    """
    Integrate all model outputs to generate the final player recommendations dataset.
    
    This function orchestrates the complete recommendation generation pipeline by:
    1. Loading player snapshot data (basic player information and performance metrics)
    2. Loading regression model outputs (market value growth predictions)
    3. Loading classification model outputs (breakout probability predictions)
    4. Loading development curve analysis (aging curves and development tiers)
    5. Merging all datasets together
    6. Calculating derived features (undervalued_score, mv_momentum_12m)
    7. Generating market value history from historical data
    8. Selecting and renaming columns for frontend compatibility
    9. Saving outputs to multiple locations (processed data and frontend mock data)
    
    The function includes fallback logic to create mock data if any input files
    are missing, making it robust for development and testing scenarios.
    
    Returns
    -------
    pd.DataFrame
        Complete player recommendations dataset with all features required by
        the frontend application. Includes:
        - Basic player info (name, age, position, club, league)
        - Market value predictions (current, predicted, growth rate)
        - Breakout probability and undervalued score
        - Performance statistics (goals, assists, minutes per 90)
        - SHAP feature importance (regression and classification)
        - Market value history (JSON format)
        - Development curve metrics (aging score, development tier)
        - Player image URL
        
    Examples
    --------
    >>> recommendations = generate_player_recommendations()
    >>> print(f"Generated {len(recommendations)} player recommendations")
    >>> print(recommendations.columns.tolist())
    """
    # Determine correct data paths
    # Dynamically adjust path based on current script location
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent.parent  # Project root directory
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    
    # Ensure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load basic snapshot data
    print("Loading snapshot data...")
    try:
        snapshot = pd.read_parquet(data_dir / "processed" / "player_snapshot.parquet")
        print(f"✓ Snapshot loaded: {len(snapshot)} records")
    except FileNotFoundError:
        print("⚠ Snapshot not found, using mock data...")
        # Create a simple mock snapshot
        snapshot = pd.DataFrame({
            'player_id': range(1001, 1011),
            'snapshot_date': pd.to_datetime(['2024-06-01'] * 10),
            'name': [f'Player {i}' for i in range(1001, 1011)],
            'age': [20, 22, 24, 19, 27, 25, 23, 21, 26, 28],
            'position': ['Attack'] * 5 + ['Midfield'] * 3 + ['Defence'] * 2,
            'sub_position': ['Right Winger', 'Central Midfield', 'Center Back', 'Attacking Midfield',
                            'Goalkeeper', 'Defensive Midfield', 'Left Winger', 'Right Back', 'Left Back', 'Striker'],
            'market_value_in_eur': [8000000, 12000000, 9500000, 6200000, 4600000, 7500000, 10000000, 5500000, 8200000, 15000000],
            'current_club_name': ['Palermo FC', 'London Bees', 'Hamburg Towers', 'Lisbon United', 
                                'Dublin City', 'Madrid Stars', 'Paris Eagles', 'Munich Lions', 'Amsterdam FC', 'Rome United'],
            'competition_name': ['Italian Serie B', 'English Championship', 'German Bundesliga', 'Portuguese Primeira Liga', 
                                'Scottish Premiership', 'Spanish La Liga', 'French Ligue 1', 'German Bundesliga', 'Dutch Eredivisie', 'Italian Serie A'],
            'minutes_per_game_365': [74.3, 82.1, 88.7, 68.2, 90.0, 75.5, 80.2, 65.8, 87.3, 78.9],
            'goals_per_90_365': [0.58, 0.29, 0.05, 0.35, 0.0, 0.12, 0.45, 0.08, 0.03, 0.62],
            'assists_per_90_365': [0.21, 0.33, 0.04, 0.25, 0.01, 0.18, 0.28, 0.15, 0.22, 0.17],
            'delta_goals_per_90': [0.27, 0.09, -0.02, 0.31, 0.0, 0.05, 0.18, -0.03, 0.01, 0.25],
            'delta_minutes_per_90': [18.0, 6.0, -4.0, 13.0, 0.0, 5.5, 8.2, -2.8, 3.3, 10.9],
            'mv_momentum_12m': [0.42, 0.24, 0.12, 0.51, 0.05, 0.22, 0.35, 0.15, 0.18, 0.28]
        })
    
    # 2. Load regression predictions
    try:
        regression = pd.read_parquet(data_dir / "processed" / "regression_outputs.parquet")
        print(f"✓ Regression outputs loaded: {len(regression)} records")
    except:
        print("⚠ Regression outputs not found, creating mock data...")
        import numpy as np
        np.random.seed(42)
        regression = snapshot[['player_id', 'snapshot_date']].copy()
        # Mock annual growth prediction (-0.3 to 1.5)
        regression['y_growth_pred'] = np.random.uniform(-0.3, 1.5, len(regression))
        # Mock 1-year-ahead market value (0.7x~1.5x current value)
        regression['mv_pred_1y'] = snapshot['market_value_in_eur'] * np.random.uniform(0.7, 1.5, len(regression))
        regression['reg_shap_top_features'] = '[{"feature": "goals_per_90", "shap_value": 0.26}, {"feature": "mv_momentum_12m", "shap_value": 0.18}]'
    
    # 3. Load classification predictions
    try:
        classification = pd.read_parquet(data_dir / "processed" / "classification_outputs.parquet")
        print(f"✓ Classification outputs loaded: {len(classification)} records")
    except:
        print("⚠ Classification outputs not found, creating mock data...")
        classification = snapshot[['player_id', 'snapshot_date']].copy()
        # Use random but reproducible mock data
        import numpy as np
        np.random.seed(42)
        classification['breakout_prob'] = np.random.uniform(0.15, 0.85, len(classification))
        classification['clf_shap_top_features'] = '[{"feature": "delta_goals_per_90", "shap_value": 0.31}, {"feature": "age", "shap_value": 0.17}]'
    
    # 4. Load development curve data
    try:
        development = pd.read_parquet(data_dir / "processed" / "development_outputs.parquet")
        print(f"✓ Development outputs loaded: {len(development)} records")
    except:
        print("⚠ Development outputs not found, creating mock data...")
        import numpy as np
        np.random.seed(42)
        development = snapshot[['player_id']].drop_duplicates().copy()
        n_players = len(development)
        
        # Mock development curve related features
        development['expected_value_million'] = np.random.uniform(3.0, 15.0, n_players)
        development['expected_ga_per_90'] = np.random.uniform(0.0, 0.6, n_players)
        development['expected_minutes_per_90'] = np.random.uniform(50.0, 90.0, n_players)
        development['valuation_above_curve'] = np.random.uniform(-2.0, 3.0, n_players)
        development['performance_above_curve'] = np.random.uniform(-0.1, 0.3, n_players)
        development['minutes_above_curve'] = np.random.uniform(-5.0, 10.0, n_players)
        development['aging_score'] = np.random.uniform(-0.5, 1.5, n_players)
        
        # Mock development tier
        tier_choices = ['aging well', 'normal', 'declining', 'peak']
        development['development_tier'] = np.random.choice(tier_choices, n_players, p=[0.25, 0.5, 0.15, 0.1])
    
    # 5. Load players info (for image_url)
    print("Loading players data...")
    try:
        players = pd.read_csv(data_dir / "players.csv")
        players = players[['player_id', 'image_url']].rename(columns={'image_url': 'img_url'})
        print(f"✓ Players data loaded: {len(players)} records")
    except:
        print("⚠ Players data not found, using mock image URLs...")
        players = pd.DataFrame({
            'player_id': snapshot['player_id'].unique(),
            'img_url': [
                'https://i.imgur.com/JMtHuVY.png',
                'https://i.imgur.com/K9DLGy8.png',
                'https://i.imgur.com/pGXzWyA.png',
                'https://i.imgur.com/2Yb7Tpk.png',
                'https://i.imgur.com/fL9Vw3q.png',
                'https://i.imgur.com/JMtHuVY.png',
                'https://i.imgur.com/K9DLGy8.png',
                'https://i.imgur.com/pGXzWyA.png',
                'https://i.imgur.com/2Yb7Tpk.png',
                'https://i.imgur.com/fL9Vw3q.png'
            ][:len(snapshot['player_id'].unique())]
        })
    
    # 6. Load market value history
    print("Loading market value history...")
    try:
        valuations = pd.read_csv(data_dir / "player_valuations.csv")
        valuations['date'] = pd.to_datetime(valuations['date'])
        print(f"✓ Valuations loaded: {len(valuations)} records")
    except:
        print("⚠ Valuations data not found, will use mock history...")
        # Create simple mock historical data
        valuations = pd.DataFrame()
    
    # 7. Merge all datasets
    print("\nMerging datasets...")
    recommendations = snapshot.merge(
        regression, on=['player_id', 'snapshot_date'], how='inner', suffixes=('', '_reg')
    ).merge(
        classification, on=['player_id', 'snapshot_date'], how='inner', suffixes=('', '_clf')
    ).merge(
        development, on='player_id', how='inner', suffixes=('', '_dev')
    ).merge(
        players, on='player_id', how='inner'
    )
    
    print(f"✓ Initial merge complete: {len(recommendations)} records")
    
    # 8. Keep only the latest snapshot for each player
    print("\nFiltering to latest snapshots...")
    latest = recommendations.groupby('player_id')['snapshot_date'].max().reset_index()
    recommendations = recommendations.merge(latest, on=['player_id', 'snapshot_date'])
    print(f"✓ Filtered to latest snapshots: {len(recommendations)} records")
    
    # 9. Derived feature calculations
    print("\nCalculating derived features...")
    
    # undervalued_score: predicted value minus current value
    recommendations['undervalued_score'] = (
        recommendations['mv_pred_1y'] - recommendations['market_value_in_eur']
    )
    
    # Rename columns to match frontend requirements
    column_mapping = {
        'name': 'player_name',
        'current_club_name': 'club_name',
        'competition_name': 'league_name',
        'market_value_in_eur': 'current_market_value',
        'minutes_per_game_365': 'minutes_per_90',
        'goals_per_90_365': 'goals_per_90',
        'assists_per_90_365': 'assists_per_90',
        'delta_minutes_per_90': 'delta_minutes_per_90',
    }
    
    recommendations = recommendations.rename(columns=column_mapping)
    
    # 10. Generate mv_history (market value history)
    print("Generating market value history...")
    
    def get_mv_history(player_id):
        """
        Extract market value history for the player from player_valuations.csv.
        Return JSON format for up to the most recent 20 records.
        If fewer than 20 records exist, return all available records.
        """
        if len(valuations) == 0:
            # If no history, use current value and simulated past values
            current_val = recommendations[recommendations['player_id'] == player_id]['current_market_value'].iloc[0]
            # Simulate last 20 records, increasing gradually
            simulated_history = []
            for i in range(19, -1, -1):  # oldest to newest (T-19 to T-0)
                factor = 0.55 + (19 - i) * 0.0237  # 0.55 to 1.00 over 20 steps
                simulated_history.append({
                    "date": f"T-{i}",
                    "value": int(current_val * factor)
                })
            
            return json.dumps(simulated_history)
        
        # Get all available history for this player, sorted by date (newest first)
        player_vals = valuations[valuations['player_id'] == player_id].sort_values('date', ascending=False)
        
        # Take up to 20 records, but if fewer exist, take all available
        num_records = min(len(player_vals), 20)
        player_vals = player_vals.head(num_records)
        
        if len(player_vals) == 0:
            # If the player has no history, use current value and simulated history
            current_val = recommendations[recommendations['player_id'] == player_id]['current_market_value'].iloc[0]
            simulated_history = []
            for i in range(19, -1, -1):
                factor = 0.55 + (19 - i) * 0.0237
                simulated_history.append({
                    "date": f"T-{i}",
                    "value": int(current_val * factor)
                })
            
            return json.dumps(simulated_history)
        
        history = []
        for _, row in player_vals.iterrows():
            history.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "value": int(row['market_value_in_eur'])
            })
        
        # Sort by date (oldest to newest) for chronological display
        return json.dumps(sorted(history, key=lambda x: x['date']))
    
    recommendations['mv_history'] = recommendations['player_id'].apply(get_mv_history)
    
    
    # 12. Calculate mv_momentum_12m (market value % change over last year)
    print("Calculating mv_momentum_12m...")
    
    def calculate_mv_momentum(player_id, current_mv):
        """
        Calculate market value momentum over the past 12 months.
        mv_momentum_12m = (this year's MV - last year's MV) / last year's MV.
        Return 0 or an estimated value if no history available.
        """
        if len(valuations) == 0:
            # No historical data, use half the predicted growth as estimate
            try:
                growth_pred = recommendations[recommendations['player_id'] == player_id]['y_growth_pred'].iloc[0]
                return float(growth_pred * 0.5)
            except:
                return 0.0
        
        # Get player history descending by date
        player_vals = valuations[valuations['player_id'] == player_id].sort_values('date', ascending=False)
        
        if len(player_vals) < 2:
            # Not enough history (less than 2 records)
            try:
                growth_pred = recommendations[recommendations['player_id'] == player_id]['y_growth_pred'].iloc[0]
                return float(growth_pred * 0.5)
            except:
                return 0.0
        
        # Take latest MV and MV from about one year ago
        current_val = float(current_mv)
        
        # Try to find a record about 12 months ago (in a window of 10-14 months)
        current_date = player_vals.iloc[0]['date']
        one_year_ago = current_date - pd.DateOffset(months=12)
        
        # Find the record closest to 12 months ago
        player_vals['days_diff'] = abs((player_vals['date'] - one_year_ago).dt.days)
        closest_record = player_vals.loc[player_vals['days_diff'].idxmin()]
        
        # Only use if within a 6-month window (9-15 months)
        days_threshold = 180
        if closest_record['days_diff'] <= days_threshold:
            last_year_val = float(closest_record['market_value_in_eur'])
            if last_year_val == 0:
                return 0.0
            momentum = (current_val - last_year_val) / last_year_val
            return float(momentum)
        else:
            # If not close enough in time, use predicted value
            try:
                growth_pred = recommendations[recommendations['player_id'] == player_id]['y_growth_pred'].iloc[0]
                return float(growth_pred * 0.5)
            except:
                return 0.0
    
    # If column already exists in snapshot, check if recalculation is needed
    if 'mv_momentum_12m' in recommendations.columns:
        print("⚠ mv_momentum_12m found in snapshot, recalculating from historical data...")
    
    # Calculate mv_momentum_12m for each player
    recommendations['mv_momentum_12m'] = recommendations.apply(
        lambda row: calculate_mv_momentum(row['player_id'], row['current_market_value']),
        axis=1
    )
    
    print(f"✓ mv_momentum_12m calculated. Range: [{recommendations['mv_momentum_12m'].min():.3f}, {recommendations['mv_momentum_12m'].max():.3f}]")
    
    # 13. Select only the columns needed for the frontend
    print("\nSelecting final columns...")
    final_cols = [
        # Basic info
        'player_id', 'player_name', 'age', 'sub_position', 'club_name', 'league_name',
        # Market value
        'current_market_value', 'mv_pred_1y', 'y_growth_pred', 'breakout_prob', 'undervalued_score',
        # Performance stats
        'minutes_per_90', 'goals_per_90', 'assists_per_90', 
        'delta_goals_per_90', 'delta_minutes_per_90', 'mv_momentum_12m',
        # SHAP feature importances
        'reg_shap_top_features', 'clf_shap_top_features',
        # Market value history
        'mv_history',
        # Development curve analysis
        'expected_value_million', 'expected_ga_per_90', 'expected_minutes_per_90',
        'valuation_above_curve', 'performance_above_curve', 'minutes_above_curve',
        'aging_score', 'development_tier',
        # Image
        'img_url'
    ]
    
    # Ensure all columns are present
    for col in final_cols:
        if col not in recommendations.columns:
            print(f"⚠ Warning: Column '{col}' not found, will be filled with NaN")
    
    available_cols = [col for col in final_cols if col in recommendations.columns]
    recommendations_final = recommendations[available_cols].copy()
    
    # 14. Save output
    print("\nSaving recommendations...")
    
    # Ensure output directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to processed directory
    recommendations_final.to_csv(processed_dir / 'player_recommendations.csv', index=False)
    recommendations_final.to_parquet(processed_dir / 'player_recommendations.parquet', index=False)
    
    # Also save a copy to app/mock_data for frontend use
    app_mock_dir = project_root / "data" / "processed"
    app_mock_dir.mkdir(parents=True, exist_ok=True)
    recommendations_final.to_csv(app_mock_dir / 'player_recommendations.csv', index=False)
    
    print(f"✓ Recommendations saved to {processed_dir}")
    print(f"✓ Copy saved to {app_mock_dir} for frontend use")
    print(f"✓ Generated {len(recommendations_final)} player recommendations")
    print(f"✓ Columns: {len(recommendations_final.columns)}")
    print("\nColumn list:")
    for col in recommendations_final.columns:
        print(f"  - {col}")
    
    return recommendations_final


if __name__ == "__main__":
    generate_player_recommendations()