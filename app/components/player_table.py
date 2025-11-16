"""
Table and selection components for the scouting UI.

This module handles:
- Displaying recommended player list
- Formatting numeric fields (market value, percentages, etc.)
- Providing player selection functionality
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

# TABLE_COLUMNS: Maps original field names to display names
# Converts technical database field names to user-friendly labels
TABLE_COLUMNS = {
    "player_name": "Player Name",
    "age": "Age",
    "sub_position": "Position",
    "club_name": "Club",
    "league_name": "League",
    "current_market_value": "Current MV (€M)",
    "mv_pred_1y": "Pred MV (€M)",
    "y_growth_pred": "Growth %",
    "breakout_prob": "Breakout %",
    "undervalued_score": "Undervalued (€M)",
    "aging_score": "Aging Score",
    "development_tier": "Dev Tier",
}


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format numeric fields to make the table more readable.
    
    Conversions:
    - Market value fields: Convert from raw values to millions of euros (divide by 1,000,000)
    - Percentage fields: Convert from decimals (0.5) to percentages (50.0)
    - Score fields: Round to two decimal places
    """
    formatted = df.copy()
    
    # Convert market value fields to millions of euros
    formatted["current_market_value"] = (formatted["current_market_value"] / 1_000_000).round(2)
    formatted["mv_pred_1y"] = (formatted["mv_pred_1y"] / 1_000_000).round(2)
    formatted["undervalued_score"] = (formatted["undervalued_score"] / 1_000_000).round(2)
    
    # Convert percentage fields (0.5 → 50.0)
    formatted["y_growth_pred"] = (formatted["y_growth_pred"] * 100).round(1)
    formatted["breakout_prob"] = (formatted["breakout_prob"] * 100).round(1)
    
    # Round score fields
    if "aging_score" in formatted.columns:
        formatted["aging_score"] = formatted["aging_score"].round(2)
    
    return formatted


def render_player_table(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Render player selection table and return selected player data.
    
    Functionality:
    1. Check if there are players matching the filter criteria
    2. Display formatted player list table
    3. Provide dropdown menu for user to select a player
    4. Return complete data for selected player (a pandas Series)
    """
    # Check if data exists
    if df.empty:
        st.warning("No players match the current filter criteria")
        return None

    # Display table title
    st.markdown("## Recommended Players")
    st.caption(f"Found {len(df)} players matching criteria")
    
    # Format and display table
    # 1. Select only the fields to display (defined in TABLE_COLUMNS)
    display_df = _format_for_display(df[list(TABLE_COLUMNS.keys())])
    
    # 2. Rename technical field names to display names
    display_df = display_df.rename(columns=TABLE_COLUMNS)
    
    # 3. Display DataFrame, use_container_width=True makes table fill container width
    st.dataframe(display_df, use_container_width=True, height=400)

    # Provide dropdown menu for user to select a player
    st.markdown("---")
    player_names = df["player_name"].tolist()
    selected = st.selectbox(
        "Select a player to view details", 
        player_names,
        help="Choose a player from the list to view complete analysis report"
    )
    
    # Return complete data row for selected player (includes all original fields)
    # iloc[0]: Get the first (and only) matching record
    return df[df["player_name"] == selected].iloc[0]
