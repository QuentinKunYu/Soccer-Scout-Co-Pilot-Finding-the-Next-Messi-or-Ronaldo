"""Table and selection components for the scouting UI."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

TABLE_COLUMNS = {
    "player_name": "Player",
    "age": "Age",
    "position": "Pos",
    "club_name": "Club",
    "league_name": "League",
    "current_market_value": "Current MV (€M)",
    "mv_pred_1y": "Pred MV (€M)",
    "y_growth_pred": "Growth %",
    "breakout_prob": "Breakout %",
    "undervalued_score": "Undervalued (€M)",
}


def _format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric fields in millions so the table stays readable."""

    formatted = df.copy()
    formatted["current_market_value"] = (formatted["current_market_value"] / 1_000_000).round(2)
    formatted["mv_pred_1y"] = (formatted["mv_pred_1y"] / 1_000_000).round(2)
    formatted["undervalued_score"] = (formatted["undervalued_score"] / 1_000_000).round(2)
    formatted["y_growth_pred"] = (formatted["y_growth_pred"] * 100).round(1)
    formatted["breakout_prob"] = (formatted["breakout_prob"] * 100).round(1)
    return formatted


def render_player_table(df: pd.DataFrame) -> Optional[pd.Series]:
    """Render the selection table and return the chosen row."""

    if df.empty:
        st.warning("No players match the current filters.")
        return None

    st.subheader("Recommended Players")
    display_df = _format_for_display(df[list(TABLE_COLUMNS.keys())])
    display_df = display_df.rename(columns=TABLE_COLUMNS)
    st.dataframe(display_df, use_container_width=True)

    player_names = df["player_name"].tolist()
    selected = st.selectbox("Select a player to inspect", player_names)
    return df[df["player_name"] == selected].iloc[0]

