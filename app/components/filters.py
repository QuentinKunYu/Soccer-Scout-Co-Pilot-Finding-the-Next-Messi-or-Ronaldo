"""Sidebar filter widgets for the scouting dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import streamlit as st


@dataclass
class FilterState:
    """Lightweight container describing the active filters."""

    league: str | None
    position: str | None
    age_range: Tuple[int, int]
    min_breakout: float
    min_undervalued: float


def render_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, FilterState]:
    """Render widgets and return the filtered dataframe along with metadata."""

    with st.sidebar:
        st.header("Filters")
        league = st.selectbox(
            "League",
            options=["All"] + sorted(df["league_name"].unique()),
            index=0,
        )
        position = st.selectbox(
            "Position",
            options=["All"] + sorted(df["position"].unique()),
            index=0,
        )
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        min_breakout = st.slider("Min breakout probability", 0.0, 1.0, 0.3, step=0.05)
        min_undervalued = st.slider("Min undervalued score (M€)", 0.0, 10.0, 1.0, step=0.5)

    mask = (df["age"].between(age_range[0], age_range[1])) & (
        df["breakout_prob"] >= min_breakout
    ) & ((df["undervalued_score"] / 1_000_000) >= min_undervalued)

    if league != "All":
        mask &= df["league_name"] == league
    if position != "All":
        mask &= df["position"] == position

    filtered = df[mask].copy()

    filters = FilterState(
        league=None if league == "All" else league,
        position=None if position == "All" else position,
        age_range=age_range,
        min_breakout=min_breakout,
        min_undervalued=min_undervalued,
    )
    return filtered, filters

