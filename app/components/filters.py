"""
Sidebar filter widgets for the scouting dashboard.

This module provides:
- Multiple filter criteria (league, position, age, etc.)
- Data structure for filter state
- Functionality to filter player data based on criteria
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import streamlit as st


@dataclass
class FilterState:
    """
    Lightweight container describing active filters.
    
    Attributes:
    - league: Selected league (None means "All")
    - position: Selected position (None means "All")
    - age_range: Age range (min_age, max_age)
    - budget_range: Budget range for market value (min_budget, max_budget) in millions of euros
    - min_breakout: Minimum breakout probability (0.0-1.0)
    - min_undervalued: Minimum undervalued amount (millions of euros)
    - min_aging_score: Minimum aging score
    - development_tier: Development stage (None means "All")
    """

    league: str | None
    position: str | None
    age_range: Tuple[int, int]
    budget_range: Tuple[float, float]
    min_breakout: float
    min_undervalued: float
    min_aging_score: float
    development_tier: str | None


def render_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, FilterState]:
    """
    Render sidebar filter widgets and return filtered data with filter state.
    
    Flow:
    1. Display various filter input widgets in the sidebar
    2. Filter data based on user-selected criteria
    3. Return filtered DataFrame and FilterState object
    """
    with st.sidebar:
        # Sidebar title
        st.markdown("# Filter Criteria")
        st.markdown("---")
        
        # League filter
        st.markdown("### Basic Criteria")
        league = st.selectbox(
            "League",
            options=["All"] + sorted(df["league_name"].unique()),
            index=0,
            help="Select a specific league or view all leagues"
        )
        
        # Sub-position filter
        sub_position = st.selectbox(
            "Position",
            options=["All"] + sorted(df["sub_position"].unique()),
            index=0,
            help="Select player sub-position (Attacker, Midfielder, Defender, Goalkeeper)"
        )
        
        # Age range filter
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        age_range = st.slider(
            "Age Range", 
            min_value=age_min, 
            max_value=age_max, 
            value=(age_min, age_max),
            help="Set minimum and maximum age for players"
        )
        
        # Budget filter (based on current_market_value)
        # Convert current_market_value to millions of euros for easier reading
        mv_min = float(df["current_market_value"].min() / 1_000_000)
        mv_max = float(df["current_market_value"].max() / 1_000_000)
        
        # Single-value max budget slider (left side fixed)
        max_budget = st.slider(
            "Maximum Budget", 
            min_value=mv_min, 
            max_value=mv_max, 
            value=mv_max,  # Default to maximum (no filtering)
            step=0.5,
            format="€%.1fM",
            help="Set maximum budget for player market value"
        )
        
        # Create budget range with fixed minimum and adjustable maximum
        budget_range = (mv_min, max_budget)
        
        st.markdown("---")
        st.markdown("### Investment Potential")
        
        # Breakout probability filter
        min_breakout = st.slider(
            "Min Breakout Probability", 
            0, 100, 30, 
            step=5,
            format="%d%%",
            help="Probability of significant performance improvement (0-100%)"
        )
        # Convert to 0-1 range for filtering
        min_breakout = min_breakout / 100.0
        
        # Undervalued score filter
        min_undervalued = st.slider(
            "Min Undervalued Amount (M€)", 
            0.0, 10.0, 1.0, 
            step=0.5,
            help="Degree to which market value is underestimated (millions of euros)"
        )
        
        # Aging score filter
        min_aging_score = st.slider(
            "Min Aging Score", 
            -1.5, 1.5, 0.0, 
            step=0.1,
            help="Positive values indicate performance above age expectations, negative values below"
        )
        
        st.markdown("---")
        st.markdown("### Development Stage")
        
        # Development tier filter
        tier_values = df["development_tier"].dropna().unique().tolist() if "development_tier" in df.columns else []
        tier_options = ["All"] + sorted(tier_values)
        tier_selection = st.selectbox(
            "Development Stage", 
            options=tier_options, 
            index=0,
            help="Player's current career development stage"
        )

    # Create filter mask: starts with all True
    mask = pd.Series(True, index=df.index, dtype=bool)
    
    # Apply filter criteria (using &= for AND logical operations)
    mask &= df["age"].between(age_range[0], age_range[1])  # Age within range
    # Budget filter: convert current_market_value to millions and check if within range
    mask &= (df["current_market_value"] / 1_000_000).between(budget_range[0], budget_range[1])
    mask &= df["breakout_prob"] >= min_breakout  # Breakout probability >= minimum value
    mask &= (df["undervalued_score"] / 1_000_000) >= min_undervalued  # Undervalued score >= minimum value
    
    if "aging_score" in df.columns:
        mask &= df["aging_score"] >= min_aging_score  # Aging score >= minimum value

    # Apply categorical filters
    if league != "All":
        mask &= df["league_name"] == league
    if sub_position != "All":
        mask &= df["sub_position"] == sub_position
    if tier_selection != "All" and "development_tier" in df.columns:
        mask &= df["development_tier"] == tier_selection

    # Filter data based on mask
    filtered = df[mask].copy()

    # Create FilterState object to record current filter state
    filters = FilterState(
        league=None if league == "All" else league,
        position=None if sub_position == "All" else sub_position,
        age_range=age_range,
        budget_range=budget_range,
        min_breakout=min_breakout,
        min_undervalued=min_undervalued,
        min_aging_score=min_aging_score,
        development_tier=None if tier_selection == "All" else tier_selection,
    )
    
    return filtered, filters
