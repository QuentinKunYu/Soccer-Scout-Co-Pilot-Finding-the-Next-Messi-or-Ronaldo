"""Detail view rendered when a user selects a player."""

from __future__ import annotations

from typing import List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from src.llm.schema import FeatureImportance


def _plot_mv_history(history: List[dict]) -> alt.Chart:
    """Build a line chart for market value history."""

    if not history:
        history = [{"date": "2024-01-01", "value": 0}]
    df = pd.DataFrame(history)
    df["value_million"] = df["value"] / 1_000_000
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="date:T", y=alt.Y("value_million:Q", title="Market Value (€M)"))
        .properties(height=300)
    )


def _render_feature_tags(features: Sequence, title: str) -> None:
    """Render SHAP features as a simple list."""

    st.markdown(f"**{title}**")
    if not features:
        st.write("No feature insights available yet.")
        return
    for item in features:
        if isinstance(item, FeatureImportance):
            feat, value = item.feature, item.shap_value
        elif isinstance(item, dict):
            feat = str(item.get("feature", "n/a"))
            value = float(item.get("shap_value", 0.0))
        else:
            feat = str(getattr(item, "feature", "n/a"))
            value = float(getattr(item, "shap_value", 0.0))
        st.write(f"- {feat}: {value:+.2f}")


def render_player_detail(row: pd.Series) -> None:
    """Render the detail panel for the selected player."""

    st.subheader(f"{row.player_name} — {row.position}")
    cols = st.columns(4)
    cols[0].metric("Age", row.age)
    cols[1].metric("Current MV (€M)", f"{row.current_market_value / 1_000_000:.2f}")
    cols[2].metric("Pred MV (€M)", f"{row.mv_pred_1y / 1_000_000:.2f}")
    cols[3].metric("Growth %", f"{row.y_growth_pred * 100:.1f}%")

    st.markdown(
        f"**Club / League:** {row.club_name} — {row.league_name}  |  "
        f"**Breakout Prob:** {row.breakout_prob * 100:.1f}%  |  "
        f"**Undervalued Score:** €{row.undervalued_score / 1_000_000:.1f}M"
    )

    stats_cols = st.columns(3)
    stats_cols[0].markdown(
        f"- Minutes/90: {row.minutes_per_90:.1f}\n"
        f"- Goals/90: {row.goals_per_90:.2f}\n"
        f"- Assists/90: {row.assists_per_90:.2f}"
    )
    stats_cols[1].markdown(
        f"- Δ Goals/90: {row.delta_goals_per_90:+.2f}\n"
        f"- Δ Minutes/90: {row.delta_minutes_per_90:+.1f}\n"
        f"- Rating: {row.rating_mean:.2f}"
    )
    stats_cols[2].markdown(f"- MV momentum (12m): {row.mv_momentum_12m:+.2f}")

    st.altair_chart(_plot_mv_history(row.mv_history), use_container_width=True)

    shap_cols = st.columns(2)
    with shap_cols[0]:
        _render_feature_tags(row.reg_shap_top_features, "Regression SHAP")
    with shap_cols[1]:
        _render_feature_tags(row.clf_shap_top_features, "Classification SHAP")
