"""Detail view rendered when a user selects a player."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from src.llm.schema import FeatureImportance

# Unified color palette for development visuals (shades of red).
ACTUAL_COLOR = "#f87171"      # light red for actual metrics
BENCHMARK_COLOR = "#dc2626"   # deeper red for curve benchmarks
PREDICTED_COLOR = "#fecaca"   # pale red for Chart Legend:

# Path to aging curve data
AGING_CURVE_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "development_outputs.parquet"


@st.cache_data(show_spinner=False)
def _load_aging_curves():
    """
    Load aging curve data from parquet file.
    
    Returns a DataFrame with aging curves for all positions, including:
    - expected_value_million: Market value curve
    - expected_ga_per_90: Goals+Assists performance curve  
    - expected_minutes_per_90: Playing time curve
    """
    if not AGING_CURVE_DATA_PATH.exists():
        return None
    
    df = pd.read_parquet(AGING_CURVE_DATA_PATH)
    
    # Group by position and age to get the aging curves
    # Take the mean of expected values for each position-age combination
    curves = df.groupby(["sub_position", "age"]).agg({
        "expected_value_million": "mean",
        "expected_ga_per_90": "mean",
        "expected_minutes_per_90": "mean"
    }).reset_index()
    
    return curves


def _plot_aging_curves(player_position: str, player_age: float, player_row: pd.Series) -> alt.Chart:
    """
    Plot aging curves for the player's position across three metrics.
    
    Args:
        player_position: Player's sub_position (e.g., "Centre-Forward")
        player_age: Player's current age
        player_row: Full player data row for plotting actual values
        
    Returns:
        Altair chart with three subplots showing aging curves
    """
    # Load aging curve data
    curves_df = _load_aging_curves()
    
    if curves_df is None:
        # Return empty chart with message if data not available
        return alt.Chart(pd.DataFrame({"message": ["Aging curve data not available"]})).mark_text(
            text="Aging curve data not available", size=14, color="#9ca3af"
        ).properties(height=300)
    
    # Filter for player's position only
    position_curves = curves_df[curves_df["sub_position"] == player_position].copy()
    
    if position_curves.empty:
        return alt.Chart(pd.DataFrame({"message": [f"No aging curve data for {player_position}"]})).mark_text(
            text=f"No aging curve data for {player_position}", size=14, color="#9ca3af"
        ).properties(height=300)
    
    # Calculate player's actual values (excluding market value)
    # Use ga_per_90 from development_outputs if available (calculated as total G+A / total minutes * 90)
    # Otherwise fall back to summing individual per-90 stats
    if "ga_per_90" in player_row.index and pd.notna(player_row.ga_per_90):
        player_ga = player_row.ga_per_90
    else:
        player_ga = (player_row.get("goals_per_90", 0) or 0) + (player_row.get("assists_per_90", 0) or 0)
    
    # Use minutes_per_90 from development_outputs if available
    if "minutes_per_90" in player_row.index and pd.notna(player_row.minutes_per_90):
        player_minutes = player_row.minutes_per_90
    else:
        player_minutes = player_row.get("minutes_per_90", 0) or 0
    
    # Create player actual value point (only for on-field metrics)
    player_point = pd.DataFrame({
        "age": [player_age, player_age],
        "value": [player_ga, player_minutes],
        "metric": ["Goals+Assists / 90min", "Playing Time (min / 90)"],
        "label": ["Your Player", "Your Player"]
    })
    
    # Prepare data for two curves (excluding market value)
    ga_data = position_curves[["age", "expected_ga_per_90"]].copy()
    ga_data["metric"] = "Goals+Assists / 90min"
    ga_data.rename(columns={"expected_ga_per_90": "value"}, inplace=True)
    
    minutes_data = position_curves[["age", "expected_minutes_per_90"]].copy()
    minutes_data["metric"] = "Playing Time (min / 90)"
    minutes_data.rename(columns={"expected_minutes_per_90": "value"}, inplace=True)
    
    # Combine curve data and player points into one DataFrame (excluding market value)
    all_curves = pd.concat([ga_data, minutes_data], ignore_index=True)
    all_curves["type"] = "Curve Benchmark"
    
    player_point["type"] = "Your Player"
    
    # Combine curves and player points
    combined_data = pd.concat([all_curves, player_point], ignore_index=True)
    
    # Create base chart with shared data
    base = alt.Chart(combined_data).encode(
        x=alt.X("age:Q", title="Age", scale=alt.Scale(domain=[16, 37])),
        y=alt.Y("value:Q", title="Value")
    )
    
    # Create line chart for curves
    line = base.transform_filter(
        alt.datum.type == "Curve Benchmark"
    ).mark_line(
        color=BENCHMARK_COLOR,
        strokeWidth=3
    ).encode(
        tooltip=[
            alt.Tooltip("age:Q", title="Age", format=".1f"),
            alt.Tooltip("value:Q", title="Expected Value", format=".2f")
        ]
    )
    
    # Create point for player's actual value
    point = base.transform_filter(
        alt.datum.type == "Your Player"
    ).mark_point(
        shape="diamond",
        size=200,
        color=ACTUAL_COLOR,
        filled=True
    ).encode(
        tooltip=[
            alt.Tooltip("label:N", title="Type"),
            alt.Tooltip("age:Q", title="Age", format=".1f"),
            alt.Tooltip("value:Q", title="Actual Value", format=".2f")
        ]
    )
    
    # Layer the charts together and then facet
    chart = (line + point).properties(
        width=600,
        height=200
    ).facet(
        row=alt.Row("metric:N", title=None, header=alt.Header(labelFontSize=13, labelFontWeight=600))
    ).resolve_scale(
        y="independent"
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )
    
    return chart


def _plot_mv_history(history: List[dict]) -> alt.Chart:
    """Build a line chart for market value history with red theme."""

    if not history:
        history = [{"date": "2024-01-01", "value": 0}]
    df = pd.DataFrame(history)
    df["value_million"] = df["value"] / 1_000_000
    
    # Create separate charts for line and points, then layer them
    line_chart = alt.Chart(df).mark_line(color="#dc2626").encode(
        x="date:T",
        y=alt.Y("value_million:Q", title="Market Value (€M)")
    )
    
    point_chart = alt.Chart(df).mark_circle(
        color="#dc2626",  # Red points
        size=60,          # Slightly larger points
        opacity=1         # Fully opaque
    ).encode(
        x="date:T",
        y="value_million:Q"
    )
    
    # Combine line and points
    return (
        (line_chart + point_chart)
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
    """
    Render detailed information panel for the selected player.
    
    This function displays:
    - Player basic information (name, position, age)
    - Market value metrics (current, predicted, growth rate)
    - Club and league information
    - Performance statistics
    - Development curve analysis
    - Market value history chart
    - AI feature importance analysis
    """
    # Display player photo, name, and position in a flex layout
    cols = st.columns([1, 3])
    
    # Left column: Player photo
    with cols[0]:
        if "img_url" in row and row.img_url:
            st.markdown(
                "<div style='padding-top: 24px;'></div>", 
                unsafe_allow_html=True
            )
            # Add border radius to player image using HTML/CSS wrapper
            st.markdown(
                f"""
                <div style="width: 150px; display: flex; align-items: center; justify-content: center;">
                    <img src="{row.img_url}" width="150" style="border-radius: 16px; object-fit: cover;" />
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Fallback if no image is available
            st.markdown(
                """
                <div style="width: 150px; height: 150px; 
                            background-color: #f3f4f6; 
                            border-radius: 16px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            color: #9ca3af;
                            font-size: 2rem;">
                    <span>📷</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Right column: Player name and position
    with cols[1]:
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; margin-top: 64px;">
                <span style="font-size: 2.125rem; font-weight: 700;">{row.player_name}</span>
                <span style="font-size: 1.15rem; color: var(--text-secondary-color, #6b7280); margin-top: 0.25rem;">{row.sub_position}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    # Core metrics card section - using clear labels
    st.markdown("### Core Metrics")
    
    # Check if wide mode is enabled
    if st.session_state.get("wide_mode", False):
        # Wide mode: display all 4 metrics in one row
        cols = st.columns(4)
        cols[0].metric("Age", f"{int(row.age)}")
        cols[1].metric(
            "Current MV", 
            f"€{row.current_market_value / 1_000_000:.2f}M",
            help="Player's current market valuation (millions of euros)"
        )
        cols[2].metric(
            "Pred MV (1 year)", 
            f"€{row.mv_pred_1y / 1_000_000:.2f}M",
            help="AI predicted market value in one year"
        )
        cols[3].metric(
            "Expected Growth", 
            f"{row.y_growth_pred * 100:.1f}%",
            help="Predicted annual market value growth rate"
        )
    else:
        # Normal mode: display 2 metrics per row (2 rows total)
        cols_row1 = st.columns(2)
        cols_row1[0].metric("Age", f"{int(row.age)}")
        cols_row1[1].metric(
            "Current MV", 
            f"€{row.current_market_value / 1_000_000:.2f}M",
            help="Player's current market valuation (millions of euros)"
        )
        
        cols_row2 = st.columns(2)
        cols_row2[0].metric(
            "Pred MV (1 year)", 
            f"€{row.mv_pred_1y / 1_000_000:.2f}M",
            help="AI predicted market value in one year"
        )
        cols_row2[1].metric(
            "Expected Growth", 
            f"{row.y_growth_pred * 100:.1f}%",
            help="Predicted annual market value growth rate"
        )

    # Club and opportunity score section with clean info box
    st.markdown("### Investment Opportunity")
    st.markdown(
        f"""
        <div class='info-box'>
            <p style='margin: 0; font-size: 1rem;'>
                <strong>Club:</strong> {row.club_name}
                <strong style='margin-left: 2rem;'>League:</strong> {row.league_name}
            </p>
            <p style='margin: 0.75rem 0 0 0; font-size: 1rem;'>
                <strong>Breakout Probability:</strong> <span style='color: #10b981; font-weight: 600;'>{row.breakout_prob * 100:.1f}%</span>
                <strong style='margin-left: 2rem;'>Undervalued Amount:</strong> <span style='color: #f59e0b; font-weight: 600;'>€{row.undervalued_score / 1_000_000:.1f}M</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Performance data section - divided into current performance and change trends
    st.markdown("### Performance Data")
    
    # Use expander with help text for better organization
    with st.expander("Current Performance (per 90 min)", expanded=True):
        perf_cols = st.columns(3)
        perf_cols[0].metric(
            "Playing Time",
            f"{row.minutes_per_90:.1f} min",
            help="Average minutes played per 90-minute match"
        )
        perf_cols[1].metric(
            "Goals",
            f"{row.goals_per_90:.2f}",
            help="Average goals scored per 90 minutes"
        )
        perf_cols[2].metric(
            "Assists",
            f"{row.assists_per_90:.2f}",
            help="Average assists per 90 minutes"
        )
    
    with st.expander("Performance Change Trends", expanded=True):
        trend_cols = st.columns(3)
        # 檢查欄位是否存在，如果不存在則顯示 N/A
        try:
            val = getattr(row, 'delta_goals_per_90', None)
            if val is not None and not pd.isna(val):
                trend_cols[0].metric(
                    "Goals Change",
                    f"{val:+.2f}",
                    help="Change in goals per 90 from previous period"
                )
            else:
                trend_cols[0].metric(
                    "Goals Change",
                    "N/A",
                    help="Data not available"
                )
        except:
            trend_cols[0].metric(
                "Goals Change",
                "N/A",
                help="Data not available"
            )
        # 檢查欄位是否存在，如果不存在則顯示 N/A
        try:
            val = getattr(row, 'delta_minutes_per_90', None)
            if val is not None and not pd.isna(val):
                trend_cols[1].metric(
                    "Time Change",
                    f"{val:+.1f} min",
                    help="Change in playing time from previous period"
                )
            else:
                trend_cols[1].metric(
                    "Time Change",
                    "N/A",
                    help="Data not available"
                )
        except:
            trend_cols[1].metric(
                "Time Change",
                "N/A",
                help="Data not available"
            )
    
    with st.expander("Market Value Dynamics", expanded=True):
        # 檢查欄位是否存在，如果不存在則顯示 N/A
        try:
            val = getattr(row, 'mv_momentum_12m', None)
            if val is not None and not pd.isna(val):
                st.metric(
                    "12-Month Momentum",
                    f"{val:+.2f}",
                    help="Market value change trend over past 12 months"
                )
            else:
                st.metric(
                    "12-Month Momentum",
                    "N/A",
                    help="Data not available"
                )
        except:
            st.metric(
                "12-Month Momentum",
                "N/A",
                help="Data not available"
            )

    # Render development curve analysis (compared to aging curve)
    _render_development_section(row)

    # Market value history chart
    st.markdown("### Market Value History")
    # 檢查欄位是否存在，如果不存在則顯示訊息
    try:
        val = getattr(row, 'mv_history', None)
        if val is not None and val != '' and str(val).strip() != '':
            st.altair_chart(_plot_mv_history(val), use_container_width=True)
        else:
            st.info("Market value history data not available")
    except:
        st.info("Market value history data not available")

    # SHAP feature importance analysis
    st.markdown("### AI Feature Importance Analysis")
    st.caption("SHAP values show the impact of each metric on predictions (positive values increase predictions, negative values decrease them)")
    
    shap_cols = st.columns(2)
    with shap_cols[0]:
        # 檢查欄位是否存在，如果不存在則顯示訊息
        try:
            val = getattr(row, 'reg_shap_top_features', None)
            if val is not None and val != '' and str(val).strip() != '':
                _render_feature_tags(val, "Market Value Key Factors")
            else:
                st.info("Market value key factors not available")
        except:
            st.info("Market value key factors not available")
    with shap_cols[1]:
        # 檢查欄位是否存在，如果不存在則顯示訊息
        try:
            val = getattr(row, 'clf_shap_top_features', None)
            if val is not None and val != '' and str(val).strip() != '':
                _render_feature_tags(val, "Breakout Probability Key Factors")
            else:
                st.info("Breakout probability key factors not available")
        except:
            st.info("Breakout probability key factors not available")


def _render_development_section(row: pd.Series) -> None:
    """
    Display player development metrics (compared to aging curve).
    
    This function analyzes whether the player performs above the average for their age group:
    - expected_value_million: Expected market value based on aging curve
    - valuation_above_curve: Difference between actual and expected market value
    - aging_score: Aging score (positive values indicate performance above age expectations)
    """
    # Check if required development metric fields exist
    required = [
        "expected_value_million",
        "valuation_above_curve",
        "expected_ga_per_90",
        "performance_above_curve",
        "expected_minutes_per_90",
        "minutes_above_curve",
        "aging_score",
        "development_tier",
    ]
    if not set(required).issubset(row.index):
        # If data is incomplete, display info message
        st.info("Development curve analysis will appear once the modeling module is connected")
        return

    # Display development curve analysis title and description
    st.markdown("### Development Curve Analysis")
    st.caption("Compare player performance with standard curve for same age group to assess if exceeding expectations")
    # Insert Aging Score metric directly below caption
    col1, col2 = st.columns(2)
    col1.metric(
        "Aging Score", 
        f"{row.aging_score:+.2f}",
        help="Composite score, positive values indicate performance above age expectations"
    )

    # Load aging curves data (this will be used by both the chart and comparison)
    curves_df = _load_aging_curves()
    
    if curves_df is not None:
        # Filter for player's position
        position_curves = curves_df[curves_df["sub_position"] == row.sub_position].copy()
        
        # Get the expected values for player's age from the curve
        # 確保使用整數年齡進行比較
        player_age_int = int(row.age)
        player_age_curve = position_curves[position_curves["age"] == player_age_int]
        
        if not player_age_curve.empty:
            # Use values from aging curve (excluding market value)
            expected_ga = player_age_curve["expected_ga_per_90"].iloc[0]
            expected_minutes = player_age_curve["expected_minutes_per_90"].iloc[0]
        else:
            # Fallback to row values if age not found in curve
            expected_ga = row.expected_ga_per_90
            expected_minutes = row.expected_minutes_per_90
    else:
        # Fallback to row values if curves not available
        expected_ga = row.expected_ga_per_90
        expected_minutes = row.expected_minutes_per_90
    
    # Display aging curves for player's position
    st.markdown("#### Position Aging Curves")
    st.caption(f"Aging curves for {row.sub_position} position at age {int(row.age)}. The diamond marker shows this player's current performance.")
    aging_chart = _plot_aging_curves(row.sub_position, int(row.age), row)
    st.altair_chart(aging_chart, use_container_width=True)
    
    # Display legend explanation
    _render_development_legend()
    
    # Display comparison chart - pass the expected values from aging curve (excluding market value)
    _render_development_comparison_chart(row, expected_ga, expected_minutes)

    # On-field performance comparison metrics
    st.markdown("#### On-Field Performance")
    perf_cols = st.columns(3)
    perf_cols[0].metric(
        "Curve Expected G+A/90", 
        f"{expected_ga:.2f}",
        help="Expected goals+assists for same age players (per 90 minutes)"
    )
    
    # Calculate actual difference using curve-based expected value
    if "ga_per_90" in row.index and pd.notna(row.ga_per_90):
        actual_ga = row.ga_per_90
    else:
        actual_ga = (row.goals_per_90 or 0) + (row.assists_per_90 or 0)
    ga_diff = actual_ga - expected_ga
    
    if "minutes_per_90" in row.index and pd.notna(row.minutes_per_90):
        actual_minutes = row.minutes_per_90
    else:
        actual_minutes = row.get("minutes_per_90", 0) or 0
    minutes_diff = actual_minutes - expected_minutes
    
    perf_cols[1].metric(
        "G+A Above Curve", 
        f"{ga_diff:+.2f}",
        help="Difference between actual and expected G+A"
    )
    perf_cols[2].metric(
        "Minutes Above Curve", 
        f"{minutes_diff:+.1f} mins",
        help="Difference between actual and expected playing time"
    )

    # Development stage summary
    years_since_peak = row.get("years_since_peak_value", 0)
    tier_label = str(row.get("development_tier", "unknown")).title()
    
    st.markdown(
        f"""
        <div class='info-box' style='margin-top: 1rem;'>
            <p style='margin: 0; font-size: 1rem;'>
                <strong>Development Stage:</strong> {tier_label}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_development_legend() -> None:
    """
    Display chart legend explaining the meaning of various symbols.
    
    Uses three visual elements:
    - Light red bar: Actual values
    - Deep red line: Standard curve benchmark
    - Pale red diamond: AI predicted values
    """
    # Use HTML to create clean visualized legend with proper alignment (red theme)
    legend_html = f"""
    <div style="margin-bottom: 1rem; 
                border: 1px solid var(--border-color);
                border-left: 4px solid #ef4444;
                padding: 0.875rem 1rem; 
                border-radius: 0.5rem;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 1.5rem;
                font-size: 0.9rem;
                font-weight: 500;">
        <strong style="margin-right: 0.5rem;">Chart Legend:</strong>
        <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <span style="width: 18px; height: 14px; background: {ACTUAL_COLOR}; border-radius: 3px; flex-shrink: 0;"></span>
            <span style="white-space: nowrap;">Actual Value (Bar)</span>
        </span>
        <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <span style="width: 22px; height: 3px; background: {BENCHMARK_COLOR}; border-radius: 2px; flex-shrink: 0;"></span>
            <span style="white-space: nowrap;">Curve Benchmark (Line)</span>
        </span>
        <span style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <span style="width: 14px; height: 14px; background: {PREDICTED_COLOR}; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%); flex-shrink: 0;"></span>
            <span style="white-space: nowrap;">AI Predicted (Diamond)</span>
        </span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


def _render_development_comparison_chart(
    row: pd.Series,
    expected_ga: float, 
    expected_minutes: float
) -> None:
    """
    Render bullet charts showing actual vs benchmark values comparison.
    
    Creates two comparison charts:
    1. Goals+assists comparison (actual vs standard curve)
    2. Playing time comparison (actual vs standard curve)
    
    Args:
        row: Player data row
        expected_ga: Expected goals+assists per 90 from aging curve
        expected_minutes: Expected minutes per 90 from aging curve
    """
    
    # Use ga_per_90 from development_outputs (calculated as total G+A / total minutes * 90)
    # This matches the calculation method in player_development_analysis.ipynb
    if "ga_per_90" in row.index and pd.notna(row.ga_per_90):
        ga_actual = row.ga_per_90
    else:
        ga_actual = (row.goals_per_90 or 0) + (row.assists_per_90 or 0)
    
    # Use minutes_per_90 from development_outputs
    # This represents total minutes / number of appearances
    if "minutes_per_90" in row.index and pd.notna(row.minutes_per_90):
        minutes_actual = row.minutes_per_90
    else:
        minutes_actual = row.get("minutes_per_90", 0) or 0
    
    # Use the expected values passed from the aging curve (NOT from row)
    ga_expected = expected_ga
    minutes_expected = expected_minutes

    # Create two bullet charts with English labels (excluding market value)
    charts = [
        _bullet_chart(
            metric="Goals+Assists / 90min",
            actual=ga_actual,
            benchmark=ga_expected,
            x_domain=(0, 2),
            actual_label="Actual Performance",
            benchmark_label="Curve Benchmark",
        ),
        _bullet_chart(
            metric="Playing Time / 90min",
            actual=minutes_actual,
            benchmark=minutes_expected,
            x_domain=(0, 90),
            actual_label="Actual Time",
            benchmark_label="Curve Benchmark",
        ),
    ]
    
    # Vertically concatenate two charts and configure styling
    # vconcat: Vertically connect multiple charts
    # resolve_scale(x="independent"): Let each chart use independent X-axis range
    chart = alt.vconcat(*charts).resolve_scale(x="independent").configure_axis(
        labelFontSize=12,  # Axis label font size
        titleFontSize=13,  # Axis title font size
    )
    st.altair_chart(chart, use_container_width=True)


def _bullet_chart(
    metric: str,
    actual: float,
    benchmark: float,
    predicted: float | None = None,
    x_domain: tuple[float, float] | None = None,
    actual_label: str | None = None,
    benchmark_label: str | None = None,
    predicted_label: str | None = None,
) -> alt.Chart:
    """Create a bullet-style chart with benchmark rule, actual bar, and optional predicted marker."""

    actual_label = actual_label or "Actual"
    benchmark_label = benchmark_label or "Curve Benchmark"
    predicted_label = predicted_label or "Predicted"

    bar_df = pd.DataFrame({"metric": [metric], "value": [actual], "label": [actual_label]})
    rule_df = pd.DataFrame({"metric": [metric], "value": [benchmark], "label": [benchmark_label]})
    x_kwargs = {"title": None}
    if x_domain is not None:
        x_kwargs["scale"] = alt.Scale(domain=list(x_domain))

    components = [
        alt.Chart(bar_df)
        .mark_bar(color=ACTUAL_COLOR, size=20)
        .encode(
            x=alt.X("value:Q", **x_kwargs),
            y=alt.Y("metric:N", title=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("label:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        ),
        alt.Chart(rule_df)
        .mark_rule(color=BENCHMARK_COLOR, strokeWidth=6.5)
        .encode(
            x=alt.X("value:Q", **x_kwargs),
            y=alt.Y("metric:N", title=None),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("label:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        ),
    ]
    if predicted is not None:
        point_df = pd.DataFrame({"metric": [metric], "value": [predicted], "label": [predicted_label]})
        components.append(
            alt.Chart(point_df)
            .mark_point(shape="diamond", size=120, color=PREDICTED_COLOR)
            .encode(
                x=alt.X("value:Q", **x_kwargs),
                y=alt.Y("metric:N", title=None),
                tooltip=[
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("label:N", title="Series"),
                    alt.Tooltip("value:Q", title="Value", format=".2f"),
                ],
            )
        )
    return alt.layer(*components).properties(height=80)
