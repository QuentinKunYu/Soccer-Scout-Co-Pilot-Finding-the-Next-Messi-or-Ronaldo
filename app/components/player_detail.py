"""Detail view rendered when a user selects a player."""

from __future__ import annotations

from typing import List, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from src.llm.schema import FeatureImportance

# Unified color palette for development visuals (shades of red).
ACTUAL_COLOR = "#f87171"      # light red for actual metrics
BENCHMARK_COLOR = "#dc2626"   # deeper red for curve benchmarks
PREDICTED_COLOR = "#fecaca"   # pale red for Chart Legend:


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
            st.image(row.img_url, width=150)
        else:
            # Fallback if no image is available
            st.markdown(
                """
                <div style="width: 150px; height: 150px; 
                            background-color: #f3f4f6; 
                            border-radius: 50%; 
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
        st.markdown(f"## {row.player_name}")
        st.caption(f"{row.sub_position}")
    
    st.markdown("---")
    
    # Core metrics card section - using clear labels
    st.markdown("### Core Metrics")
    cols = st.columns(4)
    
    # metric: Streamlit's metric display component, automatically formats values and highlights them
    cols[0].metric("Age", f"{row.age} years")
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
        trend_cols[0].metric(
            "Goals Change",
            f"{row.delta_goals_per_90:+.2f}",
            help="Change in goals per 90 from previous period"
        )
        trend_cols[1].metric(
            "Time Change",
            f"{row.delta_minutes_per_90:+.1f} min",
            help="Change in playing time from previous period"
        )
        trend_cols[2].metric(
            "Avg Rating",
            f"{row.rating_mean:.2f}",
            help="Average match rating"
        )
    
    with st.expander("Market Value Dynamics", expanded=True):
        st.metric(
            "12-Month Momentum",
            f"{row.mv_momentum_12m:+.2f}",
            help="Market value change trend over past 12 months"
        )

    # Render development curve analysis (compared to aging curve)
    _render_development_section(row)

    # Market value history chart
    st.markdown("### Market Value History")
    st.altair_chart(_plot_mv_history(row.mv_history), use_container_width=True)

    # SHAP feature importance analysis
    st.markdown("### AI Feature Importance Analysis")
    st.caption("SHAP values show the impact of each metric on predictions (positive values increase predictions, negative values decrease them)")
    
    shap_cols = st.columns(2)
    with shap_cols[0]:
        _render_feature_tags(row.reg_shap_top_features, "Market Value Key Factors")
    with shap_cols[1]:
        _render_feature_tags(row.clf_shap_top_features, "Breakout Probability Key Factors")


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
    
    # Display legend explanation
    _render_development_legend()
    
    # Display comparison chart
    _render_development_comparison_chart(row)

    # Market value comparison metrics
    st.markdown("#### Market Value Performance")
    dev_cols = st.columns(3)
    dev_cols[0].metric(
        "Curve Expected MV", 
        f"€{row.expected_value_million:.2f}M",
        help="Average market value for similar players based on age and position"
    )
    dev_cols[1].metric(
        "Above Curve", 
        f"€{row.valuation_above_curve:+.2f}M",
        help="Difference between actual and expected MV (positive values indicate exceeding expectations)"
    )
    dev_cols[2].metric(
        "Aging Score", 
        f"{row.aging_score:+.2f}",
        help="Composite score, positive values indicate performance above age expectations"
    )

    # On-field performance comparison metrics
    st.markdown("#### On-Field Performance")
    perf_cols = st.columns(3)
    perf_cols[0].metric(
        "Curve Expected G+A/90", 
        f"{row.expected_ga_per_90:.2f}",
        help="Expected goals+assists for same age players (per 90 minutes)"
    )
    perf_cols[1].metric(
        "G+A Above Curve", 
        f"{row.performance_above_curve:+.2f}",
        help="Difference between actual and expected G+A"
    )
    perf_cols[2].metric(
        "Minutes Above Curve", 
        f"{row.minutes_above_curve:+.1f} mins",
        help="Difference between actual and expected playing time"
    )

    # Development stage summary
    years_since_peak = row.get("years_since_peak_value", 0)
    valuation_slope = row.get("valuation_slope_24m", 0.0)
    tier_label = str(row.get("development_tier", "unknown")).title()
    
    st.markdown(
        f"""
        <div class='info-box' style='margin-top: 1rem;'>
            <p style='margin: 0; font-size: 1rem;'>
                <strong>Development Stage:</strong> {tier_label}
                <strong style='margin-left: 2rem;'>Peak Age:</strong> {row.get('peak_age', 'N/A')} years
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 1rem;'>
                <strong>Years Since Peak:</strong> {years_since_peak:.1f} years
                <strong style='margin-left: 2rem;'>24m Valuation Slope:</strong> <span style='color: {"#10b981" if valuation_slope >= 0 else "#ef4444"}; font-weight: 600;'>{valuation_slope:+.2f}</span>
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


def _render_development_comparison_chart(row: pd.Series) -> None:
    """
    Render bullet charts showing actual vs benchmark values comparison.
    
    Creates three comparison charts:
    1. Market value comparison (current vs expected vs predicted)
    2. Goals+assists comparison (actual vs standard curve)
    3. Playing time comparison (actual vs standard curve)
    """
    # Calculate actual and expected values for each metric
    actual_mv = row.current_market_value / 1_000_000  # Current market value (millions of euros)
    pred_mv = row.mv_pred_1y / 1_000_000  # AI predicted market value in one year
    benchmark_mv = row.expected_value_million  # Expected market value based on aging curve
    
    ga_actual = (row.goals_per_90 or 0) + (row.assists_per_90 or 0)  # Actual G+A
    ga_expected = row.expected_ga_per_90  # Expected G+A from standard curve
    
    minutes_actual = row.minutes_per_90  # Actual playing time
    minutes_expected = row.expected_minutes_per_90  # Expected playing time

    # Create three bullet charts with English labels
    charts = [
        _bullet_chart(
            metric="Market Value (€M)",
            actual=actual_mv,
            benchmark=benchmark_mv,
            predicted=pred_mv,
            x_domain=(0, 200),
            actual_label="Current MV",
            benchmark_label="Curve Benchmark",
            predicted_label="AI Predicted",
        ),
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
    
    # Vertically concatenate three charts and configure styling
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
        .mark_rule(color=BENCHMARK_COLOR, strokeWidth=7.5)
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
