"""Prompt builders dedicated to the scouting assistant workflow."""

from __future__ import annotations

import json
from typing import Dict

BASE_SYSTEM_PROMPT = (
    "You are an expert football scouting analyst specializing in player development trajectory analysis and market valuation. "
    "You understand aging curves, player development stages, and can translate statistical data into actionable scouting insights. "
    "Your analysis should explain not just what the numbers say, but WHY they matter for talent identification and investment decisions. "
    "Focus on: 1) Player development stage and trajectory, 2) Performance relative to age expectations (aging curve), "
    "3) Market value inefficiencies, 4) Breakout potential drivers, and 5) Risk factors. "
    "Provide clear, concise, and decision-oriented analysis."
)


def build_player_prompt(player_payload: Dict) -> str:
    """Return the formatted instruction block for a single player.

    The function keeps Streamlit lean by separating text generation from the UI
    layer. The payload is already sanitized by ``PlayerLLMInput`` so we can
    safely embed it as JSON for the downstream language model.
    """

    instruction = (
        "Analyze the player data and provide a comprehensive scouting report covering:\n\n"
        
        "1. DEVELOPMENT TRAJECTORY & AGING CURVE ANALYSIS:\n"
        "   - Interpret the 'development_tier' (Rising Star/Aging Well/Peak/Declining) - explain what this stage means for investment timing\n"
        "   - Analyze 'aging_score' (positive = performing above age expectations, negative = below) - WHY this matters for future value\n"
        "   - Explain 'valuation_above_curve' (€M above/below expected MV for age) - is market underpricing/overpricing this player?\n"
        "   - Assess 'performance_above_curve' (G+A above expected) and 'minutes_above_curve' - is on-field output exceeding age norms?\n"
        "   - Evaluate 'valuation_slope_24m' (recent MV trajectory) - is momentum accelerating or declining?\n"
        "   - Consider 'years_since_peak_value' and 'peak_age' - where in career arc? Windows of opportunity?\n\n"
        
        "2. MARKET VALUE INEFFICIENCY:\n"
        "   - Why is 'undervalued_score' high/low? Connect to aging curve position and performance metrics\n"
        "   - Compare 'current_market_value' vs 'mv_pred_1y' - expected appreciation and WHY\n"
        "   - Reference 'expected_value_million' from aging curve - is market mispricing age-adjusted value?\n\n"
        
        "3. BREAKOUT PROBABILITY DRIVERS:\n"
        "   - Explain 'breakout_prob' % - what factors drive this (age, development stage, performance trends)?\n"
        "   - Connect to 'delta_goals_per_90', 'delta_minutes_per_90' - improving trajectory?\n"
        "   - How do SHAP features (clf_shap_top_features) reveal breakout catalysts?\n\n"
        
        "4. STATISTICAL EVIDENCE:\n"
        "   - Reference concrete stats: minutes_per_90, goals_per_90, assists_per_90, rating_mean\n"
        "   - Cite key SHAP drivers (reg_shap_top_features for MV, clf_shap_top_features for breakout)\n"
        "   - Connect stats to development stage - are numbers typical or exceptional for this age/tier?\n\n"
        
        "5. RISK ASSESSMENT:\n"
        "   - Age-related risks: if past peak or declining tier, quantify downside\n"
        "   - Development risks: if young Rising Star, consider volatility and adjustment needs\n"
        "   - Market timing risks: valuation slope trends, momentum sustainability\n\n"
        
        "6. ACTIONABLE RECOMMENDATION:\n"
        "   - One clear sentence: buy/monitor/pass, with specific timing considerations based on development stage\n\n"
        
        "Format your response to match the JSON schema. Be specific, quantitative, and explain the MEANING behind metrics, "
        "not just their values. Connect aging curve insights to investment decisions."
    )
    return (
        f"{instruction}\n\n"
        "Player data:\n"
        f"```json\n{json.dumps(player_payload, indent=2)}\n```"
    )


def render_report_to_markdown(response_payload: Dict[str, str]) -> str:
    """Convert a structured LLM response into markdown for Streamlit."""

    lines = []
    for key, value in response_payload.items():
        pretty_key = key.replace("_", " ").title()
        lines.append(f"**{pretty_key}:** {value}")
    return "\n\n".join(lines)
