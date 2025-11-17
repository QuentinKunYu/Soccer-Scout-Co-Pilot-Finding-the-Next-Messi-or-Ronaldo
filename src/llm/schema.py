"""
Data schema and containers for LLM prompt generation.

This module defines typed data structures (dataclasses) that ensure consistency
between the player recommendation data and the LLM prompt payload format.

The module provides:
- FeatureImportance: Container for SHAP feature importance values
- KeyStats: Normalized statistics block for player performance metrics
- PlayerLLMInput: Complete player data structure for LLM analysis
- DevelopmentSnapshot: Aging curve and development metrics

These containers handle data validation, type conversion, and serialization
for safe transmission to the LLM API.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FeatureImportance:
    """Simple container for feature importance rows.

    We keep it lean because the Streamlit UI only needs a label and value,
    but wrapping it in a dataclass clarifies the structure for the LLM prompt.
    """

    feature: str
    shap_value: float


@dataclass
class KeyStats:
    """Normalized stat block that the prompt can easily reuse."""

    minutes_per_90: float
    goals_per_90: float
    assists_per_90: float
    delta_goals_per_90: float
    delta_minutes_per_90: float
    rating_mean: float
    mv_momentum_12m: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "KeyStats":
        """Build the stat block straight from the dataframe row."""

        columns = [
            "minutes_per_90",
            "goals_per_90",
            "assists_per_90",
            "delta_goals_per_90",
            "delta_minutes_per_90",
            "rating_mean",
            "mv_momentum_12m",
        ]
        data = {col: float(row.get(col, 0.0) or 0.0) for col in columns}
        return cls(**data)


@dataclass
class PlayerLLMInput:
    """Row from the recommendation table converted into prompt-friendly JSON."""

    player_name: str
    age: int
    position: str
    club_name: str
    league_name: str
    current_market_value: float
    mv_pred_1y: float
    y_growth_pred: float
    breakout_prob: float
    undervalued_score: float
    key_stats: KeyStats
    reg_shap_top_features: List[FeatureImportance]
    clf_shap_top_features: List[FeatureImportance]
    development: Optional["DevelopmentSnapshot"]

    @classmethod
    def from_row(cls, row: pd.Series) -> "PlayerLLMInput":
        """Map a pandas row to the nested structure required by the LLM."""

        def _parse_feature_list(values: Any) -> List[FeatureImportance]:
            items = values if isinstance(values, list) else []
            return [
                FeatureImportance(feature=str(item.get("feature")), shap_value=float(item.get("shap_value", 0.0)))
                for item in items
            ]

        numeric = {
            "current_market_value": float(row.get("current_market_value", 0.0) or 0.0),
            "mv_pred_1y": float(row.get("mv_pred_1y", 0.0) or 0.0),
            "y_growth_pred": float(row.get("y_growth_pred", 0.0) or 0.0),
            "breakout_prob": float(row.get("breakout_prob", 0.0) or 0.0),
            "undervalued_score": float(row.get("undervalued_score", 0.0) or 0.0),
        }

        return cls(
            player_name=str(row.get("player_name", "")),
            age=int(row.get("age", 0) or 0),
            position=str(row.get("sub_position", row.get("position", ""))),  # Use sub_position, fallback to position
            club_name=str(row.get("club_name", "")),
            league_name=str(row.get("league_name", "")),
            key_stats=KeyStats.from_row(row),
            reg_shap_top_features=_parse_feature_list(row.get("reg_shap_top_features")),
            clf_shap_top_features=_parse_feature_list(row.get("clf_shap_top_features")),
            development=DevelopmentSnapshot.from_row(row),
            **numeric,
        )

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Return a JSON serializable dict for the LLM prompt."""

        payload = asdict(self)
        payload["key_stats"] = asdict(self.key_stats)
        if self.development is not None:
            payload["development"] = asdict(self.development)
        else:
            payload.pop("development", None)
        return payload


@dataclass
class DevelopmentSnapshot:
    """Container for aging-curve signals."""

    expected_value_million: float
    expected_ga_per_90: float
    expected_minutes_per_90: float
    valuation_above_curve: float
    performance_above_curve: float
    minutes_above_curve: float
    aging_score: float
    development_tier: str
    peak_age: float
    years_since_peak_value: float

    @classmethod
    def from_row(cls, row: pd.Series) -> Optional["DevelopmentSnapshot"]:
        """Return an instance if the underlying columns exist."""

        required_cols = {
            "expected_value_million",
            "expected_ga_per_90",
            "expected_minutes_per_90",
            "valuation_above_curve",
            "performance_above_curve",
            "minutes_above_curve",
            "aging_score",
            "development_tier",
            "peak_age",
            "years_since_peak_value",
        }
        if not set(required_cols).issubset(row.index):
            return None

        return cls(
            expected_value_million=float(row.get("expected_value_million", 0.0) or 0.0),
            expected_ga_per_90=float(row.get("expected_ga_per_90", 0.0) or 0.0),
            expected_minutes_per_90=float(row.get("expected_minutes_per_90", 0.0) or 0.0),
            valuation_above_curve=float(row.get("valuation_above_curve", 0.0) or 0.0),
            performance_above_curve=float(row.get("performance_above_curve", 0.0) or 0.0),
            minutes_above_curve=float(row.get("minutes_above_curve", 0.0) or 0.0),
            aging_score=float(row.get("aging_score", 0.0) or 0.0),
            development_tier=str(row.get("development_tier", "unknown")),
            peak_age=float(row.get("peak_age", 0.0) or 0.0),
            years_since_peak_value=float(row.get("years_since_peak_value", 0.0) or 0.0),
        )
