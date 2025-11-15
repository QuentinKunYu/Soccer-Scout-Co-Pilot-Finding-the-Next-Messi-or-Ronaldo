"""Typed containers that keep the prompt payload consistent."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

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
            position=str(row.get("position", "")),
            club_name=str(row.get("club_name", "")),
            league_name=str(row.get("league_name", "")),
            key_stats=KeyStats.from_row(row),
            reg_shap_top_features=_parse_feature_list(row.get("reg_shap_top_features")),
            clf_shap_top_features=_parse_feature_list(row.get("clf_shap_top_features")),
            **numeric,
        )

    def to_prompt_payload(self) -> Dict[str, Any]:
        """Return a JSON serializable dict for the LLM prompt."""

        payload = asdict(self)
        payload["key_stats"] = asdict(self.key_stats)
        return payload

