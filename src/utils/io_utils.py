"""Small IO helpers for loading mock recommendation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pandas as pd


def _safe_json_load(value: Any) -> Any:
    """Return parsed JSON when possible, otherwise keep the raw value.

    The CSV stores nested structures (feature lists, MV history) as JSON strings.
    When Streamlit reads the file we want Python objects (list/dict) for plotting,
    so the helper centralizes the error handling instead of repeating it.
    """

    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return []
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def load_player_recommendations(csv_path: str | Path) -> pd.DataFrame:
    """Load the mock recommendations CSV and parse JSON columns.

    Parameters
    ----------
    csv_path: str | Path
        Location of the CSV file described in the roadmap. We keep the
        signature generic so the same function can later point to the real
        `player_recommendations.parquet` once it exists.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Recommendation file not found: {path}")

    df = pd.read_csv(path)
    json_columns: List[str] = [
        "reg_shap_top_features",
        "clf_shap_top_features",
        "mv_history",
    ]
    for column in json_columns:
        df[column] = df[column].apply(_safe_json_load)
    return df

