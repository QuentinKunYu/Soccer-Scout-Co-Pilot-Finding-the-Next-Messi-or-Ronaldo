"""
I/O utility functions for loading and parsing player recommendation data.

This module provides helper functions for loading player recommendation CSV files
and parsing JSON-encoded columns that contain nested data structures (feature lists,
market value history, etc.).

The CSV format stores complex data structures as JSON strings, which need to be
parsed back into Python objects (lists/dicts) for use in the Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import pandas as pd


def _safe_json_load(value: Any) -> Any:
    """
    Safely parse JSON string to Python object, with fallback handling.
    
    The CSV stores nested structures (feature lists, MV history) as JSON strings.
    When Streamlit reads the file, we need Python objects (list/dict) for plotting
    and data manipulation. This helper centralizes error handling to avoid repetition.
    
    Args:
        value: Input value that may be a JSON string, already a dict/list, or NaN
        
    Returns:
        Parsed Python object (dict/list) if JSON string, original value if already
        a dict/list, or empty list if NaN or parsing fails
        
    Examples:
        >>> _safe_json_load('[{"feature": "goals", "value": 0.5}]')
        [{'feature': 'goals', 'value': 0.5}]
        >>> _safe_json_load(None)
        []
        >>> _safe_json_load({'already': 'parsed'})
        {'already': 'parsed'}
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
    """
    Load player recommendations CSV and parse JSON-encoded columns.
    
    This function loads a CSV file containing player recommendations and automatically
    parses JSON-encoded columns into Python objects. The function is designed to work
    with the player_recommendations.csv format, which stores complex nested data
    structures (SHAP features, market value history) as JSON strings.
    
    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file containing player recommendations. Can be either a
        string or Path object. The function signature is kept generic to allow
        pointing to different recommendation files (mock data, production data, etc.)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with player recommendations. JSON columns are parsed into
        Python objects (lists/dicts) for easier manipulation.
        
    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist
        
    Examples
    --------
    >>> df = load_player_recommendations("data/processed/player_recommendations.csv")
    >>> print(df.columns)
    Index(['player_name', 'age', 'breakout_prob', 'reg_shap_top_features', ...])
    >>> print(type(df['reg_shap_top_features'].iloc[0]))
    <class 'list'>
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

