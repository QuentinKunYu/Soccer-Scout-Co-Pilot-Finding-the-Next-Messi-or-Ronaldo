"""Prompt builders dedicated to the scouting assistant workflow."""

from __future__ import annotations

import json
from typing import Dict

BASE_SYSTEM_PROMPT = (
    "You are a football scouting analyst that can interpret model outputs. "
    "Explain upside, risks, and market context clearly and concisely."
)


def build_player_prompt(player_payload: Dict) -> str:
    """Return the formatted instruction block for a single player.

    The function keeps Streamlit lean by separating text generation from the UI
    layer. The payload is already sanitized by ``PlayerLLMInput`` so we can
    safely embed it as JSON for the downstream language model.
    """

    instruction = (
        "Given the JSON payload, cover the following points in 5-6 sentences: "
        "1) undervaluation rationale, 2) breakout probability drivers, "
        "3) concrete stats references, 4) key SHAP drivers, and 5) a one-line recommendation."
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

