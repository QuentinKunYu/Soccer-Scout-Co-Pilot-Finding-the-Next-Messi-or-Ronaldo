"""Simple wrapper that either calls a real LLM or emits a deterministic stub."""

from __future__ import annotations

import json
import os
from typing import Dict, Any

try:  # The OpenAI SDK might not be installed in the hackathon environment.
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency.
    OpenAI = None

from .prompts import BASE_SYSTEM_PROMPT, build_player_prompt
from .schema import PlayerLLMInput


def _json_schema_spec() -> Dict[str, Any]:
    """Shared JSON schema so both Responses API and Chat API get the same contract."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "scouting_report",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "undervaluation_reason": {"type": "string"},
                    "breakout_reason": {"type": "string"},
                    "risk_factors": {"type": "string"},
                    "one_line_recommendation": {"type": "string"},
                },
                "required": [
                    "summary",
                    "undervaluation_reason",
                    "breakout_reason",
                    "risk_factors",
                    "one_line_recommendation",
                ],
                "additionalProperties": False,
            },
        },
    }


class LLMClient:
    """Dispatches prompt/response cycles for the Streamlit app."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        
        # Try to create OpenAI client with error handling for compatibility issues
        try:
            self._client = OpenAI(api_key=self.api_key) if (self.api_key and OpenAI) else None
        except Exception as e:
            print(f"⚠️  Failed to initialize OpenAI client: {e}")
            print("📝 Will use stub responses instead")
            self._client = None

    def generate_report(self, player: PlayerLLMInput) -> Dict[str, str]:
        """Return a structured report for the provided player."""

        payload = player.to_prompt_payload()
        prompt = build_player_prompt(payload)

        if self._client is None:
            print("No API key found, building stub response")
            return self._build_stub_response(player)

        print("API key found, calling OpenAI API...")
        schema_spec = _json_schema_spec()
        
        # Try to use the newer responses API first (available in OpenAI SDK >= 1.50)
        if hasattr(self._client, "responses"):
            try:
                print("Using Responses API...")
                response = self._client.responses.create(
                    model=self.model,
                    temperature=self.temperature,
                    input=[
                        {"role": "system", "content": BASE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema_spec,
                )
                raw_text = response.output[0].content[0].text  # type: ignore[index]
                return json.loads(raw_text)
            except Exception as e:
                print(f"⚠️  Responses API failed: {e}, falling back to chat completions")

        # Fall back to chat completions API (works with all OpenAI SDK versions >= 1.0)
        try:
            print("Using Chat Completions API...")
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=schema_spec,
            )
            content = completion.choices[0].message.content  # type: ignore[index]
            if isinstance(content, list):
                # Some SDK versions return a list of content blocks; join them together.
                text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
            else:
                text = str(content or "{}")
            return json.loads(text)
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            print("📝 Falling back to stub response")
            return self._build_stub_response(player)

    def _build_stub_response(self, player: PlayerLLMInput) -> Dict[str, str]:
        """Return a deterministic summary so the UI works without API keys."""

        stats = player.key_stats
        current = player.current_market_value / 1_000_000
        predicted = player.mv_pred_1y / 1_000_000
        growth_pct = player.y_growth_pred * 100
        breakout_pct = player.breakout_prob * 100

        summary = (
            f"{player.player_name} ({player.age}, {player.position}) plays for {player.club_name} "
            f"in {player.league_name}. Current MV is €{current:.1f}M with a model projection of "
            f"€{predicted:.1f}M (+{growth_pct:.0f}%)."
        )
        undervaluation = (
            f"High undervalued score ({player.undervalued_score/1_000_000:.1f}M) is driven by {stats.goals_per_90:.2f} "
            f"goals/90 and {stats.mv_momentum_12m:.2f} MV momentum."
        )
        breakout = (
            f"Breakout probability sits at {breakout_pct:.0f}% thanks to {stats.delta_goals_per_90:.2f} delta goals/90 "
            f"and consistent {stats.minutes_per_90:.1f} minutes."
        )
        risk = (
            "Monitor workload and finishing volatility; SHAP factors still rely on a limited sample size."
        )
        recommendation = (
            f"Recommended as a {'high-upside' if breakout_pct > 60 else 'steady'} acquisition for ambitious clubs."
        )
        return {
            "summary": summary,
            "undervaluation_reason": undervaluation,
            "breakout_reason": breakout,
            "risk_factors": risk,
            "one_line_recommendation": recommendation,
        }
