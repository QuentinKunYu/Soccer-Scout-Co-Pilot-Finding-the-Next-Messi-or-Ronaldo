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
                    "summary": {
                        "type": "string",
                        "description": "2-3 sentence overview including age, position, club, current MV, and predicted growth"
                    },
                    "development_analysis": {
                        "type": "string",
                        "description": "Comprehensive analysis of development stage, aging curve position, and career trajectory. "
                                     "Explain aging_score, development_tier, performance vs curve, and what these mean for future value"
                    },
                    "undervaluation_reason": {
                        "type": "string",
                        "description": "Why the player is undervalued, connecting market inefficiency to aging curve position and performance data"
                    },
                    "breakout_reason": {
                        "type": "string",
                        "description": "Drivers of breakout probability, with reference to development stage, performance trends, and SHAP features"
                    },
                    "risk_factors": {
                        "type": "string",
                        "description": "Age-related, development, and market timing risks. Quantify downside scenarios based on aging curve"
                    },
                    "one_line_recommendation": {
                        "type": "string",
                        "description": "Clear actionable recommendation with timing based on development stage"
                    },
                },
                "required": [
                    "summary",
                    "development_analysis",
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

    def __init__(self, model: str = "gpt-5.1-chat-latest", temperature: float = 0.2) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        # GPT-5.1 models only support default temperature (1.0)
        # Set to None for GPT-5.1, otherwise use provided value
        self.temperature = None if "gpt-5" in model else temperature
        
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
                # Build request params, only include temperature if not None
                request_params = {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": BASE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": schema_spec,
                }
                if self.temperature is not None:
                    request_params["temperature"] = self.temperature
                
                response = self._client.responses.create(**request_params)
                raw_text = response.output[0].content[0].text  # type: ignore[index]
                return json.loads(raw_text)
            except Exception as e:
                print(f"⚠️  Responses API failed: {e}, falling back to chat completions")

        # Fall back to chat completions API (works with all OpenAI SDK versions >= 1.0)
        try:
            print("Using Chat Completions API...")
            # Build request params, only include temperature if not None
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": BASE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "response_format": schema_spec,
            }
            if self.temperature is not None:
                request_params["temperature"] = self.temperature
            
            completion = self._client.chat.completions.create(**request_params)
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
            f"{player.player_name} ({player.age}, {player.position}) currently plays for {player.club_name} "
            f"in {player.league_name}. His current market value stands at €{current:.1f}M, with our AI model "
            f"projecting growth to €{predicted:.1f}M within one year, representing a {growth_pct:.0f}% appreciation."
        )
        
        # Comprehensive development analysis
        if player.development:
            dev = player.development
            tier_interpretation = {
                "rising star": "an ascending trajectory with significant upside potential",
                "aging well": "remarkable performance sustainability beyond typical age expectations",
                "peak": "optimal performance window with maximum current output",
                "declining": "post-peak phase requiring careful risk assessment"
            }.get(dev.development_tier.lower(), "active development")
            
            development_analysis = (
                f"Development Stage Analysis: {player.player_name} is classified as '{dev.development_tier}', indicating {tier_interpretation}. "
                f"His aging score of {dev.aging_score:+.2f} reveals he's performing {'above' if dev.aging_score > 0 else 'below'} age-adjusted expectations "
                f"({'significantly' if abs(dev.aging_score) > 0.5 else 'moderately'}). "
                f"Market valuation analysis shows he's €{abs(dev.valuation_above_curve):.2f}M {'above' if dev.valuation_above_curve > 0 else 'below'} "
                f"the expected €{dev.expected_value_million:.1f}M for players of his age and position. "
                f"His on-field output of {dev.performance_above_curve:+.2f} G+A above curve and {dev.minutes_above_curve:+.1f} minutes above expected "
                f"demonstrates {'exceptional' if dev.performance_above_curve > 0.15 else 'solid'} productivity. "
                f"With a 24-month valuation slope of {dev.valuation_slope_24m:+.2f}, his market trajectory is "
                f"{'accelerating' if dev.valuation_slope_24m > 0.3 else 'stable' if dev.valuation_slope_24m > 0 else 'declining'}. "
                f"At {dev.years_since_peak_value:.1f} years {'past' if dev.years_since_peak_value > 0 else 'before'} peak age ({dev.peak_age:.0f}), "
                f"the investment window {'requires urgency' if dev.years_since_peak_value > 2 else 'remains favorable' if dev.years_since_peak_value <= 0 else 'is still viable'}."
            )
        else:
            development_analysis = (
                "Development curve data unavailable. Analysis based on raw performance metrics and market signals only. "
                "Recommend additional aging curve modeling for comprehensive risk assessment."
            )
        
        undervaluation = (
            f"Undervaluation Assessment: The €{player.undervalued_score/1_000_000:.1f}M undervalued score signals "
            f"{'significant' if player.undervalued_score/1_000_000 > 3 else 'moderate'} market inefficiency. "
            f"Key drivers include {stats.goals_per_90:.2f} goals per 90 minutes and {stats.mv_momentum_12m:+.2f} market momentum over 12 months. "
            f"{'The aging curve analysis supports this, showing market underpricing of age-adjusted performance.' if player.development and player.development.valuation_above_curve < 0 else 'Market appears to be catching up to true value.'}"
        )
        
        breakout = (
            f"Breakout Probability Analysis: {breakout_pct:.0f}% breakout probability driven by "
            f"{stats.delta_goals_per_90:+.2f} delta goals per 90 (trend is {'positive' if stats.delta_goals_per_90 > 0 else 'concerning'}) "
            f"and consistent {stats.minutes_per_90:.1f} minutes per 90, indicating {'strong' if stats.minutes_per_90 > 70 else 'developing'} playing time security. "
            f"Rating of {stats.rating_mean:.2f} suggests {'elite' if stats.rating_mean > 7.5 else 'solid' if stats.rating_mean > 7.0 else 'average'} overall performance. "
            f"SHAP analysis would reveal specific catalysts, but trajectory indicators are {'promising' if breakout_pct > 60 else 'moderate'}."
        )
        
        # Enhanced risk assessment
        if player.development:
            dev = player.development
            age_risk = "Low" if dev.years_since_peak_value < 0 else "Moderate" if dev.years_since_peak_value < 2 else "High"
            tier_risk = {
                "rising star": "volatility and adjustment period",
                "aging well": "sustainability of exceptional aging pattern",
                "peak": "timing of eventual decline",
                "declining": "accelerated value depreciation"
            }.get(dev.development_tier.lower(), "unknown trajectory")
            
            risk = (
                f"Risk Factors: Age-related risk is {age_risk} given {dev.years_since_peak_value:.1f} years from peak. "
                f"Development stage risk centers on {tier_risk}. "
                f"Market timing risk {'elevated' if dev.valuation_slope_24m < 0 else 'manageable'} with {dev.valuation_slope_24m:+.2f} recent slope. "
                f"Monitor: workload sustainability at {stats.minutes_per_90:.1f} min/90, finishing consistency ({stats.delta_goals_per_90:+.2f} trend), "
                f"and aging curve adherence (current score: {dev.aging_score:+.2f})."
            )
        else:
            risk = (
                f"Risk Factors: Age {player.age} presents {'youth volatility' if player.age < 23 else 'decline risk' if player.age > 28 else 'moderate'} concerns. "
                f"Workload at {stats.minutes_per_90:.1f} min/90 and form volatility require monitoring. "
                f"Limited aging curve data increases uncertainty. SHAP factor sample size may be limited."
            )
        
        # Intelligent recommendation
        if breakout_pct > 70 and (not player.development or player.development.aging_score > 0.5):
            recommendation = "STRONG BUY: High-upside acquisition with favorable development stage and aging curve trajectory. Act quickly."
        elif breakout_pct > 50 and (not player.development or player.development.years_since_peak_value < 1):
            recommendation = "BUY: Solid investment opportunity with reasonable risk-return profile. Good timing window."
        elif player.development and player.development.years_since_peak_value > 2:
            recommendation = "MONITOR: Post-peak player requires careful due diligence on sustainability. Consider as short-term option only."
        else:
            recommendation = "EVALUATE: Further analysis recommended to assess development trajectory and market timing before commitment."
        
        return {
            "summary": summary,
            "development_analysis": development_analysis,
            "undervaluation_reason": undervaluation,
            "breakout_reason": breakout,
            "risk_factors": risk,
            "one_line_recommendation": recommendation,
        }
