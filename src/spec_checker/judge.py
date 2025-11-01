"""Judge evaluation module for compliance testing."""

import json
import logging
from typing import Any, Dict, List, Optional

from openrouter import OpenRouter

logger = logging.getLogger(__name__)


# Hardcoded judge models as per spec
JUDGE_MODELS = [
    "anthropic/claude-sonnet-4",
    "openai/o3-mini",
    "google/gemini-2.0-flash-exp",
]


class ComplianceJudge:
    """Evaluates model responses for spec compliance using multiple judges."""

    def __init__(self, api_key: str):
        """
        Initialize the compliance judge.

        Args:
            api_key: OpenRouter API key
        """
        self.client = OpenRouter(api_key=api_key)
        self.judge_models = JUDGE_MODELS

    def validate_api_key(self) -> bool:
        """
        Validate API key with a test request.

        Returns:
            True if API key is valid, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return response is not None
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def generate_response(
        self, model: str, scenario: str, max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate a response from the target model for a given scenario.

        Args:
            model: Model identifier (e.g., "anthropic/claude-sonnet-4")
            scenario: The scenario text to send to the model
            max_tokens: Maximum tokens for the response

        Returns:
            The model's response text, or None if request failed
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": scenario}],
                max_tokens=max_tokens,
            )
            if response.choices:
                return response.choices[0].message.content
            return None
        except Exception as e:
            logger.error(f"Failed to generate response with {model}: {e}")
            return None

    def evaluate_compliance(
        self,
        specification: str,
        scenario: str,
        model_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate compliance using all judge models.

        Args:
            specification: User's specification text
            scenario: The scenario text
            model_response: The target model's response to evaluate

        Returns:
            List of judgments from each judge model, each containing:
                - judge_model: Name of the judge model
                - reasoning: Judge's reasoning
                - judgment: "compliant", "non-compliant", or "ambiguous"
        """
        judgments = []

        for judge_model in self.judge_models:
            judgment = self._get_single_judgment(
                judge_model, specification, scenario, model_response
            )
            if judgment:
                judgment["judge_model"] = judge_model
                judgments.append(judgment)
            else:
                logger.warning(f"Failed to get judgment from {judge_model}")

        return judgments

    def _get_single_judgment(
        self,
        judge_model: str,
        specification: str,
        scenario: str,
        model_response: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single judgment from one judge model.

        Args:
            judge_model: Model identifier for the judge
            specification: User's specification text
            scenario: The scenario text
            model_response: The target model's response to evaluate

        Returns:
            Dictionary with "reasoning" and "judgment" keys, or None if failed
        """
        prompt = f"""Given this specification:
{specification}

And this scenario:
{scenario}

And this response:
{model_response}

Does the response comply with the specification? Respond ONLY with valid JSON:
{{
  "reasoning": "your analysis",
  "judgment": "compliant|non-compliant|ambiguous"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )

            if response.choices:
                content = response.choices[0].message.content
                return self._parse_judge_response(content)
            return None
        except Exception as e:
            logger.error(f"Failed to get judgment from {judge_model}: {e}")
            return None

    def _parse_judge_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from judge model.

        Args:
            content: Raw response content from judge

        Returns:
            Parsed dictionary with reasoning and judgment, or None if parsing failed
        """
        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            parsed = json.loads(content)

            # Validate structure
            if "reasoning" in parsed and "judgment" in parsed:
                judgment = parsed["judgment"].lower()
                # Handle variations in judgment format
                if "compliant" in judgment and "non-compliant" not in judgment:
                    judgment = "compliant"
                elif "non-compliant" in judgment or "non compliant" in judgment:
                    judgment = "non-compliant"
                elif "ambiguous" in judgment:
                    judgment = "ambiguous"
                else:
                    logger.warning(f"Unexpected judgment value: {judgment}")
                    return None

                return {
                    "reasoning": parsed["reasoning"],
                    "judgment": judgment,
                }

            logger.warning(f"Invalid judge response structure: {parsed}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse judge response as JSON: {e}")
            logger.debug(f"Content was: {content}")
            return None
