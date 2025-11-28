"""Model pricing data for cost calculation.

Fetches pricing dynamically from OpenRouter API.
See: https://openrouter.ai/api/v1/models
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Cache for model pricing data
_pricing_cache: Optional[dict[str, tuple[float, float]]] = None


def fetch_pricing(api_key: Optional[str] = None) -> dict[str, tuple[float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Args:
        api_key: Optional API key (not required for this endpoint)

    Returns:
        Dictionary mapping model_id to (input_cost_per_token, output_cost_per_token)
    """
    global _pricing_cache

    if _pricing_cache is not None:
        return _pricing_cache

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        pricing = {}
        for model in data.get("data", []):
            model_id = model.get("id")
            model_pricing = model.get("pricing", {})

            # Pricing is per token (as strings)
            prompt_cost = float(model_pricing.get("prompt", "0") or "0")
            completion_cost = float(model_pricing.get("completion", "0") or "0")

            pricing[model_id] = (prompt_cost, completion_cost)

        _pricing_cache = pricing
        logger.info(f"Loaded pricing for {len(pricing)} models from OpenRouter API")
        return pricing

    except Exception as e:
        logger.warning(f"Failed to fetch pricing from OpenRouter API: {e}")
        return {}


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    api_key: Optional[str] = None,
) -> float:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        api_key: Optional API key for fetching pricing

    Returns:
        Cost in USD
    """
    pricing = fetch_pricing(api_key)

    if model_id not in pricing:
        # Try without provider prefix variations
        logger.debug(f"No pricing found for {model_id}, using zero cost")
        return 0.0

    input_cost_per_token, output_cost_per_token = pricing[model_id]

    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token

    return input_cost + output_cost


def get_model_pricing(model_id: str, api_key: Optional[str] = None) -> tuple[float, float]:
    """
    Get the pricing for a model.

    Args:
        model_id: The model identifier
        api_key: Optional API key for fetching pricing

    Returns:
        Tuple of (input_cost_per_token, output_cost_per_token) in USD
    """
    pricing = fetch_pricing(api_key)
    return pricing.get(model_id, (0.0, 0.0))


def clear_pricing_cache():
    """Clear the pricing cache to force a fresh fetch."""
    global _pricing_cache
    _pricing_cache = None
